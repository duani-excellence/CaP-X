"""Timestamped wrist sensor, robot-only compensation and bounded evidence memory.

MuJoCo force/torque are internal reaction wrenches; external wrench is -raw.
No object mass, object pose, contacts or task score is read by this adapter.
"""
from collections import deque
from typing import Protocol
import numpy as np


def transform_wrench(wrench, rotation, source_origin_in_target):
    w=np.asarray(wrench,float);r=np.asarray(rotation,float);p=np.asarray(source_origin_in_target,float)
    if w.shape!=(6,) or r.shape!=(3,3) or p.shape!=(3,) or not np.isfinite(np.r_[w,r.ravel(),p]).all():raise ValueError('Invalid wrench transform')
    f=r@w[:3];return np.r_[f,r@w[3:]+np.cross(p,f)]


def compensate_gravity(raw,tool_gravity,bias=None):
    return -(np.asarray(raw,float)-np.asarray(bias if bias is not None else np.zeros(6)))-np.asarray(tool_gravity,float)


class FTAdapter(Protocol):
    def sample(self) -> dict: ...


class ConfiguredHardwareFT:
    """A caller-owned reader only. No SDK import, connection or guessed units.

    BOX get_sensor_cache is NOT assumed to return a particular layout. A verified
    decode callback, device identity, units, clock mapping and calibration are
    required before this adapter can produce a valid sample.
    """
    def __init__(self,reader,decode,calibration):
        required={'sensor_identity','driver_revision','units','clock_mapping','bias_sensor','sensor_to_tcp','tool_mass_kg','tool_com_sensor_m'}
        if not required<=calibration.keys() or calibration['units']!=['N','Nm']:raise ValueError('Verified hardware identity/units/calibration required')
        self.reader=reader;self.decode=decode;self.calibration=calibration
    def sample(self):
        row=self.decode(self.reader(),self.calibration)
        if not {'timestamp','raw_sensor','valid','source','frame'}<=row.keys():raise ValueError('Invalid hardware decoder result')
        if not np.isfinite(np.r_[row['timestamp'],row['raw_sensor']]).all():raise ValueError('Nonfinite hardware reading')
        return row


class ForceWindow:
    """Short, versionable public evidence, never a force-closure certificate."""
    def __init__(self,force_error_bound_n=.005):
        if not np.isfinite(force_error_bound_n) or force_error_bound_n<=0:raise ValueError('Positive calibrated force error bound required')
        self.error=float(force_error_bound_n);self.rows=deque(maxlen=200)
    def append(self,row):
        if not np.isfinite(np.r_[row['timestamp'],row['external_base'],row['gripper_width_m']]).all():raise ValueError('Nonfinite force sample')
        if self.rows and row['timestamp']<self.rows[-1]['timestamp']:raise ValueError('Sensor clock moved backwards')
        if self.rows and row['timestamp']==self.rows[-1]['timestamp']:return
        self.rows.append(row)
    def assess(self,now,span=.3):
        rows=[x for x in self.rows if now-span-1e-8<=x['timestamp']<=now]
        out=dict(status='UNKNOWN',reason='missing_fresh_quasistatic_window',hold_permission=False,release_permission=False,
                 samples=len(rows),timestamp=now,expires_at=now+.15)
        if len(rows)<4 or now-rows[-1]['timestamp']>.15 or not all(x['valid'] and x['quasistatic'] for x in rows):return out
        w=np.array([x['external_base'] for x in rows]);jaw=np.array([x['gripper_width_m'] for x in rows])
        load=float(-np.median(w[:,2]));spread=float(np.max(np.linalg.norm(w[:,:3]-np.median(w[:,:3],axis=0),axis=1)))
        torque=float(np.max(np.linalg.norm(w[:,3:],axis=1)));drift=float(np.ptp(jaw))
        uncertainty=self.error+spread
        out.update(load_n=load,load_interval_n=[load-uncertainty,load+uncertainty],force_error_bound_n=self.error,
                   force_spread_n=spread,torque_nm=torque,jaw_drift_m=drift,
                   status='PASS' if load-uncertainty>2*self.error and load+uncertainty<15 and spread<.3 and drift<.0005 and torque<2 else 'UNKNOWN',
                   reason='load_and_jaw_window_only',source_ids=[x['sequence'] for x in rows if 'sequence' in x])
        return out


class MujocoWristFT:
    def __init__(self,low,config=None,emit=None):
        import mujoco
        self.low=low;self.config=config or {};self.emit=emit or (lambda row:None)
        self.window=ForceWindow(self.config.get('force_error_bound_n',.005))
        self.model=low.robosuite_env.sim.model;self.data=low.robosuite_env.sim.data
        m=self.model;self.ids=[i for i in range(m.nsensor) if m.sensor_id2name(i).endswith(('force_ee','torque_ee'))]
        if len(self.ids)!=2:raise ValueError('Exactly one wrist force/torque pair required')
        self.ids.sort(key=lambda i:0 if m.sensor_id2name(i).endswith('force_ee') else 1)
        self.site=int(m.sensor_objid[self.ids[0]])
        if any(int(m.sensor_dim[i])!=3 or int(m.sensor_objid[i])!=self.site for i in self.ids):raise ValueError('Wrist sensor site mismatch')
        root=int(m.site_bodyid[self.site]);self.bodies=[]
        for i in range(m.nbody):
            b=i
            while b and b!=root:b=int(m.body_parentid[b])
            if b==root:self.bodies.append(i)
        # Robot model metadata only: reject accidental task-object attachment.
        if any(not m.body_id2name(i).startswith('gripper') for i in self.bodies):raise ValueError('Tool subtree contains non-gripper bodies')
        self.scratch=mujoco.MjData(m._model);self.previous=None;self.filtered=None;self.sequence=0
        self.rng=np.random.default_rng(self.config.get('noise_seed',0))
        self.bias=np.asarray(self.config.get('bias_sensor',[0]*6),float)
        self.noise=np.asarray(self.config.get('noise_std',[0]*6),float)
        if self.bias.shape!=(6,) or self.noise.shape!=(6,) or not np.isfinite(np.r_[self.bias,self.noise]).all() or (self.noise<0).any():raise ValueError('Six finite bias/noise channels required')
        if (self.noise[:3]*6>self.window.error).any():raise ValueError('Noise exceeds configured evidence error bound')
        self.calibration=dict(schema='capx.ft_calibration.v1',source='MuJoCo_native_force_torque',
            sensor_names=[m.sensor_id2name(i) for i in self.ids],sensor_site=m.site_id2name(self.site),
            tool_mass_kg=float(sum(m.body_mass[i] for i in self.bodies)),
            tool_bodies=[m.body_id2name(i) for i in self.bodies],bias_sensor=self.bias.tolist(),noise_std=self.noise.tolist(),
            bias_provenance='simulator configured zero; no hardware calibration claimed',
            sign='external = -internal_reaction - tool_gravity',timestamp='MuJoCo_sim_seconds',
            synchronization='mj_copyData + mj_forward in isolated scratch data',
            gravity_compensation='robot-only subtree inertials and joint FK',payload_compensation='unknown until measured; no object mass lookup',
            dynamic_compensation=False,filter_tau_seconds=.08,validity_seconds=.15,
            force_error_bound_n=self.window.error,error_bound_source=self.config.get('error_bound_source','unverified simulator test configuration'))
    def sample(self):
        import mujoco
        m=self.model;d=self.scratch;time=float(self.data.time)
        if self.previous is not None and time==self.previous['timestamp']:return self.previous
        mujoco.mj_copyData(d,m._model,self.data._data);mujoco.mj_forward(m._model,d)
        raw=np.concatenate([d.sensordata[int(m.sensor_adr[i]):int(m.sensor_adr[i])+3] for i in self.ids])
        r=d.site_xmat[self.site].reshape(3,3);origin=d.site_xpos[self.site];g=np.asarray(m.opt.gravity)
        fg=sum(m.body_mass[i]*g for i in self.bodies)
        tg=sum(np.cross(d.xipos[i]-origin,m.body_mass[i]*g) for i in self.bodies)
        tool=np.r_[r.T@fg,r.T@tg];measured=raw+self.bias+self.rng.normal(size=6)*self.noise
        ext=compensate_gravity(measured,tool,self.bias)
        base=self.low.base_link_idx;rb=d.xmat[base].reshape(3,3);hand=m.body_name2id('robot0_right_hand');rh=d.xmat[hand].reshape(3,3)
        tcp=d.xpos[hand]+rh@np.array([0,0,.107]);rtcp=rb.T@rh;ptcp=rb.T@(tcp-d.xpos[base])
        from scipy.spatial.transform import Rotation
        pose=np.r_[ptcp,Rotation.from_matrix(rtcp).as_quat()[[3,0,1,2]]]
        base_w=transform_wrench(ext,rb.T@r,rb.T@(origin-tcp));tcp_w=transform_wrench(ext,rh.T@r,rh.T@(origin-tcp))
        robot=self.low.robosuite_env.robots[0];qv=d.qvel[robot._ref_joint_vel_indexes]
        jaw=float(np.sum(np.abs(d.qpos[robot._ref_gripper_joint_pos_indexes['right']])))
        dt=time-self.previous['timestamp'] if self.previous else .05;alpha=1-np.exp(-max(dt,0)/.08)
        acceleration=(qv-np.asarray(self.previous['robot_joint_velocity']))/max(dt,1e-6) if self.previous else np.zeros_like(qv)
        self.filtered=base_w.copy() if self.filtered is None else self.filtered+alpha*(base_w-self.filtered)
        self.sequence+=1
        row=dict(schema='capx.wrist_ft.v1',sequence=self.sequence,timestamp=time,source='MuJoCo_native_wrist_sensor',
                 frame='robot_base_at_TCP',raw_frame=self.calibration['sensor_site'],units=['N','Nm'],
                 raw_sensor=measured.tolist(),tool_gravity_sensor=tool.tolist(),external_base=base_w.tolist(),external_tcp=tcp_w.tolist(),filtered_base=self.filtered.tolist(),
                 valid=bool(np.isfinite(base_w).all()),quasistatic=bool(np.max(abs(qv))<.08 and np.max(abs(acceleration))<.5),
                 robot_joint_velocity=qv.tolist(),robot_acceleration_estimate=acceleration.tolist(),
                 expires_at=time+.15,calibration_version=1,gripper_width_m=jaw,tcp=pose.tolist(),
                 payload_compensated=False,dynamic_compensated=False)
        self.previous=row;self.window.append(row);self.emit(row);return row
