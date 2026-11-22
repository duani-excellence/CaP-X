"""Known-robot visual-mesh depth agreement from PUBLIC encoders and calibration.

This is conditional same-camera self segmentation, not an obstacle detector.
Only points matching the FIRST robot-mesh intersection on their camera ray can
receive robot identity. Unmatched points are retained, including points inside a
robot volume or behind a visible robot surface. Named-object masks always win.
No simulator, scene truth, learned model, network lookup or large-box erasure.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def _vector(value,n):
    a=np.asarray(value,float)
    if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError(f'Expected finite vector of length {n}')
    return a


def _se3(value):
    a=np.asarray(value,float)
    if (a.shape!=(4,4) or not np.isfinite(a).all() or not np.allclose(a[3],[0,0,0,1],atol=1e-9)
        or not np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(a[:3,:3]),1)):
        raise ValueError('Expected rigid calibration transform')
    return a


def _mask(value,n):
    if value is None:return np.zeros(n,bool)
    a=np.asarray(value)
    if a.shape!=(n,) or a.dtype.kind!='b':raise ValueError('Protected mask must be boolean and exactly align with points')
    return a


def classify_mesh_depth(points,camera_origin,mesh,*,protected_mask=None,tolerance_m=.002):
    """Classify current point samples by robot first-ray surface depth agreement.

    `points` and camera origin share the public robot-base frame. Return mask
    aligns exactly with input; callers must retain every False point. Tolerance
    is an explicit small geometric heuristic, not a calibrated uncertainty.
    """
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=3:raise ValueError('Expected Nx3 points')
    camera=_vector(camera_origin,3);protected=_mask(protected_mask,len(p))
    if not np.isfinite(tolerance_m) or not 0<tolerance_m<=.003:raise ValueError('Surface tolerance must be positive and at most 3 mm')
    mask=np.zeros(len(p),bool);delta=p-camera;depth=np.linalg.norm(delta,axis=1)
    valid=np.isfinite(p).all(axis=1)&(depth>1e-8)&~protected;ids=np.flatnonzero(valid)
    expected=np.full(len(p),np.nan)
    if len(ids) and len(mesh.faces):
        # Chunking bounds rtree candidate allocations for large RGB-D frames.
        for start in range(0,len(ids),128):
            batch=ids[start:start+128];directions=delta[batch]/depth[batch,None]
            locations,ray_ids,_=mesh.ray.intersects_location(np.repeat(camera[None],len(batch),axis=0),directions,multiple_hits=False)
            if len(ray_ids):expected[batch[ray_ids]]=np.linalg.norm(locations-camera,axis=1)
        mask=valid&np.isfinite(expected)&(np.abs(expected-depth)<=tolerance_m)
    return {'robot_mask':mask,'evidence':{
        'kind':'known_robot_mesh_first_ray_depth_agreement','status':'PASS' if mask.any() else 'UNCERTAIN',
        'input_points':len(p),'robot_points':int(mask.sum()),'retained_points':int((~mask).sum()),
        'protected_points':int(protected.sum()),'ray_intersections':int(np.isfinite(expected).sum()),
        'surface_tolerance_m':float(tolerance_m),'independent_sensor':False,
        'assumptions':['configured robot visual meshes match physical robot','current encoder FK and calibrated camera are accurate'],
        'limitation':'same-camera known-robot surface agreement; an unmasked foreign surface within tolerance is geometrically ambiguous; retained points remain unclassified'}}


class PandaSelfGeometry:
    """Parent-owned calibrated Panda mesh producer; actor cannot supply geometry.

    Both independent finger encoders are required to segment finger meshes.
    Without them, arm and hand meshes are available but BOTH fingers are omitted.
    The loaded URDF instance's mimic joint is disabled only in memory so actual
    asymmetric public finger positions can be represented. No file is changed.
    """
    def __init__(self,urdf_path,calibration):
        p=Path(urdf_path).expanduser()
        if not p.is_file():raise FileNotFoundError('An existing configured local robot URDF is required; network fallback is forbidden')
        self.urdf_path=p.resolve();self.urdf_hash=hashlib.sha256(p.read_bytes()).hexdigest()
        c=dict(calibration);digest=c.pop('calibration_hash',None)
        if digest!=hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest():raise ValueError('Robot calibration hash mismatch')
        if c.get('schema')!='capx.public_robot_calibration.v1' or c.get('ik_urdf_sha256')!=self.urdf_hash:raise ValueError('Robot calibration does not match configured URDF')
        self.calibration_hash=digest;self.public_ik=np.linalg.inv(_se3(c['T_ik_base_public_base']))
        self.actual_ik_hand=_se3(c['T_actual_hand_ik_hand']);self.tcp_offset=_vector(c['actual_tcp_offset_in_hand_m'],3)
        from capx.utils.robot_urdf import load_urdf_for_robot, _filename_handler_for_urdf
        self.urdf=load_urdf_for_robot(str(p))
        resolver=_filename_handler_for_urdf(self.urdf_path)
        assets={str(self.urdf_path):{'path':str(self.urdf_path),'sha256':self.urdf_hash,'kind':'urdf'}}
        for link in self.urdf.robot.links:
            for visual in link.visuals:
                source=visual.geometry.mesh
                if source is None:continue
                mesh_path=Path(resolver(source.filename)).resolve()
                if not mesh_path.is_file():raise FileNotFoundError('Loaded visual mesh source is unavailable for asset closure')
                row=assets.setdefault(str(mesh_path),{'path':str(mesh_path),'sha256':hashlib.sha256(mesh_path.read_bytes()).hexdigest(),'kind':'visual_mesh','links':[]})
                row['links'].append(link.name)
        grippers=[row for row in c.get('sources',[]) if row.get('role')=='gripper_mjcf']
        if len(grippers)>1:raise ValueError('Ambiguous configured gripper geometry')
        self.gripper_visual=_ConfiguredGripper(grippers[0]) if grippers else None
        if self.gripper_visual:
            for row in self.gripper_visual.sources:assets[row['path']]=row
        self.sources=[assets[path] for path in sorted(assets)]
        self.metadata={'asset_sources':self.sources,'geometry_asset_hash':hashlib.sha256(json.dumps(self.sources,sort_keys=True,separators=(',', ':')).encode()).hexdigest(),
                       'scope':'URDF plus every loaded visual mesh file; primitive geometry is covered by URDF bytes'}
        self.arm_names=[f'panda_joint{i}' for i in range(1,8)];self.finger_names=['panda_finger_joint1','panda_finger_joint2']
        joints={x.name:x for x in self.urdf.robot.joints}
        if any(n not in joints for n in self.arm_names+self.finger_names):raise ValueError('Configured model is not the supported Panda chain')
        for name in self.finger_names:joints[name].mimic=None
        self._mesh_links={}
        for link in self.urdf.robot.links:
            for node in self.urdf.scene.graph.nodes_geometry:
                # Scene graph shortest paths reveal the owning link without
                # naming mesh files or relying on their arbitrary suffixes.
                path=self.urdf.scene.graph.transforms.shortest_path(self.urdf.base_link,node)
                if link.name in path:self._mesh_links[node]=link.name

    def mesh(self,arm_joints,*,finger_positions_m=None,observed_tcp=None):
        import trimesh
        q=_vector(arm_joints,7);cfg=dict(zip(self.arm_names,q));excluded=[]
        if finger_positions_m is None:
            fingers=np.zeros(2);excluded=['panda_leftfinger','panda_rightfinger']
        else:
            fingers=_vector(finger_positions_m,2)
            if np.any(fingers<-.001) or np.any(fingers>.041):raise ValueError('Public finger positions outside calibrated Panda range')
        cfg.update(dict(zip(self.finger_names,fingers)));self.urdf.update_cfg(cfg)
        ik_hand=self.urdf.get_transform('panda_hand');actual_hand=self.public_ik@ik_hand@np.linalg.inv(self.actual_ik_hand)
        # Preserve the independent joint-FK TCP for the consistency gate. Only
        # actual configured gripper visuals use the public current TCP signal;
        # arm geometry remains at encoder FK and legacy URDF visuals stay legacy.
        fk_tcp=actual_hand.copy();fk_tcp[:3,3]+=actual_hand[:3,:3]@self.tcp_offset
        gripper_hand=actual_hand.copy();gripper_pose_source='joint_fk_legacy_geometry'
        if self.gripper_visual:
            gripper_pose_source='joint_fk_configured_geometry'
            if observed_tcp is not None:
                from scipy.spatial.transform import Rotation
                tcp=np.asarray(observed_tcp,float)
                if tcp.ndim!=1 or len(tcp)<7 or not np.isfinite(tcp[:7]).all():raise ValueError('Expected current public TCP XYZ WXYZ')
                norm=np.linalg.norm(tcp[3:7])
                if abs(norm-1)>.02:raise ValueError('Expected unit public TCP quaternion')
                rotation=Rotation.from_quat(np.roll(tcp[3:7]/norm,-1)).as_matrix()
                gripper_hand[:3,:3]=rotation;gripper_hand[:3,3]=tcp[:3]-rotation@self.tcp_offset
                gripper_pose_source='current_public_tcp_plus_static_hand_offset'
        meshes=[];nodes=[]
        for node in self.urdf.scene.graph.nodes_geometry:
            link=self._mesh_links[node]
            if link in excluded:continue
            if self.gripper_visual and link in ('panda_hand','panda_leftfinger','panda_rightfinger'):continue
            t,name=self.urdf.scene.graph[node];part=self.urdf.scene.geometry[name].copy()
            if link in ('panda_hand','panda_leftfinger','panda_rightfinger'):
                public=actual_hand@np.linalg.inv(ik_hand)@t
            else:public=self.public_ik@t
            part.apply_transform(public);meshes.append(part);nodes.append(node)
        if self.gripper_visual:
            extra=self.gripper_visual.parts(gripper_hand,None if finger_positions_m is None else fingers)
            meshes.extend(extra);nodes.extend(['configured_gripper_visual']*len(extra))
        tcp=actual_hand.copy();tcp[:3,3]+=actual_hand[:3,:3]@self.tcp_offset
        return trimesh.util.concatenate(meshes),{'excluded_links':excluded,'finger_positions_m':None if finger_positions_m is None else fingers.tolist(),
            'configured_urdf_sha256':self.urdf_hash,'calibration_hash':self.calibration_hash,'visual_mesh_nodes':len(nodes),
            **self.metadata,'gripper_geometry_source':'configured_mjcf' if self.gripper_visual else 'legacy_ik_urdf',
            'public_fk_tcp':fk_tcp.tolist(),'public_gripper_hand':gripper_hand.tolist(),'gripper_pose_source':gripper_pose_source,'arm_pose_source':'current_joint_fk','finger_policy':'independent_current_encoders_required; no symmetric contact assumption'}

    def classify(self,points,camera_origin,arm_joints,*,finger_positions_m=None,observed_tcp=None,protected_mask=None,tolerance_m=.002):
        p=np.asarray(points,float)
        if p.ndim!=2 or p.shape[1]!=3:raise ValueError('Expected Nx3 points')
        protected=_mask(protected_mask,len(p))
        if observed_tcp is None:
            return {'robot_mask':np.zeros(len(p),bool),'evidence':{'status':'UNCERTAIN','reason':'public_tcp_required_for_current_fk_validation'}}
        tcp=np.asarray(observed_tcp,float)
        if tcp.ndim!=1 or len(tcp)<7 or not np.isfinite(tcp[:7]).all():raise ValueError('Expected current public TCP XYZ WXYZ')
        quaternion=tcp[3:7];norm=np.linalg.norm(quaternion)
        if abs(norm-1)>.02:raise ValueError('Expected unit public TCP quaternion')
        from scipy.spatial.transform import Rotation
        r=Rotation.from_quat(np.roll(quaternion/norm,-1)).as_matrix();mesh,meta=self.mesh(arm_joints,finger_positions_m=finger_positions_m,observed_tcp=tcp)
        fk=np.asarray(meta.pop('public_fk_tcp'));distance=float(np.linalg.norm(fk[:3,3]-tcp[:3]));angle=float(Rotation.from_matrix(fk[:3,:3]@r.T).magnitude())
        meta.update(fk_public_tcp_position_error_m=distance,fk_public_tcp_rotation_error_rad=angle)
        if distance>.003 or angle>np.deg2rad(1):
            return {'robot_mask':np.zeros(len(p),bool),'evidence':{**meta,'status':'UNCERTAIN','reason':'public_joint_fk_tcp_inconsistent'}}
        result=classify_mesh_depth(p,camera_origin,mesh,protected_mask=protected,tolerance_m=tolerance_m)
        result['evidence'].update(meta);return result


class _ConfiguredGripper:
    """Actual configured static gripper visuals plus independently read slides.

    Reads only hashed robot configuration assets. Replaces the IK model hand and
    fingers whose visual mesh/joint origin need not match the deployed gripper.
    Collision boxes are not invented as visible surfaces. Unknown MJCF features
    fail closed; no runtime scene model or object truth is read.
    """
    def __init__(self,source):
        import xml.etree.ElementTree as ET
        import trimesh
        path=Path(source['path']).resolve()
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=source['sha256']:raise ValueError('Configured gripper MJCF source drift')
        root=ET.fromstring(raw)
        if root.find('default') is not None or root.find('include') is not None:raise ValueError('Unsupported gripper MJCF defaults/includes')
        assets={x.get('name'):x for x in root.findall('./asset/mesh')}
        self.sources=[{'path':str(path),'sha256':source['sha256'],'kind':'configured_gripper_mjcf'}]
        self._pieces=[];self._assets={}
        def transform(element):
            from .control_frames import _rotation
            if any(k in element.attrib for k in ('euler','axisangle','xyaxes','zaxis','childclass','class')):raise ValueError('Unsupported gripper geometry orientation/defaults')
            t=np.eye(4);t[:3,3]=_vector(np.fromstring(element.get('pos','0 0 0'),sep=' '),3)
            t[:3,:3]=_rotation(np.fromstring(element.get('quat','1 0 0 0'),sep=' '));return t
        def walk(body,chain):
            step={'transform':transform(body),'joints':[]}
            for joint in body.findall('joint'):
                name=joint.get('name')
                if name not in ('finger_joint1','finger_joint2') or joint.get('type')!='slide':raise ValueError('Unsupported configured gripper joint')
                step['joints'].append((0 if name=='finger_joint1' else 1,_vector(np.fromstring(joint.get('axis','0 0 1'),sep=' '),3)))
            chain=chain+[step]
            for geom in body.findall('geom'):
                if geom.get('group')!='1':continue
                if geom.get('type')!='mesh':raise ValueError('Unsupported configured gripper visual primitive')
                asset=assets[geom.get('mesh')];file=(path.parent/asset.get('file')).resolve()
                if str(file) not in self._assets:
                    part=trimesh.load_mesh(file,force='mesh');part.apply_scale(_vector(np.fromstring(asset.get('scale','1 1 1'),sep=' '),3))
                    self._assets[str(file)]=part;self.sources.append({'path':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'kind':'configured_gripper_visual_mesh'})
                self._pieces.append((chain,transform(geom),self._assets[str(file)]))
            for child in body.findall('body'):walk(child,chain)
        bodies=root.findall('./worldbody/body')
        if len(bodies)!=1:raise ValueError('Expected one configured gripper root')
        walk(bodies[0],[])
        if not self._pieces:raise ValueError('Configured gripper has no visible meshes')

    def parts(self,actual_hand,fingers):
        result=[]
        for chain,offset,source in self._pieces:
            if fingers is None and any(step['joints'] for step in chain):continue
            t=actual_hand.copy()
            for step in chain:
                t=t@step['transform']
                for index,axis in step['joints']:
                    slide=np.eye(4);slide[:3,3]=axis*float(fingers[index])*(1 if index==0 else -1)
                    t=t@slide
            part=source.copy();part.apply_transform(t@offset);result.append(part)
        return result
