import hashlib,json
import numpy as np
import pytest
import trimesh
from capx.execution.robot_self_geometry import PandaSelfGeometry
from test_robot_self_geometry import fake_urdf


def configured(tmp_path):
 p,c=fake_urdf(tmp_path);mesh=tmp_path/'visual.stl';trimesh.creation.box(extents=[.002,.002,.002]).export(mesh)
 xml=tmp_path/'gripper.xml';xml.write_text('<mujoco><asset><mesh name="v" file="visual.stl"/></asset><worldbody><body name="hand"><geom type="mesh" group="1" mesh="v"/><body name="left" pos="0 0 .0524"><joint name="finger_joint1" type="slide" axis="0 1 0"/><geom type="mesh" group="1" mesh="v"/></body><body name="right" pos="0 0 .0524"><joint name="finger_joint2" type="slide" axis="0 1 0"/><geom type="mesh" group="1" mesh="v"/></body></body></worldbody></mujoco>')
 c['sources']=[{'role':'gripper_mjcf','path':str(xml),'sha256':hashlib.sha256(xml.read_bytes()).hexdigest()}]
 c.pop('calibration_hash');c['calibration_hash']=hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 return p,c,xml,mesh


def test_current_configured_gripper_geometry_replaces_ik_finger_origin(tmp_path):
 p,c,xml,mesh=configured(tmp_path);g=PandaSelfGeometry(p,c);_,meta=g.mesh(np.zeros(7),finger_positions_m=[.01,.03])
 assert meta['gripper_geometry_source']=='configured_mjcf'
 parts=g.gripper_visual.parts(np.eye(4),np.array([.01,.03]));centers=np.array([x.centroid for x in parts])
 assert np.allclose(centers,[[0,0,0],[0,.01,.0524],[0,-.03,.0524]])
 assert {str(xml.resolve()),str(mesh.resolve())}<={r['path'] for r in g.sources}
 assert len(g.gripper_visual.parts(np.eye(4),None))==1


def test_configured_gripper_source_drift_rejected(tmp_path):
 p,c,xml,mesh=configured(tmp_path);xml.write_text(xml.read_text()+' ')
 with pytest.raises(ValueError,match='source drift'):PandaSelfGeometry(p,c)


def test_actual_visual_mesh_bytes_part_of_geometry_asset_hash(tmp_path):
 p,c,xml,mesh=configured(tmp_path);first=PandaSelfGeometry(p,c)
 trimesh.creation.box(extents=[.003,.002,.002]).export(mesh)
 second=PandaSelfGeometry(p,c)
 assert first.metadata['geometry_asset_hash']!=second.metadata['geometry_asset_hash']
