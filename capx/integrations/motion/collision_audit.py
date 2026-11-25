"""Configured-robot collision refinement using mature FCL, no task knowledge."""
from pathlib import Path
import numpy as np


def fixed_assembly_pairs(urdf):
    """Meshes in one rigid assembly can overlap by design, independent of q."""
    parent={name:name for name in urdf.link_map}
    def find(x):
        while parent[x]!=x:x=parent[x]
        return x
    for joint in urdf.joint_map.values():
        if joint.type=='fixed':parent[find(joint.child)]=find(joint.parent)
    names=list(parent)
    return tuple((a,b) for i,a in enumerate(names) for b in names[i+1:] if find(a)==find(b))


def refine_sphere_contacts(urdf_path,coll,path,distances):
    """Refine conservative sphere overlaps with the URDF collision meshes.

    Does not ignore a pair merely because it already overlaps at the start.
    Optional FCL lives in a separate installation; unavailable => UNKNOWN.
    """
    try:import fcl
    except ImportError:return dict(status='UNKNOWN',reason='FCL narrow phase unavailable')
    import yourdfpy,trimesh
    from capx.utils.robot_urdf import _filename_handler_for_urdf
    urdf_path=Path(urdf_path)
    u=yourdfpy.URDF.load(urdf_path,build_collision_scene_graph=True,load_collision_meshes=True,
        filename_handler=_filename_handler_for_urdf(urdf_path))
    scene=u.collision_scene;owners={}
    for node in scene.graph.nodes_geometry:
        chain=scene.graph.transforms.shortest_path(u.base_link,node)
        owners[node]=next(name for name in reversed(chain) if name in u.link_map)
    def mesh(name):
        parts=[]
        for node in scene.graph.nodes_geometry:
            if owners[node]==name:
                t,k=scene.graph[node];m=scene.geometry[k].copy();m.apply_transform(t);parts.append(m)
        if not parts:raise ValueError('Missing collision mesh for '+name)
        m=trimesh.util.concatenate(parts);v=fcl.BVHModel();v.beginModel(len(m.vertices),len(m.faces))
        v.addSubModel(np.asarray(m.vertices),np.asarray(m.faces,dtype=np.int32));v.endModel();return fcl.CollisionObject(v)
    pairs={}
    for i,j in zip(*np.where(distances<.004)):
        names=tuple(coll.link_names[int(coll._geom_to_link_idx[g])] for g in (coll.active_idx_i[j],coll.active_idx_j[j]))
        pairs.setdefault(int(i),set()).add(names)
    checked=[]
    for index,names in pairs.items():
        u.update_cfg(path[index]);cache={}
        for a,b in names:
            try:
                for name in (a,b):
                    if name not in cache:cache[name]=mesh(name)
            except ValueError as exc:return dict(status='UNKNOWN',reason=str(exc),checked=checked)
            hit=fcl.collide(cache[a],cache[b],fcl.CollisionRequest(),fcl.CollisionResult())
            distance=fcl.distance(cache[a],cache[b],fcl.DistanceRequest(),fcl.DistanceResult())
            checked.append(dict(sample=index,links=[a,b],collision=bool(hit),mesh_distance_m=float(distance)))
            if hit:return dict(status='FAIL',reason='mesh_self_collision',checked=checked)
    return dict(status='PASS',scope='configured_URDF_collision_mesh_samples',checked=checked,
        limitation='not continuous proof; actual configured gripper and environment need separate checks')
