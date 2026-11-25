"""Offline public-robot collision audit: PyRoKi sphere pairs vs FCL meshes."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    from capx.utils.robot_urdf import load_urdf_for_robot,_filename_handler_for_urdf
    from capx.serving.launch_pyroki_server import set_min_distance_from_limits
    import pyroki as pk,jax.numpy as jnp,yourdfpy,fcl
    cal=json.loads((a.episode/'calibration.json').read_text());urdf=Path(next(x['path'] for x in cal['sources'] if x['role']=='ik_urdf'))
    row=next(json.loads(x) for x in (a.episode/'trace.jsonl').read_text().splitlines() if json.loads(x).get('kind')=='shared_trajectory_plan')
    u=set_min_distance_from_limits(load_urdf_for_robot(str(urdf)));robot=pk.Robot.from_urdf(u)
    coll=pk.collision.RobotCollision.from_sphere_decomposition(json.loads((Path(__file__).resolve().parents[2]/'capx/serving/assets/panda_spheres.json').read_text()),u)
    path=np.array(row['trajectory']);dist=np.asarray(coll.compute_self_collision_distance(robot,jnp.array(path)))
    pairs={}
    for i,j in zip(*np.where(dist<0)):
        gi,gj=coll.active_idx_i[j],coll.active_idx_j[j]
        names=tuple(coll.link_names[int(coll._geom_to_link_idx[g])] for g in (gi,gj))
        pairs.setdefault(names,[]).append((int(i),float(dist[i,j])))
    exact=yourdfpy.URDF.load(urdf,build_collision_scene_graph=True,load_collision_meshes=True,filename_handler=_filename_handler_for_urdf(urdf))
    scene=exact.collision_scene;owners={}
    for node in scene.graph.nodes_geometry:
        chain=scene.graph.transforms.shortest_path(exact.base_link,node)
        owners[node]=next(name for name in reversed(chain) if name in exact.link_map)
    results=[]
    def mesh(name):
        import trimesh
        parts=[]
        for node in scene.graph.nodes_geometry:
            if owners[node]!=name:continue
            t,k=scene.graph[node];m=scene.geometry[k].copy();m.apply_transform(t);parts.append(m)
        if not parts:raise ValueError('No exact collision mesh for '+name)
        m=trimesh.util.concatenate(parts);v=fcl.BVHModel();v.beginModel(len(m.vertices),len(m.faces));v.addSubModel(np.asarray(m.vertices),np.asarray(m.faces,dtype=np.int32));v.endModel();return fcl.CollisionObject(v)
    for names,values in pairs.items():
        index,min_sphere=min(values,key=lambda v:v[1]);exact.update_cfg(path[index]);x,y=map(mesh,names)
        result=fcl.CollisionResult();hit=fcl.collide(x,y,fcl.CollisionRequest(),result)
        distance=fcl.distance(x,y,fcl.DistanceRequest(),fcl.DistanceResult())
        results.append(dict(links=names,index=index,sphere_signed_distance_m=min_sphere,mesh_collision=bool(hit),mesh_distance_m=float(distance)))
    a.output.write_text(json.dumps(dict(mode='public_configured_robot_OFFLINE_geometry_diagnostic',initial_min_sphere_m=float(dist[0].min()),pairs=results,permission=False),indent=2));print(json.dumps(results))

if __name__=='__main__':main()
