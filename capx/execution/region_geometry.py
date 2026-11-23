"""Public geometric primitives for upright objects and planar receptacles.

Only calibrated point clouds enter this module. Visible extents are NOT a pose
error certificate; interval admission requires a separate qualified error model.
No task identities, simulator objects, scores or private fixture coordinates.
"""
import numpy as np
from scipy.spatial import ConvexHull


def instance_masks(rows,minimum_score=.5):
    groups=[]
    for row in sorted(rows,key=lambda r:r['score'],reverse=True):
        mask=np.asarray(row['mask'],bool)
        if not np.isfinite(row['score']) or row['score']<minimum_score or not mask.any():continue
        for group in groups:
            common=np.count_nonzero(mask&group['mask'])
            if common/max(1,min(mask.sum(),group['mask'].sum()))>=.85:
                group['mask'] |= mask;group['scores'].append(float(row['score']));break
        else:groups.append(dict(mask=mask.copy(),scores=[float(row['score'])]))
    return groups


def deproject(camera):
    depth=np.asarray(camera['images']['depth']).squeeze();k=np.asarray(camera['intrinsics']);t=np.asarray(camera['pose_mat'])
    y,x=np.indices(depth.shape)
    return np.stack([(x-k[0,2])*depth/k[0,0],(y-k[1,2])*depth/k[1,1],depth],axis=-1)@t[:3,:3].T+t[:3,3]


def horizontal_pixels(xyz):
    return axis_pixels(xyz,2)


def axis_pixels(xyz,axis):
    normal=np.cross(xyz[1:-1,2:]-xyz[1:-1,:-2],xyz[2:,1:-1]-xyz[:-2,1:-1])
    norm=np.linalg.norm(normal,axis=-1);good=np.isfinite(normal).all(-1)&(norm>1e-12)
    mask=np.zeros(xyz.shape[:2],bool)
    mask[1:-1,1:-1]=good&(abs(normal[:,:,axis])/np.maximum(norm,1e-12)>.98)
    return mask


def receptacle_geometry(floor_points,walls):
    """Measure orthogonal wall planes, rather than extrapolate hidden floor.

    Plane selection requires broad support in BOTH other dimensions. These are
    measured surfaces, not certified inner free-space boundaries: wall thickness
    and object uncertainty must still be included by the motion contract.
    """
    fit=horizontal_region(floor_points)
    if fit['status']!='PASS':return fit
    planes=[]
    for axis in (0,1):
        candidates=[_points(p) for a,p in walls if a==axis]
        if not candidates:return dict(status='UNKNOWN',reason='missing_observed_wall_axis')
        p=np.concatenate(candidates);p=p[(p[:,2]>fit['floor_z']+.004)&(p[:,2]<fit['floor_z']+.2)]
        values,counts=np.unique(np.round(p[:,axis]/.001).astype(int),return_counts=True)
        selected=[]
        for value in values[counts>=100]*.001:
            q=p[abs(p[:,axis]-value)<.0015]
            if np.ptp(q[:,1-axis])<.08 or np.ptp(q[:,2])<.015:continue
            center=float(np.median(q[:,axis]))
            if not any(abs(center-v)<.004 for v in selected):selected.append(center)
        selected.sort()
        if len(selected)<2 or selected[-1]-selected[0]<.1:
            return dict(status='UNKNOWN',reason='missing_bounded_wall_pair',axis=axis)
        planes.append(selected)
    return dict(fit,floor_visible_lo=fit['lo'],floor_visible_hi=fit['hi'],
                lo=[v[0] for v in planes],hi=[v[-1] for v in planes],
                center=[*[(v[0]+v[-1])/2 for v in planes],fit['floor_z']],
                wall_planes=planes,source='current_multiview_floor_and_wall_planes',
                uncertainty='unvalidated_measured_wall_planes')


def _points(value):
    p=np.asarray(value,float)
    if p.ndim!=2 or p.shape[1]!=3:raise ValueError('Nx3 public points required')
    return p[np.isfinite(p).all(axis=1)]


def foreground_component(points):
    """Remove small depth-disconnected mask leakage for instance fitting only.

    Removed points are NOT labelled free space and cannot clear motion paths.
    Two substantial connected components remain ambiguous.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    p=_points(points)
    voxels,inverse=np.unique(np.round(p/.001).astype(int),axis=0,return_inverse=True)
    pairs=cKDTree(voxels*.001).query_pairs(.007,output_type='ndarray')
    matrix=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(voxels),len(voxels)))
    _,label=connected_components(matrix,directed=False);point_labels=label[inverse]
    values,counts=np.unique(point_labels,return_counts=True);best=values[np.argmax(counts)]
    keep=point_labels==best
    if keep.mean()<.8:raise ValueError('ambiguous disconnected instance components')
    return p[keep],dict(retained_points=int(keep.sum()),discarded_points=int((~keep).sum()),
                        discarded_points_are_not_free_space=True)


def horizontal_region(points):
    p=_points(points)
    if len(p)<300:return dict(status='UNKNOWN',reason='insufficient_planar_evidence')
    # Dominant horizontal plane, followed by a residual/area screen. Interior
    # quantiles produce an inscribed rectangle; they do not extrapolate an edge.
    values,counts=np.unique(np.round(p[:,2]/.001).astype(int),return_counts=True)
    z=float(values[np.argmax(counts)]*.001);q=p[abs(p[:,2]-z)<.0015]
    if len(q)<300:return dict(status='UNKNOWN',reason='missing_horizontal_plane')
    lo,hi=np.quantile(q[:,:2],[.01,.99],axis=0)
    if np.any(hi-lo<.1):return dict(status='UNKNOWN',reason='plane_too_small_for_receptacle')
    z=float(np.median(q[:,2]));residual=float(np.quantile(abs(q[:,2]-z),.99))
    return dict(status='PASS',floor_z=z,lo=lo.tolist(),hi=hi.tolist(),center=[*((lo+hi)/2),z],
                residual_z=residual,points=len(q),frame='robot_base',
                assumptions=['horizontal axis-aligned rectangular receptacle','observed floor extrema'],
                source='current_segmented_RGBD_horizontal_plane',uncertainty='unvalidated_visible_extent')


def region_cells(region,grid):
    nx,ny=grid
    if region.get('status')!='PASS' or nx<1 or ny<1:raise ValueError('Qualified floor and positive grid required')
    lo=np.asarray(region['lo']);hi=np.asarray(region['hi']);delta=(hi-lo)/[nx,ny]
    result=[]
    for y in range(ny):
        for x in range(nx):
            a=lo+delta*[x,y];b=a+delta
            result.append(dict(lo=a.tolist(),hi=b.tolist(),center=[*((a+b)/2),region['floor_z']],
                               floor_z=region['floor_z'],grid_cell=[x,y],source=region.get('source')))
    return result


def same_region(a,b):
    intersection=np.maximum(0,np.minimum(a['hi'],b['hi'])-np.maximum(a['lo'],b['lo']))
    area=min(np.prod(np.subtract(a['hi'],a['lo'])),np.prod(np.subtract(b['hi'],b['lo'])))
    return bool(area>0 and np.prod(intersection)/area>.85 and abs(a['floor_z']-b['floor_z'])<.003)


def upright_shape(points,floor_z):
    p=_points(points)
    if len(p)<30:raise ValueError('Insufficient object geometry')
    q=p[p[:,2]>floor_z+.003]
    if len(q)<30:raise ValueError('No measured object above support')
    xy=q[:,:2];hull=xy[ConvexHull(xy).vertices];edges=np.roll(hull,-1,axis=0)-hull
    candidates=[]
    for angle in np.unique(np.arctan2(edges[:,1],edges[:,0]) % (np.pi/2)):
        u=np.array([np.cos(angle),np.sin(angle)]);v=np.array([-u[1],u[0]])
        basis=np.column_stack([u,v]);local=xy@basis;lo,hi=np.quantile(local,[.005,.995],axis=0)
        candidates.append((np.prod(hi-lo),basis,lo,hi))
    _,basis,lo,hi=min(candidates,key=lambda x:x[0]);ext=hi-lo;center=basis@((lo+hi)/2)
    top=float(np.quantile(q[:,2],.995));height=top-floor_z
    return dict(center=[*center,floor_z+height/2],height_m=height,bottom_z=float(floor_z),top_z=top,
                basis_xy=basis.tolist(),extent_xy=ext.tolist(),half_extent_xy=(np.abs(basis)@(ext/2)).tolist(),
                frame='robot_base',source='current_multiview_visible_upright_extent',
                uncertainty='unvalidated_visible_extent',point_count=len(q))


def region_permission(shape,region,*,xy_error,z_error,edge_margin=.008,max_gap=.025):
    if xy_error is None or z_error is None:return dict(status='UNKNOWN',reason='missing_qualified_error_model')
    if not np.isfinite([xy_error,z_error]).all() or min(xy_error,z_error)<0:raise ValueError('Invalid interval')
    center=np.asarray(shape['center'])[:2];half=np.asarray(shape['half_extent_xy'])
    lo=center-half-xy_error;hi=center+half+xy_error
    gap=shape['bottom_z']-region['floor_z'];interval=[gap-z_error,gap+z_error]
    contained=bool(np.all(lo>=np.asarray(region['lo'])+edge_margin) and np.all(hi<=np.asarray(region['hi'])-edge_margin))
    safe=contained and interval[0]>=0 and interval[1]<=max_gap
    return dict(status='PASS' if safe else 'FAIL',reason='region_interval_contained' if safe else 'region_or_drop_interval_outside_contract',
                contained=contained,gap_interval_m=interval,object_interval_xy=[lo.tolist(),hi.tolist()],
                edge_margin_m=edge_margin,max_gap_m=max_gap)
