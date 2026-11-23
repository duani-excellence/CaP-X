"""Public surface features for flat ring + protruding handle insertion parts.

Feature identification only. An enclosed image void is NOT by itself a qualified
insertion clearance certificate; online control must confirm target and errors.
"""
import numpy as np
from scipy.ndimage import binary_closing,binary_fill_holes,label
from .region_geometry import upright_shape


def flat_ring_features(points,resolution=.0015):
    p=np.asarray(points,float);p=p[np.isfinite(p).all(1)]
    if len(p)<100:return dict(status='UNKNOWN',reason='insufficient_surface')
    top=float(np.quantile(p[:,2],.98));surface=p[abs(p[:,2]-top)<.002]
    lo=surface[:,:2].min(0)-resolution*4;idx=np.round((surface[:,:2]-lo)/resolution).astype(int)
    size=idx.max(0)+5
    if np.max(size)>500:return dict(status='UNKNOWN',reason='surface_exceeds_feature_domain')
    mask=np.zeros(tuple(size),bool);mask[idx[:,0],idx[:,1]]=1
    mask=binary_closing(mask,iterations=2);holes=binary_fill_holes(mask)&~mask;labels,n=label(holes)
    regions=[]
    for i in range(1,n+1):
        xy=np.argwhere(labels==i)*resolution+lo
        if len(xy)<50 or np.min(np.ptp(xy,axis=0))<.015:continue
        shape=upright_shape(np.c_[xy,np.ones(len(xy))*top],top-.01)
        regions.append(shape)
    if len(regions)!=1:return dict(status='UNKNOWN',reason='missing_or_ambiguous_enclosed_hole',candidates=len(regions))
    hole=regions[0];center=np.array(hole['center']);center[2]=top;basis=np.array(hole['basis_xy']);q=(surface[:,:2]-center[:2])@basis
    ext=np.c_[-q.min(0),q.max(0)];axis=int(np.argmax(abs(ext[:,0]-ext[:,1])));sign=1 if ext[axis,1]>ext[axis,0] else -1
    if abs(ext[axis,0]-ext[axis,1])<.012:return dict(status='UNKNOWN',reason='no_distinct_handle_protrusion')
    distance=ext[axis].mean();handle=center.copy();handle[:2]+=sign*distance*basis[:,axis]
    closing=basis[:,1-axis];rotation=np.c_[np.cross(np.r_[closing,0],[0,0,-1]),np.r_[closing,0],[0,0,-1]]
    from scipy.spatial.transform import Rotation
    return dict(status='PASS',hole_center=center.tolist(),hole_extent_xy=hole['extent_xy'],hole_basis_xy=basis.tolist(),
        handle_center=handle.tolist(),grasp_quaternion=np.roll(Rotation.from_matrix(rotation).as_quat(),1).tolist(),
        top_z=top,resolution_m=resolution,source='measured_planar_ring_void_and_asymmetric_protrusion',
        scope='feature_only_NOT_insertion_or_release_permission')
