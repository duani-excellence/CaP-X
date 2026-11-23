import numpy as np
import pytest
from capx.execution.region_geometry import instance_masks, horizontal_region, upright_shape, region_cells, region_permission


def test_nested_same_instance_masks_merge_but_distinct_instances_remain():
    a=np.zeros((30,30),bool);a[2:12,2:12]=1
    b=np.zeros_like(a);b[3:10,3:10]=1
    c=np.zeros_like(a);c[18:28,18:28]=1
    rows=[dict(mask=x,score=.8) for x in [a,b,c]]
    assert len(instance_masks(rows))==2
    assert len(instance_masks([dict(mask=a,score=.1)]))==0


def floor(x0=0.,y0=0.):
    x,y=np.meshgrid(np.linspace(x0,x0+.4,90),np.linspace(y0,y0+.5,100))
    return np.c_[x.ravel(),y.ravel(),np.ones(x.size)*.3]


def test_region_fitted_from_observations_not_private_coordinates():
    r=horizontal_region(floor(1.,2.))
    assert r['status']=='PASS' and abs(r['floor_z']-.3)<1e-6
    cells=region_cells(r,[2,2]);assert len(cells)==4
    assert cells[0]['center'][0]<cells[1]['center'][0]
    assert cells[0]['center'][1]<cells[2]['center'][1]


def test_missing_floor_cannot_be_certified():
    assert horizontal_region(np.zeros((10,3)))['status']=='UNKNOWN'


def test_full_interval_required_for_release():
    cell=region_cells(horizontal_region(floor()),[2,2])[0]
    center=np.array(cell['center']);center[2]=.35
    shape=dict(center=center.tolist(),half_extent_xy=[.02,.02],bottom_z=.305)
    assert region_permission(shape,cell,xy_error=.003,z_error=.003)['status']=='PASS'
    shape['center'][0]=cell['lo'][0]+.01
    assert region_permission(shape,cell,xy_error=.003,z_error=.003)['status']=='FAIL'
    assert region_permission(shape,cell,xy_error=None,z_error=None)['status']=='UNKNOWN'


def test_shape_fit_keeps_measured_height_and_is_not_an_error_certificate():
    p=floor()*.1;p[:,2]=.06
    q=upright_shape(p,0.)
    assert q['height_m']==pytest.approx(.06)
    assert q['uncertainty']=='unvalidated_visible_extent'


def test_partial_camera_region_associates_by_overlap_not_biased_centroid():
    from capx.execution.region_geometry import same_region
    a=dict(lo=[0,0],hi=[.4,.5],floor_z=.3)
    b=dict(lo=[0,.05],hi=[.4,.2],floor_z=.3)
    assert same_region(a,b)
    assert not same_region(a,dict(lo=[0,.52],hi=[.4,1.02],floor_z=.3))


def test_depth_disconnected_leakage_cannot_change_foreground_height():
    from capx.execution.region_geometry import foreground_component
    rng=np.random.default_rng(5);p=rng.uniform(0,.03,(1000,3))
    q=np.r_[p,rng.uniform(.1,.12,(10,3))]
    selected,meta=foreground_component(q)
    assert len(selected)==1000 and meta['discarded_points']==10
    assert np.max(selected[:,2])<.04
    with pytest.raises(ValueError,match='ambiguous'):
        foreground_component(np.r_[p,p+.15])


def test_vertical_wall_samples_cannot_define_floor_bounds():
    from capx.execution.region_geometry import horizontal_pixels
    y,x=np.indices((40,40));flat=np.stack([x*.001,y*.001,np.ones_like(x)*.3],-1)
    wall=np.stack([x*.001,np.zeros_like(x),y*.001],-1)
    assert horizontal_pixels(flat)[3:-3,3:-3].all()
    assert not horizontal_pixels(wall).any()


def test_measured_wall_planes_recover_occluded_bounds_without_private_fixture():
    from capx.execution.region_geometry import receptacle_geometry
    x,y=np.meshgrid(np.linspace(1.,1.2,40),np.linspace(2.,2.4,45))
    floor_points=np.c_[x.ravel(),y.ravel(),np.ones(x.size)*.3]
    z,t=np.meshgrid(np.linspace(.31,.4,20),np.linspace(0,1,40))
    walls=[]
    for axis,values,other in [(0,[1.,1.2,1.4],[2.,2.5]),(1,[2.,2.25,2.5],[1.,1.4])]:
        for value in values:
            p=np.zeros((z.size,3));p[:,axis]=value;p[:,1-axis]=other[0]+t.ravel()*(other[1]-other[0]);p[:,2]=z.ravel()
            walls.append((axis,p))
    g=receptacle_geometry(floor_points,walls)
    assert g['status']=='PASS'
    assert g['lo']==pytest.approx([1.,2.]);assert g['hi']==pytest.approx([1.4,2.5])
    assert g['floor_visible_hi'][0]<1.3
    assert g['uncertainty']=='unvalidated_measured_wall_planes'
    assert receptacle_geometry(floor_points,[])['status']=='UNKNOWN'


def test_alias_fallback_does_not_merge_distinct_semantic_descriptions():
    from capx.experiments.native_vision import PublicScene
    scene=PublicScene(None,None,{})
    # Each alias can denote different object sets. Stop at an unambiguous
    # descriptor; do not merge a broad fallback into an already bound instance.
    scene.cache={('specific',None):[dict(points=np.zeros((40,3)))],('broad',None):[dict(points=np.ones((40,3)))]}
    scene.obs={}
    result=scene.candidates(['specific','broad'])
    assert len(result)==1 and np.all(result[0]['points']==0)


def test_predicted_query_volume_rejects_robot_above_object_without_certifying_hold():
    from capx.experiments.native_vision import PublicScene
    roi=dict(lo=[0,0],hi=[1,1],floor_z=.1,ceiling_z=.3)
    assert PublicScene.in_region({'points':np.tile([.5,.5,.2],(40,1))},roi)
    assert not PublicScene.in_region({'points':np.tile([.5,.5,.5],(40,1))},roi)


def test_target_evidence_does_not_require_irrelevant_source_rebinding():
    from capx.experiments.native_vision import PublicScene
    s=PublicScene(None,None,{});s.source='new-current-frame'
    f=floor();z,t=np.meshgrid(np.linspace(.31,.4,20),np.linspace(0,1,40));walls=[]
    for axis,values,extent in [(0,[0,.4],.5),(1,[0,.5],.4)]:
        for value in values:
            p=np.zeros((z.size,3));p[:,axis]=value;p[:,1-axis]=t.ravel()*extent;p[:,2]=z.ravel();walls.append((axis,p))
    s.roles={'destination':dict(lo=[0,0],hi=[.4,.5],floor_z=.3,center=[.2,.25,.3])}
    s.candidates=lambda label:[dict(floor_points=f,walls=walls)]
    assert s.observe_region('destination')['status']=='PASS'
