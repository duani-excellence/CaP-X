"""Binding regression: real monitor producer preserves CGN source and remaps axes."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_grounded_monitor import fixture


def rotation(q):return Rotation.from_quat(np.roll(q,-1)).as_matrix()


def observe(mode):
    m,state,surfaces,calls,raw=fixture(mode)
    q=np.roll(Rotation.from_euler('xyz',[174.,8.,23.],degrees=True).as_quat(),1)
    p=np.array([.49,.01,.017]);q0=q.copy();p0=p.copy()
    def candidate(name,args,kwargs):
        if name=='sample_grasp_pose':
            calls.append((name,args));return p,q
        return raw(name,args,kwargs)
    m.invoke('observe_object',['red cube'],{},candidate)
    grasp=m.invoke('observe_grasp',['red cube'],{},candidate)
    assert np.array_equal(p,p0) and np.array_equal(q,q0)
    return m,grasp,p,q,candidate


@pytest.mark.parametrize('mode',['G0','G1'])
def test_monitor_maps_cgn_baseline_to_actual_panda_closing_axis(mode):
    m,grasp,p,q,raw=observe(mode)
    expected=rotation(q)[:,0];expected[2]=0;expected/=np.linalg.norm(expected)
    result=rotation(grasp['quaternion_wxyz'])
    np.testing.assert_allclose(result[:,1],expected,atol=1e-12)
    np.testing.assert_allclose(result[:,2],[0,0,-1],atol=1e-12)
    # Only orientation convention changes; centre recipe still uses public geometry.
    np.testing.assert_allclose(grasp['position'],[.5,0,.02],atol=1e-12)


@pytest.mark.parametrize('mode',['G0','G1'])
def test_raw_candidate_and_conversion_matrices_are_linked_unmodified(mode):
    m,grasp,p,q,raw=observe(mode)
    source=m.graph.nodes[grasp['raw_candidate_id']]
    assert source['kind']=='verification_fact'
    np.testing.assert_array_equal(source['data']['position'],p)
    np.testing.assert_array_equal(source['data']['quaternion_wxyz'],q)
    conversion=m.graph.nodes[grasp['conversion_id']]
    assert conversion['kind']=='verification_fact'
    d=conversion['data'];assert d['type']=='gripper_frame_conversion'
    assert d['source_axis']=='contact_graspnet_x' and d['target_axis']=='panda_tcp_y'
    np.testing.assert_allclose(d['raw_rotation_base'],rotation(q))
    np.testing.assert_allclose(d['output_rotation_base'],rotation(grasp['quaternion_wxyz']))
    np.testing.assert_allclose(np.array(d['raw_rotation_base'])@np.array(d['raw_to_output_right_rotation']),d['output_rotation_base'],atol=1e-12)
    edges={(e['source'],e['target']) for e in m.graph.edges}
    assert (source['id'],conversion['id']) in edges
    assert (source['id'],grasp['id']) in edges
    assert (conversion['id'],grasp['id']) in edges
    assert d['upright_projection'] is True and d['collision_verified'] is False


@pytest.mark.parametrize('mode',['G0','G1'])
def test_approach_plan_binds_converted_pose_and_retains_raw_evidence(mode):
    m,grasp,p,q,raw=observe(mode)
    plan=m.invoke('plan_skill',['approach',{'grasp':grasp['id']}],{},raw)
    np.testing.assert_array_equal(m.graph.nodes[plan]['data']['quaternion_wxyz'],grasp['quaternion_wxyz'])
    ancestors=m.graph.ancestors([plan])
    assert grasp['raw_candidate_id'] in ancestors
    assert grasp['conversion_id'] in ancestors
