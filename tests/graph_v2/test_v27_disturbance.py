import numpy as np
import pytest
from capx.experiments.native_faults import disturbance_wrench


def test_impulse_scales_with_mass_instead_of_throwing_light_meshes():
    spec=dict(impulse_velocity_world=[.5,0.,.25],duration_seconds=.12,compensate_gravity=True)
    gravity=np.array([0.,0.,-9.81])
    for mass in [0.0001,.1,1.]:
        force=disturbance_wrench(spec,mass,gravity)
        assert np.allclose((force[:3]/mass+gravity)*.12,[.5,0.,.25])
        assert np.allclose(force[3:],0)


def test_disturbance_rejects_unbounded_velocity_or_mass():
    with pytest.raises(ValueError):disturbance_wrench(dict(impulse_velocity_world=[20,0,0],duration_seconds=.12),.1,[0,0,-9.81])
    with pytest.raises(ValueError):disturbance_wrench(dict(impulse_velocity_world=[.5,0,0],duration_seconds=.12),0,[0,0,-9.81])


def test_legacy_force_is_preserved_for_exact_historical_reproduction():
    assert np.allclose(disturbance_wrench(dict(wrench_world=[6,0,12,0,0,0]),.1,[0,0,-9.81]),[6,0,12,0,0,0])
