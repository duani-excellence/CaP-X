"""Read-only evidence obligations for a forthcoming place/release contract.

No task identities, private state, robot actions or new sensor implementation.
This forecast is a necessary feasibility check, not a future action permit.
Actual elapsed time, contradiction and geometry are checked again at dispatch.
"""
import numpy as np
from .state_memory import bounded_place_metrics


def placement_obligations(memory,*,held_object,support,geometry,tcp,target,quaternion,
                          stamp,calibration,grasp_epoch,intent_id=None):
    future=np.r_[target,quaternion,tcp[7]].tolist()
    estimate=memory.forecast(held_object,tcp=future,stamp=stamp,calibration=calibration,
        grasp_epoch=grasp_epoch,hold_verified=True,occlusion='expected')
    row=dict(schema='capx.future_evidence.v1',future_intent_id=intent_id,
        permission=False,forecast_only=True,
        expected_visibility='conditional_on_future_depth_confirming_occlusion',
        requirements=['held_geometry_anchor','rigid_attachment','support_geometry',
                      'bounded_relative_alignment','fresh_final_capture','released_before_retreat'],
        held_forecast=estimate,missing=[],retained_sources=[],
        limitation='zero_future_latency_lower_bound; feasibility is not a guarantee or permission')
    if estimate['status']!='PREDICTED':
        row.update(status='UNCERTAIN',reason='held_anchor_not_propagatable',
            missing=['held_geometry_anchor'],next_work='acquire_anchor')
        return row
    row['retained_sources']=[estimate['source_id']]
    if support is None:
        row.update(status='UNCERTAIN',reason='support_geometry_missing',missing=['support_geometry'],next_work='observe_support')
        return row
    base={k:memory.profile[k] for k in ('xy_m','z_m','height_m')}
    check=bounded_place_metrics(estimate['surface'],support,geometry,estimate['bounds'],support.get('error_bounds',base))
    row.update(status=check['status'],reason=check['reason'],release_forecast=check,
        next_work='continue_to_boundary_checks' if check['status']=='PASS' else 'recompute_alignment')
    if check['status']!='PASS':row['missing']=['bounded_relative_alignment']
    return row
