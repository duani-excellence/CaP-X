from capx.experiments.feedback_analysis import confusion

def test_uncertain_is_not_a_true_positive():
    rows=[{'status':'PASS','truth_bad':False},{'status':'FAIL','truth_bad':True},
          {'status':'PASS','truth_bad':True},{'status':'FAIL','truth_bad':False},
          {'status':'UNCERTAIN','truth_bad':True}]
    r=confusion(rows)
    assert (r['tp'],r['tn'],r['fp'],r['fn'],r['uncertain'])==(1,1,1,1,1)
    assert r['accuracy_decided']==.5

def test_tracking_truth_uses_base_frame():
    from capx.experiments.feedback_analysis import truth_tracking
    geometry={'base_position_world':[1,2,3],'base_rotation_world':[[-1,0,0],[0,-1,0],[0,0,1]]}
    state={'tcp':{'position':[.5,2,3.2],'quaternion_wxyz':[0,0,0,1]}}
    metrics={'target_tcp':[.5,0,.2,1,0,0,0]}
    result=truth_tracking(state,geometry,metrics)
    assert result['position_error_m']<1e-10 and not result['truth_bad']

def test_initial_open_is_not_a_placement_attempt():
    from capx.experiments.feedback_analysis import skill_counts
    def state(t,command,contact,z=.82,success=False):
        return {'sim_time':t,'gripper_command':command,'grasp_contact':contact,'raw_success':success,
                'gripper_opening':.08 if command else .04,
                'tcp':{'position':[0,0,1],'quaternion_wxyz':[1,0,0,0]},
                'objects':{'primary':{'position':[0,0,z],'speed':0,'angular_speed':0},
                           'secondary':{'position':[0,0,.8],'speed':0,'angular_speed':0}}}
    a=state(0,1,False);b=state(.1,0,True);c=state(.2,0,True,.92);d=state(.3,1,False,.92,True)
    apis=[{'name':'open_gripper','before':a,'after':a}, {'name':'close_gripper','before':a,'after':b},
          {'name':'goto_pose','args':[[0,0,1.1],[1,0,0,0]],'kwargs':{},'before':b,'after':c},
          {'name':'open_gripper','before':c,'after':d}]
    states=[d,state(.9,1,False,.92,True)]
    r=skill_counts(apis,states,{'base_rotation_world':[[1,0,0],[0,1,0],[0,0,1]],'base_position_world':[0,0,0]})
    assert r==dict(grasp_attempts=1,grasp_contact_after_close=1,lift_attempts=1,lift_with_contact=1,release_attempts=1,stable_after_release=1)
