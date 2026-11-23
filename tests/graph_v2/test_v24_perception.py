import numpy as np
from capx.experiments.native_vision import select_instances


def test_low_scores_are_not_instances_and_duplicates_are_not_two_objects():
    a=np.zeros((20,20),bool);a[2:8,2:8]=True
    b=np.zeros_like(a);b[12:18,12:18]=True
    rows=[dict(score=.9,mask=a),dict(score=.85,mask=a.copy()),dict(score=.2,mask=b)]
    assert len(select_instances(rows))==1


def test_two_confident_distinct_instances_remain_ambiguous():
    a=np.zeros((20,20),bool);a[:8,:8]=True
    b=np.zeros_like(a);b[12:,12:]=True
    assert len(select_instances([dict(score=.9,mask=a),dict(score=.8,mask=b)]))==2


def test_all_low_confidence_is_unknown_not_best_guess():
    assert select_instances([dict(score=.1,mask=np.ones((4,4),bool))])==[]
