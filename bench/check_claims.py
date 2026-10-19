# -*- coding: utf-8 -*-
"""The README's claims as assertions about bench/results.json."""
import json
r = json.load(open('bench/results.json'))
assert r['fk']['max_position_error_m'] < 1e-12
assert r['jacobian']['max_linear_error'] < 1e-8
assert r['jacobian']['max_angular_error'] < 1e-8
assert r['ik']['multi_start']['rate'] >= 0.96
assert r['ik']['single_start']['rate'] < 0.90
assert r['gate']['legal']['false_refusal_rate'] == 0.0
for rule, row in r['gate']['per_rule'].items():
    assert row['caught'] == row['planted'], rule
    assert row['caught_by_intended_rule'] == row['planted'], rule
assert r['gate']['mutation_control']['step_jump']['caught_with_rule_off'] == 0
assert r['pd']['pd_naive']['steady_err'] > 0.2
assert r['pd']['pd_gravity_comp']['steady_err'] < 1e-3
g = r['g1_at_scale']
assert g['jacobian_vs_finite_difference_max_err'] < 1e-8
assert g['ik_position_only']['ok'] is True
assert g['gate_at_zero_pose_absolute']['ok'] is False
assert g['gate_at_zero_pose_calibrated']['ok'] is True
assert g['gate_after_a_10cm_move']['ok'] is True
mj = r.get('mujoco', {})
if 'skipped' in mj:
    # a skipped physics check must be LOUD, and must not be confused with a pass
    print('mujoco section SKIPPED:', mj['skipped'], '--', str(mj.get('detail'))[:200])
else:
    assert mj['gravity']['max_abs_difference_Nm'] < 1e-9
    assert mj['same_pd_two_plants']['max_gap_in_the_final_state_rad'] == 0.0
    assert mj['same_pd_two_plants']['max_gap_along_the_trajectory_rad'] < 0.05
    assert mj['gravity_comp_through_the_engine']['max_joint_error_rad'] < 1e-4
print('claims hold')
