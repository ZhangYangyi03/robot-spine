# CLAIMS

Every sentence in README.md that contains a number is a key in
`bench/results.json`, written by `python bench/bench.py`, and asserted by
`python bench/check_claims.py`. This file is the map between the two, plus what
each claim would look like if it were false. It is written because the resume has
one line per project and a line cannot carry a falsification condition.

The rule for the whole repo: a claim that cannot be made false by running
something is not a claim, it is a description.

## 1. Kinematics

| key in results.json | asserted | false would look like |
| --- | --- | --- |
| `fk.max_position_error_m` | `< 1e-12` | a wrong joint order, a sign error in the parent transform, or a mismatched URDF convention (URDF applies the joint transform before the child offset; getting that backwards still looks like an arm) |
| `jacobian.max_linear_error` | `< 1e-8` | a missing `z x (p_tip - p_joint)` term; a Jacobian built at the wrong link |
| `jacobian.max_angular_error` | `< 1e-8` | using the tip position instead of the joint axis for the angular block |
| `g1_at_scale.jacobian_vs_finite_difference_max_err` | `< 1e-8` at 29 DOF | a per-link loop that silently drops the fixed joints |

The reference for FK is a closed form written independently in `bench/bench.py`
(`l1 cos q1 + l2 cos(q1+q2)`, and so on), not a second copy of the same loop. The
reference for the Jacobian is a central finite difference on FK with a step chosen
per DOF. Two copies of one mistake agree with each other, which is why neither
reference is a refactor of the thing it checks.

## 2. Inverse kinematics

| key | asserted | meaning |
| --- | --- | --- |
| `ik.single_start.rate` | `< 0.90` -- deliberately a ceiling | one start from the zero pose misses the far elbow branch. This number is asserted to stay BAD, so nobody quietly reports only the multi-start rate |
| `ik.multi_start.rate` | `>= 0.96` | a deterministic Halton lattice over the URDF's own joint limits |
| `ik.unreachable.residual` | reported, not raised | a target 0.75 m out returns a residual, because "cannot reach" is an answer and an exception is not |

57% single-start is the honest number. It was 57% *after* a bug fix: the earlier
version re-clamped each joint into its limit and kept stepping, which parked the
arm on the limit with a 0.8 m residual for a target that is reachable. Refusing
the step converts a silent wrong answer into an early stop with a residual.

## 3. Collision and the gate

| key | asserted | note |
| --- | --- | --- |
| `collision.capsule_capsule` | exact against two hand-computed cases (0.500, 0.250) | hand-computed, not engine-computed |
| `gate.legal.false_refusal_rate` | `== 0.0` over 200 legal commands | a gate that refuses everything scores perfectly on the fault table, which is why this row exists |
| `gate.per_rule[*].caught == planted` | every rule | per-rule, not aggregate |
| `gate.per_rule[*].caught_by_intended_rule == planted` | every rule | the reason string must name the intended rule |
| `gate.mutation_control.step_jump.caught_with_rule_off` | `== 0` | the only fault type where the control isolates cleanly |
| `g1_at_scale.gate_after_a_10cm_move.ok` | `True` | calibration must not silently switch the rule off |

The mutation control is the part that matters and the part that mostly failed:
with `joint_limit` switched off, the 40 planted commands are STILL caught, by
`step_jump`. So for that fault type the control proves nothing, and the assertion
is written only where it holds. A control that quietly passes because a different
rule did the work is not a control.

Calibration exists because 3 cm capsules on the real G1 overlap at the zero pose:
the torso stack sits inside the pelvis radius. That is a property of a hand-set
capsule model, not of the pose. `Gate(calibrate=q0)` baselines every pair at a
known-healthy pose and refuses only what got closer.

## 4. Control

| key | asserted | what it shows |
| --- | --- | --- |
| `pd.pd_naive.steady_err` | `> 0.2` rad | without the gravity term the arm droops; asserted to stay bad |
| `pd.pd_gravity_comp.steady_err` | `< 1e-3` rad | with it the droop is gone |
| `closed_loop.rogue_controller_gated.peak_tau_reaching_plant` | `0.0` | a gate in the loop is not advisory |
| `closed_loop.rogue_controller_ungated.joint_stop_contacts` | `61` measured | the same controller without a gate drives the arm into its own stops |

## 5. Physics, against something that is not me

| key | asserted | why it is worth having |
| --- | --- | --- |
| `mujoco.gravity.max_abs_difference_Nm` | `< 1e-9` over 49 poses | my Lagrangian `G(q)` and MuJoCo's `qfrc_bias` are the same number to float precision. If the XML described a different arm, this row would fail even though every other row passed |
| `mujoco.same_pd_two_plants.max_gap_in_the_final_state_rad` | `== 0.0` | the two plants share a fixed point to 8 decimals |
| `mujoco.same_pd_two_plants.max_gap_along_the_trajectory_rad` | reported, ~8e-3 | a SHARED FIXED POINT IS NOT A SHARED TRAJECTORY. MuJoCo runs RK4 at 2 ms, the plant runs its own stepper; the gap peaks at 0.23 s and decays. Reported rather than called zero |
| `mujoco.gravity_comp_through_the_engine.max_joint_error_rad` | `< 1e-4` | adding the engine's own bias to the PD holds the setpoint, which is what entitles the report to say gravity was the whole droop |

## What is deliberately not claimed

- Not real-time. One gate check on the 29-DOF G1 is 6-14 ms; a 1 kHz servo loop has
  1 ms. The broad phase cuts 595 candidate pairs to 97-111 and that is still not a
  control period.
- Not a safety device. `gate.py` is a software check with a reason string, not a
  safety-rated relay. Nothing here has been through a certification process.
- Not torque-controlled hardware. There is no actuator model, no encoder noise, no
  CAN delay, no compliance. The G1 section measures kinematics on the real URDF;
  it does not touch the robot.
