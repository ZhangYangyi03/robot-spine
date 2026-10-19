# robot-spine

A robot control kernel in pure Python: forward kinematics, inverse kinematics,
capsule collision, a safety gate, and a PD loop running on a plant whose equations
are written out. No numpy, no MuJoCo, no simulator, no GPU, no network.

It exists for one reason, and it is the reason `eda-spine` exists:

> a controller that runs is not a controller that is correct, and the only thing
> that tells the two apart is a number produced by something other than the
> controller.

Every claim below is a line in `bench/results.json`, written by
`python bench/bench.py`, and asserted by `python -m unittest discover -s tests`.
If a claim and the file disagree, the file is right.

## The four stops, borrowed from the EDA chain

    eda-spine                        robot-spine
    -----------------------------    ------------------------------------------
    qoragent: search synthesis       ik: search joint solutions, and refuse any
      passes, gate the wins          candidate the joint-limit/gate check rejects
    assertforge: prove a property    fk/jac: the model is checked against a closed
      of the RTL                     form and against finite differences
    assertforge: perturb, the gate   gate: plant a fault, require a refusal, and
      must refute                    require the mutation control to collapse
    covagent: measure coverage       bench: what fraction of reachable targets the
      from the raw database          solver reaches, measured not assumed

Those are not metaphors. `ik_solve(gate=...)` will not return a joint vector the
gate rejects; `control.run(gate=...)` will not pass a torque the gate rejects to
the plant; `bench_gate.py` re-runs every planted fault with the rule switched off.

## What is measured

`python bench/bench.py` -- 5 seconds, stdlib only, Windows 11 / CPython 3.13.

    fk vs closed form           200 poses, max position error        2.2e-16 m
    jacobian vs finite diff     50 poses, max error                  1.5e-11
    jacobian on G1, 29 joints   29 columns, max error                4.6e-12
    capsule distance            2 hand-computed cases                0.500, 0.250 exact
    IK single start             114 / 200 reachable targets          57%
    IK multi start              200 / 200 reachable targets          100%
    IK worst residual           9.9e-10 m at tol 1e-9                12.4 ms/case
    IK unreachable target       residual 0.75 m, no exception        reported, not raised
    PD, no gravity term         steady error 0.412 rad               tip off by 0.361 m
    PD, gravity term            steady error 2e-05 rad               settles 1.30 s
    gate legal commands         200 / 200 admitted                    false refusal 0.0
    gate planted joint limit    40 / 40 caught, reason `joint_limit`
    gate planted step jump      40 / 40 caught, reason `step_jump`   mutation off: 0/40
    gate velocity/torque/obstacle/self-collision   1/1 each, right reason each
    rogue controller ungated    asks 360 N*m on a 30 N*m joint        reaches the plant, 61 joint-stop hits
    rogue controller gated      1500 / 1500 ticks refused             0.0 N*m reaches the plant

Three of those numbers are uncomfortable on purpose. They are explained below
rather than tuned away.

### IK from one start solves 57% of reachable targets

The 2-link arm has two elbow branches. From the zero pose one of them cannot be
reached without a joint crossing its limit, so the solver stops and reports the
residual. Multi-start over a deterministic Halton lattice of the URDF's own limits
reaches 100% at a mean of 2.5 attempts. Quoting only the second number would hide
that the first is bad; quoting only the first would hide that the fix is cheap.

The clamp was the bug: the earlier version re-clamped each joint into its limit
and kept stepping, which parked the arm ON the limit with a 0.8 m residual for a
target that is perfectly reachable. Refusing the step turns a silent wrong answer
into an honest early stop.

### The joint-limit mutation control does not collapse

    joint_limit planted, rule ON      40 / 40 caught
    joint_limit planted, rule OFF     40 / 40 caught

With the joint-limit rule off, those commands are still refused -- by the step
rule, because a command that jumps past a limit is also a large step from the pose
the bench is in. So "40/40 caught by the joint-limit rule" is real (the reason
string says `joint_limit`), but the mutation control cannot isolate that rule for
this fault type. Step-jump is the row where the control works cleanly, and that is
the row the test asserts. A mutation control quietly satisfied by a different rule
is not a control, so this is written down.

### 14 ms is not a control period

On the 29-DOF G1, one gate check costs 6-14 ms in pure Python. A 1 kHz servo loop
has 1 ms. The broad phase cuts the exact tests from 595 pairs to 97-111, which is
the reason it exists, and it is still not enough. Holding a real control period at
29 DOF needs C or a proper broad phase tree, not a tighter Python loop.

## Two bugs the bench found in this repo

Both were found by a measurement disagreeing with a claim, which is the whole point
of having the measurement:

1. `so3_log` returned ZERO for a 1e-7 rad rotation (the `cos -> 1` guard was too
   loose), so every finite-difference Jacobian check reported a 1.0 rad/s error.
   The Jacobian was right; the check was reading a lie. Now split three ways
   (small angle, generic, near-pi) and covered by a test.
2. The capsule for a link spanned parent-frame to child-frame, which turned every
   decorative link at a coincident origin (head_link, logo_link, imu_in_pelvis,
   pelvis_contour_link on the G1) into a phantom fat capsule around the pelvis,
   and made the self-collision rule fire on the zero pose. A capsule derived from
   the wrong span is not a conservative approximation, it is a wrong model. Now
   each capsule spans the link's own frame to the next joint's frame and is
   tapered, and pairs of capsules belong to *different* links by construction.

## The gate, and why it is a separate object

`gate.py` does not know about the controller, the plant, or the solver. It is
handed a command and a state and answers with a reason string naming the rule:

    joint_limit     q outside the URDF's <limit>
    step_jump       one command moves a joint further than max_step
    velocity        |qdot| above the servo rating
    torque          |tau| above the joint rating
    self_collision  two non-adjacent links inside the clearance
    obstacle        a link inside the clearance of a scene obstacle

Each rule can be switched off individually -- that is what makes the mutation
control possible -- and each has a calibration mode. Calibration matters for a
real URDF: with 3 cm capsules on the G1, eight capsule pairs overlap at the zero
pose because the torso stack sits inside the pelvis radius. That is a property of
a hand-set capsule model, not of the pose. `Gate(calibrate=q0)` measures every
pair at a pose the machine is known healthy in and refuses only what got closer
than that, or what crossed the absolute clearance when it started outside it.

    absolute clearance at q=0     REFUSED (self_collision), 111 exact pairs tested
    calibrated against q=0        admitted, 8 pairs baselined at the start pose
    10 cm wrist move afterwards   admitted -- calibration did not switch the rule off
    planted joint-limit break     REFUSED by `joint_limit` (and `step_jump`)

## Scale: the same code on 29 DOF

`bench/bench_g1.py`, on the Unitree G1 URDF from
[unifolm-wla-toolkit](https://github.com/ZhangYangyi03/unifolm-wla-toolkit):
39 links, 38 joints, 29 movable.

    load                                      3.5 ms
    world Jacobian (6 x 29)                   1.8 ms, error 4.6e-12 vs finite difference
    IK, wrist +10 cm, position only           1 start, 3 iterations, 17.9 ms, residual 2.3e-06 m
    IK, wrist +10 cm, position + orientation  1 start, 18.3 ms, residual 2.3e-05 m / 1.4e-05 rad
    gate check, absolute clearance            6.3 ms
    gate check, calibrated                    14.3 ms first call (includes baselining), less after

## Run it

    python bench/bench.py                 # every number, -> bench/results.json
    python -m unittest discover -s tests  # the same numbers, as assertions
    python -m spine                       # the built-in 2-link arm
    python -m spine demo/g1.urdf --link left_wrist_yaw_link --dz 0.10 --radius 0.03

`python -m spine` prints the model, the target, what IK returned *with its
residual*, and what the gate says about the answer -- and asks the gate even when
IK failed, because "the solver could not" and "the solution is unsafe" are
different results and must not be collapsed into one line.

## What this is not

Not a physics engine. The plant in `control.py` is a rigid 2-link arm with the
Lagrangian terms written out; no contact, no friction, no actuator dynamics, no
integration-error budget. MuJoCo or PyBullet is the right tool for those, and that
port is the next step, not this repo.

Not a controller for a real arm. No trajectory generation, no time
synchronisation, no fieldbus, no safety-rated stop. `gate.py` is a software check,
not a safety relay.

Not an EDA or CAD kernel. `eda-spine` is where the silicon lives; this is the same
skeleton pointed at motion.

## Layout

    spine/linalg.py     4x4 rigid transforms, Rodrigues, so3 log, rotation-6D with
                        Gram-Schmidt, Cholesky, damped least squares
    spine/urdf.py       URDF tree, FK, geometric Jacobian (adapted from
                        unifolm-wla-toolkit/urdf_kin.py, stdlib instead of numpy)
    spine/ik.py         DLS IK with a joint-limit wall, backtracking line search,
                        multi-start, gate-aware, residual reporting
    spine/collision.py  capsule-capsule / capsule-sphere distance, tapered radii,
                        bounding-sphere broad phase
    spine/gate.py       six rules, individually switchable, each with a reason,
                        plus calibration against a known-healthy pose
    spine/control.py    Lagrangian 2-link plant, PD with gravity compensation,
                        closed-loop runner, joint stops, metrics
    spine/scenes.py     the 2-link URDF and the G1 adapters, link radii, adjacency
    bench/bench.py      every claim above, written to bench/results.json
    bench/bench_ik.py   IK success rate: single vs multi start, unreachable target
    bench/bench_gate.py admitted/legal, caught/planted, and the mutation control
    bench/bench_g1.py   the same at 29 DOF
    demo/g1.urdf        the Unitree G1 kinematic model, from unifolm-wla-toolkit

## Licence

Apache-2.0. `spine/urdf.py` adapts `urdf_kin.py` from
[unifolm-wla-toolkit](https://github.com/ZhangYangyi03/unifolm-wla-toolkit), same
author, reimplemented on the stdlib linear algebra here so this repo has no numpy
dependency. `demo/g1.urdf` is that repo's file, unchanged.
