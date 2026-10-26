# robot-spine

A robot control kernel in pure Python: forward kinematics, inverse kinematics,
capsule collision, a safety gate, and a PD loop on a plant whose equations are
written out. The stdlib path has no numpy, no simulator, no GPU and no network;
the physics check next to it runs MuJoCo in a subprocess, because the plant's
equations deserve a witness that did not write them.

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

Those are not metaphors: `ik_solve(gate=...)` will not return a joint vector the
gate rejects, `control.run(gate=...)` will not pass a rejected torque to the plant,
and `bench_gate.py` re-runs every planted fault with the rule switched off.

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
    gravity vs MuJoCo           49 poses, max difference              1.8e-15 N*m
    same PD on both plants      final state gap                       0.0 rad
    same PD on both plants      worst transient gap                    8.0e-3 rad at 0.23 s
    PD + engine bias            final joint error                      2.2e-05 rad

Three of those numbers are uncomfortable on purpose, and are explained below
rather than tuned away.

### IK from one start solves 57% of reachable targets

The 2-link arm has two elbow branches. From the zero pose one cannot be reached
without a joint crossing its limit, so the solver stops with a residual. Multi-start
over a deterministic Halton lattice of the URDF's own limits reaches 100% at a mean
of 2.5 attempts. Quoting only the second would hide that the first is bad; quoting
only the first would hide that the fix is cheap.

The clamp was the bug: the earlier version re-clamped each joint into its limit and
kept stepping, parking the arm ON the limit with a 0.8 m residual for a reachable
target. Refusing the step turns a silent wrong answer into an honest early stop.

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
   loose), so every finite-difference Jacobian check reported a 1.0 rad/s error. The
   Jacobian was right; the check was reading a lie. Now split three ways and tested.
2. The capsule for a link spanned parent-frame to child-frame, turning every
   decorative link at a coincident origin (head_link, logo_link, imu_in_pelvis) into
   a phantom fat capsule around the pelvis and firing self-collision at the zero
   pose. A capsule from the wrong span is not a conservative approximation, it is a
   wrong model. Now each spans the link's own frame to the next joint's frame.

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
control possible -- and each has a calibration mode: with 3 cm capsules on the real
G1, eight pairs overlap at the zero pose because the torso stack sits inside the
pelvis radius. `Gate(calibrate=q0)` baselines every pair at a pose the machine is
known healthy in and refuses only what got closer, or what crossed the absolute
clearance when it started outside it.

    absolute clearance at q=0     REFUSED (self_collision), 111 exact pairs tested
    calibrated against q=0        admitted, 8 pairs baselined at the start pose
    10 cm wrist move afterwards   admitted -- calibration did not switch the rule off
    planted joint-limit break     REFUSED by `joint_limit` (and `step_jump`)

## The plant, checked against a physics engine

Everything above is checked against arithmetic I wrote: a closed form, a finite
difference, and a Lagrangian plant. All three were written by the same person on
the same day. `bench/bench_mujoco.py` asks something that did not:
is `demo/two_link.xml` the same arm as `spine/control.py`?

    max |G_mine(q) - qfrc_bias_mujoco(q)|   1.8e-15 N*m    over 49 poses
    same PD, both plants, final q           -0.41166928, -0.16216466   identical
    same PD, worst transient gap            8.0e-3 rad, peaking at 0.23 s
    PD + the engine's own bias              final joint error 2.2e-05 rad

They are the same arm, and the trajectories are not the same curve: the engine runs
RK4 at 2 ms, the plant runs its own stepper. A shared fixed point is a statement
about the equations; it is not a statement about the path, so both are printed.

Three traps are in `docs/MUJOCO.md`, each with the measurement that found it. The
one that matters: a hinge carrying `range="-3 3"` is clamped by MuJoCo 3.2.7 at
every pose, not only at the limit -- `efc_force = 528 N*m` at `q = 0.6`, holding the
arm 8x stiffer than its own gains. So the model declares no joint range. Joint
limits are the gate's job, where they can be audited and switched off one at a time;
a safety property handed to a solver is a property nobody can inspect.

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

`python -m spine` prints the model, the target, what IK returned with its residual,
and the gate's verdict -- and asks the gate even when IK failed, because "the solver
could not" and "the solution is unsafe" must not collapse into one line.

## What this is not

Not a physics engine. The plant in `control.py` is a rigid 2-link arm with the
Lagrangian terms written out; no contact, no friction, no actuator dynamics. MuJoCo
is reached only through `spine/mujoco_backend.py`, and only to check that plant --
the gate, the solver and the controller never import it.

Not a controller for a real arm. No trajectory generation, no time synchronisation,
no fieldbus, no safety-rated stop: `gate.py` is a software check, not a safety relay.

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
    bench/bench_mujoco.py  the plant against MuJoCo: gravity, both loops, bias comp
    bench/check_claims.py  the README's numbers as 20 assertions, run by CI
    spine/mujoco_backend.py  subprocess wrapper; sets the env MuJoCo needs here
    demo/g1.urdf        the Unitree G1 kinematic model, from unifolm-wla-toolkit
    demo/two_link.xml   the same arm as control.py, for MuJoCo. No joint ranges.

    docs/CLAIMS.md      every number -> its assertion -> what false would look like
    docs/DESIGN.md      the six decisions, the rejected alternatives, their cost
    docs/MUJOCO.md      the three traps, each with the measurement that found it
    docs/REPRODUCE.md   how to run it, and which assertions are written to stay bad
    docs/BEYOND_THE_RESUME.md  the five things a resume line cannot carry

## The embodied half: the same gate, in front of a real game

`spine/embodied.py` points the gate at a body in a running Minecraft server.
`spine/voxel.py` is the world it reads; `spine/live.py` is the wire to the game.
The reason this is in the same repo as the IK and the MuJoCo check is the same
reason those are: a gate is worth what its world model is worth, and a world model
this repo also wrote is worth almost nothing.

So the live bench removes the author from the loop:

    the world the gate reads        the server's chunks, over the body's socket
    the physics the gate meets      the server's own 20 Hz integrator
    the verdict the gate is judged by   the server's health, which the gate never sees

`python bench/live/bench_live.py` starts nothing (the server must already be up)
and runs two arms on a floating platform 160 blocks up, so "off the edge" is
lethal instead of a two-block step onto grass:

    arm           intents   refused   reason         final position        fell
    gated           138        80     drop_height    y=101, z=-30.15       no
    ungated         240         0     --             y=-60,  z=2.5         yes

The control arm is the point. Without it, "the body survived" only proves the
platform was big enough. With it, survival is attributable to the gate: identical
intent, gate switched to admit everything, body falls 160 blocks and dies.

### What the live bench found that the simulator never did

The gate predicted the *intended* velocity. In the simulator that is exactly what
the body does, because the plant was written to obey the intent. In the game the
body has momentum: it keeps `GROUND_DRAG` (0.6) of its horizontal speed per tick
with no input, `GROUND_FRICTION` (0.546) with input, `AIR_DRAG` (0.91) in the air.
So the first live run refused at z = -30.23 and the body coasted to z = -30.34 and
left the platform anyway: the refusal was correct about the next tick and one tick
late about the body.

The fix is `coast_offset()`: the gate now also scans the column momentum alone will
carry the body to, and refuses on that. The regression test is
`test_momentum_carries_the_body_past_where_the_intent_stopped`; reverting the fix
makes it fail, and makes the live body fall. That is the whole argument for doing
the bench in a real game rather than a second simulator.

    simulator, before the fix     gated arm: 44 refusals, then fell
    simulator, after the fix      gated arm: 80 refusals, stayed on the platform

The plant itself was checked against the game before any of this, tick by tick:

    walk, steady state         game 0.2159 blocks/tick    plant 0.2158
    jump arc, apex height      game 1.2522 blocks         plant 1.2522
    jump arc, per-tick vy      game == plant              max difference 1e-5

and `test_plant_jump_arc_matches_the_closed_form` holds the arc to 1 cm.

### Running it

    java -Xmx900M -jar mc/server.jar nogui        # 1.21.1, offline, rcon on 25575
    cd mc/bot && npm install mineflayer@4
    python bench/live/bench_live.py --ticks 240  # both arms

`mc/server.properties`, `mc/eula.txt` and the flat world are in `mc/`. The server
is *not* checked in: `mc/server.jar` is 52 MB and re-downloadable from Mojang.

## Layout, the embodied half

    spine/voxel.py        blocks, what is solid, what hurts, capsule-to-block
                          clearance, and the one frame rotation to the URDF
    spine/embodied.py     the voxel plant (gravity, collision, fall damage, hazards)
                          and EmbodiedGate: five rules, each naming itself
    spine/live.py         RCON, framed JSON over a socket, RemoteWorld -- the gate's
                          world answered by the running server, cached per column
    bench/live/bench_live.py      the two arms above, on a real server
    bench/live/results_live.json  what it printed
    mc/bot/spine-body.js  the body: presses keys, reports state, answers column
                          queries out of the chunk cache. No policy, no physics
    mc/bot/live_gate.py   the loop: state -> intent -> gate -> keys


## Licence

Apache-2.0. `spine/urdf.py` adapts `urdf_kin.py` from
[unifolm-wla-toolkit](https://github.com/ZhangYangyi03/unifolm-wla-toolkit), same
author, reimplemented on the stdlib linear algebra here so this repo has no numpy
dependency. `demo/g1.urdf` is that repo's file, unchanged.
