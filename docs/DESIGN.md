# DESIGN

Why this repo is shaped the way it is. `CLAIMS.md` says what is measured;
this says what was decided, what was rejected, and what the alternative would have
cost. Written because "I built a robot kernel" is not information and "I chose a
capsule model over a mesh model, and here is the number that made that choice
survivable" is.

## Decision 1: stdlib only, no numpy

Three reasons, in order of how much they mattered:

1. **The thing being checked must not be able to fail for an unrelated reason.**
   `import numpy` on this host dies with an OpenBLAS allocation error unless the
   thread count is pinned (see `MUJOCO.md`). A verification suite that cannot start
   is indistinguishable from a verification suite that passes.
2. **The arithmetic is small enough to write.** A 4x4 transform is 16 floats. A
   29-DOF Jacobian is 6x29. Nothing here needs a library; the largest solve is a
   Cholesky on a 6x6.
3. **It makes the claim portable.** CI runs on a bare `actions/setup-python` on
   Windows and Ubuntu with no install step. If a number changes between the two, it
   is the number, not the environment.

The cost is real: pure-Python collision on 29 DOF is 6-14 ms per gate check and
that is 6-14x too slow for a 1 kHz servo loop. That cost is written in the README
as a limitation rather than hidden by quoting only the 2-DOF number.

## Decision 2: a capsule model, and a calibration pose

Capsules, not meshes, because a capsule-to-capsule distance is six lines and can be
hand-checked (`collision.capsule_capsule` is asserted against 0.500 and 0.250
computed on paper). Meshes would need a library, a broad phase tree, and a tolerance
story, and the gate would then be checking a geometry library rather than a rule.

The honest problem with capsules appears immediately on the real G1: at 3 cm radii,
eight capsule pairs overlap at the zero pose, because the torso stack sits inside
the pelvis radius. That is a property of a hand-set model, not of the pose.

Two options were available. Either shrink the radii until the zero pose is clean,
which makes the model wrong in the direction that matters (it would stop noticing
real collisions), or calibrate. Calibration measures every pair at a pose the
machine is known healthy in and refuses only what got closer than its baseline, or
what crossed the absolute clearance when it started outside it. The second option
keeps the rule sensitive where it counts, and the bench proves calibration does not
become an off switch: a 10 cm wrist move after calibration is still admitted, a
planted joint-limit break after calibration is still refused.

The first option is what most quick projects do, and the failure is silent: nothing
ever reports that the collision model was tuned to the test pose.

## Decision 3: the gate is a separate object with a reason string

`gate.py` does not import the controller, the plant or the solver. It is handed a
command and a state and returns a decision that names the rule:

    joint_limit / step_jump / velocity / torque / self_collision / obstacle

Rejected alternative: a `try/except` around the controller that aborts on a
suspicious value. It is shorter and it is worse in exactly one way that matters --
an exception does not say WHICH invariant broke, so the mutation control cannot be
run. With a reason string, `bench_gate.py` can plant a fault, switch off the rule
that is supposed to catch it, and check that the catch disappears. For
`step_jump` it does (40/40 -> 0/40). For `joint_limit` it does not, because the
step rule catches the same commands, and that failure is printed rather than
smoothed over.

A gate whose rules cannot be individually disabled cannot be tested. A gate whose
rules can be disabled but are not is a gate nobody tested.

## Decision 4: IK returns a residual, never raises

Target out of reach: 0.75 m residual, no exception. Target reachable but the
elbow branch needs a joint to cross its limit: early stop with a residual.

Rejected alternative: raising `UnreachableError`. It reads better and it collapses
two different facts -- "the solver could not" and "the solution is unsafe" -- into
one signal. `python -m spine` asks the gate even when IK failed, on purpose,
because those are two results a user needs to be able to tell apart.

## Decision 5: multi-start, and reporting the single-start number anyway

The deterministic Halton lattice over the URDF's own limits takes 200/200 targets at
2.5 attempts on average. Quoting that alone would be the standard move. The
single-start number is 57%, it is asserted in CI to stay under 0.90, and it appears
in the README's table, because a solver that needs three starts and a solver that
needs one are different products.

## Decision 6: MuJoCo as a subprocess, and joint limits taken away from it

`spine/mujoco_backend.py` shells out. Three consequences, all intended: the stdlib
path never pays for numpy, a crashed engine cannot take the bench with it, and the
environment pinning (`OPENBLAS_NUM_THREADS=1`) is local to the child.

And the decision that came out of a measurement rather than a preference: the model
file declares no joint `range`. MuJoCo 3.2.7 clamps a hinge that carries a range at
every pose, not only at the limit -- measured, `efc_force = 528 N*m` at `q = 0.6`
inside a -3..3 range, holding the arm 8x stiffer than its own gains. The engine was
solving a different problem from the plant. The limits live in the gate instead,
where they can be audited and switched off one at a time. Handing a safety property
to a solver returns a property nobody can inspect.

## What was rejected outright

- **A mesh-based collision model.** Needs a library, needs a tree, and the gate
  would be testing the library.
- **Learning the dynamics.** No hardware, no data, and a learned model cannot be
  checked against a formula; it can only be checked against more data.
- **Sim-to-real language.** There is no real arm in this repo. Saying "sim-to-real"
  about a plant with no friction would be the exact failure mode the README's
  footer warns about: a description masquerading as a result.
- **A GUI.** The output of this repo is numbers in a JSON file and a reason string.
