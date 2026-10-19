# BEYOND THE RESUME

The resume gives this project three lines. Those three lines are the right shape for
a resume and they force out everything that makes the work checkable: the number
that is bad, the rule whose test does not work, the decision that looks arbitrary
until you see what it was chosen over, and the three error messages that said the
opposite of the truth. This file is what those three lines cannot carry.

    resume line                    what it costs to say it that way
    ---------------------------    ---------------------------------------------
    "FK / IK / collision /         which of the six numbers is bad: single-start IK
     safety gate, stdlib only"     solves 57% of reachable targets, and it is
                                   asserted in CI to stay under 90%
    "gate catches planted faults"  the mutation control does not isolate the
                                   joint-limit rule at all: switch it off and the
                                   commands are still caught, by step_jump. Only
                                   step_jump survives its own control
    "checked against MuJoCo"       the two plants share a fixed point exactly and
                                   diverge by 8e-3 rad on the way there, because
                                   the integrators differ. A shared fixed point is
                                   not a shared trajectory

## The five things a resume line cannot hold

**1. The honest bad number.** Single-start IK reaches 57% of reachable targets;
multi-start reaches 100%. The resume may only quote the second. The repo quotes both,
and asserts the first one to stay below 0.90 so that a future change cannot quietly
promote it. A number that is asserted to remain bad is a stronger claim than a good
number, because it means nobody is allowed to hide the failure later.

**2. The experiment that failed.** For the joint-limit fault, the mutation control
does not work: with the rule off, the same 40 commands are still refused by the step
rule, because a command that jumps past a limit is also a large step. This is
printed in the README's limitation section rather than removed from the table. A
verification suite where every control passes is a suite nobody has attacked.

**3. The decision and its cost.** Capsules instead of meshes, with a calibration
pose, because 3 cm capsules on the real G1 overlap at the zero pose -- the torso
stack sits inside the pelvis radius. The alternative (shrink the radii until the pose
is clean) makes the model wrong in the direction that matters, and the failure is
silent. `docs/DESIGN.md` decision 2.

**4. The limits of the thing, stated as limits.** One gate check on the 29-DOF G1 is
6-14 ms; a 1 kHz servo loop has 1 ms. The repo says that in the README, with the
reason the broad phase exists (595 pairs to 97-111) and the reason it is not enough
(needs C or a tree, not a tighter Python loop). A project that has never measured
its own control period does not get to claim real-time.

**5. The traps, with their error messages.** Three of them, in `docs/MUJOCO.md`.
The one worth repeating: MuJoCo 3.2.7 clamps a hinge that carries a `range` at every
pose, not only at the limit. At `q = 0.6` inside a -3..3 range, `efc_force` was
528 N*m and the arm was held 8x stiffer than its own gains -- a wrong answer that
looked exactly like physics. The measurement that found it, and what it changed (the
model file declares no joint range; the gate owns the limits), is a longer story than
"I integrated MuJoCo".

## What is genuinely new here, in one line each

- A plant checked against an engine it did not write, with the transient gap reported
  rather than rounded away.
- A safety gate whose rules can be disabled one at a time, and a mutation control
  that is allowed to fail and says so.
- The G1 at 29 DOF, so the numbers include the cost of scale (6-14 ms/gate check)
  rather than only the 2-DOF case where everything looks cheap.
- Two bugs found by the bench disagreeing with a claim: a `so3_log` that returned
  zero for a 1e-7 rad rotation (making every finite-difference check report a lie),
  and a capsule span that turned coincident decorative links into phantom collisions.
- No dependency in the verification path: `import numpy` failing with an OpenBLAS
  allocation error on this host is a real risk, and a suite that cannot start looks
  the same as a suite that passes.
