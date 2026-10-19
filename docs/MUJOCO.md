# MUJOCO

`bench/bench_mujoco.py` exists because everything else in this repository is
checked against arithmetic I wrote myself. A closed form, a finite difference and a
Lagrangian plant are three witnesses, and all three were written by the same
person on the same day. MuJoCo was not.

The claim it tests is narrow and it is the only one worth testing first: **is the
arm in `demo/two_link.xml` the same arm as the plant in `spine/control.py`?**
If yes, `G(q)` from the Lagrangian and `qfrc_bias` from the engine agree. If no,
one of the two files is describing something else, and every other number in the
repo is a number about a model nobody looked at.

Measured, 49 poses on a 7x7 grid over q in [-1.2, 1.2]:

    max |G_mine(q) - qfrc_bias_mujoco(q)|      1.78e-15 N*m
    worst single axis                          1.78e-15 N*m

They are the same arm. That is the whole result, and it is worth the file because
it was not obvious before it was measured.

## The three traps, each found the hard way

These are recorded because each one cost a debugging round and each one would
have looked like "MuJoCo is broken" to someone reading only the error message.

### 1. MuJoCo cannot open a path with non-ASCII characters

The repo lives under `.../项目_开发/robot-spine/`. `mj_loadXML` returned

    ValueError: ParseXML: Error opening file 'D:\...\robot-spine\demo\two_link.xml':
    No such file or directory

for a file that existed, with `os.path.exists` returning `True` one line earlier.
The C++ loader takes a narrow string. The fix is to copy the model to
`%TEMP%/robot_spine_mj/` before loading; the copy is refreshed when the source is
newer. The error message is the exact opposite of the truth, so it goes in the
docstring rather than in a comment.

### 2. numpy's OpenBLAS dies on this host unless the thread count is pinned

    OpenBLAS error: Memory allocation still failed after 10 retries, giving up.

on `import numpy`, with 2.4 GB of 14 GB free. This is the same failure that was
attributed to "numpy needs too much memory" earlier: it is not memory in
aggregate, it is per-thread buffers multiplied by the core count. Pinning

    OPENBLAS_NUM_THREADS=1
    OMP_NUM_THREADS=1
    MKL_NUM_THREADS=1

makes numpy import in under a second. `spine/mujoco_backend.py` sets these for the
subprocess it spawns, so a caller does not have to know.

Consequence for the repo: MuJoCo is a *subprocess*, not an import. The stdlib-only
main path never touches numpy, and the physics check cannot take the interpreter
down with it.

### 3. A hinge with a `range` is clamped at every pose, not only at the limit

This one produced a wrong answer that looked like physics. The model file first
declared `range="-3 3"` on both hinges, which is the correct thing to write for an
arm that has limits. At `q = 0.6`, well inside that range, the arm settled at
-0.054 rad and stayed there with a 0.96 N*m holding torque, against a gravity bias
of 8.20 N*m. A PD that should have drooped by 0.41 rad was being held 8x stiffer
than its own gains allow.

    nefc            2
    efc_type        3 3            (friction/limit constraints)
    efc_force       528.85, 148.58 N*m
    qacc            -1211.8, -1543.0 rad/s^2   at rest, inside the range

The engine was solving a different problem from the plant: a joint limit that
engages at every pose. Removing `range` entirely removes it (`nefc == 0`,
`qacc = -28.7, +56.4`, the true free-swing acceleration). `limited="true"` does not
help; it is the presence of `range` that does it in 3.2.7.

The fix is not a MuJoCo flag, it is a decision: **this repo does not ask the engine
to police joint limits.** The limits are enforced by `spine/gate.py`, where they
are checkable, individually switchable, and covered by the mutation control.
Asking a physics engine for a safety property gives a property that cannot be
audited. After removing the ranges:

    same PD, both plants, final q       -0.41166928, -0.16216466   (identical)
    steady error, both plants           0.41167 rad
    max gap along the trajectory        8.0e-3 rad, peaking at 0.23 s
    max gap in the final state          0.0
    PD + the engine's own bias          final error 2.2e-5 rad

## What the two plants still disagree about

The steady states agree exactly, which is a statement about the equations. The
trajectories do not: the engine integrates RK4 at 2 ms and the plant uses the
caller's stepper, and the transient gap between them is 8e-3 rad at its worst. This
is reported instead of being called zero, because "they agree" and "they agree on
the way there" are different claims and only the first one is true.

What the engine has that the plant does not: contact, friction (both were left at
zero in this model, which is why the agreement is this tight), actuator limits,
and a mass distribution derived from the geometry rather than asserted. Adding
`damping` to a hinge would break the exact agreement immediately and that would be
the correct outcome, not a regression.
