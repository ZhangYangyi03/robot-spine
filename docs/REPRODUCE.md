# REPRODUCE

## The short path

    git clone https://github.com/ZhangYangyi03/robot-spine
    cd robot-spine
    python bench/bench.py                 # writes bench/results.json, ~6 s
    python bench/check_claims.py          # asserts every number in the README
    python -m unittest discover -s tests  # 23 tests, ~2 s

Python 3.9+ . No pip install. On Windows 11 / CPython 3.13 / Ubuntu-latest, both in
CI under 3.9 and 3.12.

## With the physics engine

    pip install mujoco==3.2.7
    python -c "import sys; sys.path.insert(0,'.'); from bench import bench_mujoco; \
               import json; print(json.dumps(bench_mujoco.run(), indent=1))"

If `import mujoco` fails on a machine where it is installed, see `MUJOCO.md`: on
this host the failure is OpenBLAS thread buffers, not missing memory, and
`OPENBLAS_NUM_THREADS=1` fixes it. The backend sets that for itself when it spawns
the subprocess, so the command above works without exporting anything.

If MuJoCo is absent, `bench/bench.py` still runs and the `mujoco` section of
`results.json` records the reason it was skipped rather than a zero. A skipped
physics check and a passed physics check must not look the same.

## What a wrong result looks like

`bench/check_claims.py` is 20 assertions over `bench/results.json`. Each one is a
line of README prose turned into a comparison. If a change makes an honest number
worse -- single-start IK going from 57% to 40%, the gate's false-refusal rate
leaving zero -- CI fails. That is deliberate: the repo is allowed to get worse, it
is not allowed to get worse quietly.

The three assertions that a naive reader will find strange:

    assert ik.single_start.rate < 0.90      # 57%, asserted to stay bad
    assert pd.pd_naive.steady_err > 0.2     # asserted to keep drooping
    assert gate.mutation_control.step_jump.caught_with_rule_off == 0

Each of those would still pass if the code were crippled in the right way, which is
the point: they are anchors on the *narrative*, not on performance. The rest of the
file is normal upper bounds.

## Determinism

Every random draw goes through `random.Random(seed)` with the seed written into
`results.json`'s context, or through a Halton sequence built from a fixed base, or
through an explicit grid. Running `bench.py` twice produces byte-identical JSON
apart from `runtime_s`. The MuJoCo section is the exception in principle (an
external integrator) and in practice reproduces to the last digit on this host; if
it ever stops, the runtime is not the thing to suspect.
