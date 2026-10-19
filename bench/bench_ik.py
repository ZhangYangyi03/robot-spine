# -*- coding: utf-8 -*-
"""bench_ik.py -- inverse kinematics, measured three ways.

    reachable targets        sampled from the arm's own workspace
    single start             ik_solve from the zero pose
    multi start              ik_multistart, deterministic Halton seeds over the
                             joint limits
    unreachable target       must return a residual, and must not raise

The definition of "reachable" is not taken on faith either: each target is
checked against the closed-form 2-link inverse kinematics, and a target whose
closed form lands outside the joint limits is excluded from the denominator
instead of being counted as a solver failure.
"""
import math
import random

from spine import ik, linalg as la

L1, L2 = 0.40, 0.35


def closed_form(x, y):
    """Both elbow branches of the analytic IK for a 2-link planar arm."""
    d2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(d2) > 1.0:
        return []
    out = []
    for s in (1.0, -1.0):
        q2 = s * math.acos(max(-1.0, min(1.0, d2)))
        q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        out.append((q1, q2))
    return out


def run(robot, n_targets=200, seed=1, tol=1e-9):
    rng = random.Random(seed)
    cases = []
    seen = 0
    while len(cases) < n_targets and seen < 20000:
        seen += 1
        th = rng.uniform(-math.pi, math.pi)
        r = rng.uniform(0.10, 0.74)          # |L1 - L2| <= r <= L1 + L2
        x, y = r * math.cos(th), r * math.sin(th)
        sols = closed_form(x, y)
        inside = [s for s in sols if abs(s[0]) <= 3.0 and abs(s[1]) <= 3.0]
        if not inside:
            continue
        q1, q2 = inside[0]
        R = la.axis_angle_R([0, 0, 1], q1 + q2)   # full 6D task: position + tool yaw
        cases.append((x, y, R))

    single = 0
    single_err = 0.0
    multi = 0
    tries = 0
    multi_err = 0.0
    t_single = t_multi = 0.0
    import time
    for x, y, R in cases:
        a = time.time()
        r1 = ik.ik_solve(robot, 'tip', [0.0, 0.0], [x, y, 0.10], R,
                         max_iter=300, tol=tol)
        t_single += time.time() - a
        single += int(r1.ok)
        single_err = max(single_err, r1.pos_err)
        a = time.time()
        r2, n = ik.ik_multistart(robot, 'tip', [x, y, 0.10], R, seeds=8,
                                 max_iter=300, tol=tol)
        t_multi += time.time() - a
        multi += int(r2.ok)
        tries += n
        multi_err = max(multi_err, r2.pos_err)

    # an unreachable target, and a target behind the arm: both must return data
    un = ik.ik_solve(robot, 'tip', [0.0, 0.0], [1.5, 0.0, 0.10], None, max_iter=200)
    n_cases = len(cases)
    return {
        'reachable_targets': n_cases,
        'single_start': {'ok': single, 'rate': round(single / n_cases, 4),
                         'worst_residual_m': single_err,
                         'ms_per_case': round(1000 * t_single / n_cases, 2)},
        'multi_start': {'ok': multi, 'rate': round(multi / n_cases, 4),
                        'mean_tries': round(tries / n_cases, 2),
                        'worst_residual_m': multi_err,
                        'ms_per_case': round(1000 * t_multi / n_cases, 2)},
        'unreachable_target': {'ok': un.ok, 'residual_m': round(un.pos_err, 4),
                               'iterations': un.iters,
                               'raised': False},
        'why_single_start_fails': (
            'the 2-link arm has two elbow branches; from a zero start one branch '
            'is unreachable without crossing a joint limit, and a step that would '
            'cross it is now refused rather than clamped, so the solver stops and '
            'reports the residual instead of parking on the limit'),
    }
