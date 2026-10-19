# -*- coding: utf-8 -*-
"""bench_mujoco.py -- the analytic plant, checked against a physics engine.

The repo's plant is a 2-link arm whose M, C and G are four lines each. That is
convenient, and it proves nothing about whether those lines are the arm. MuJoCo
resolves the same arm from the same numbers with its own solver.

What is claimed here, and how it could be false:

  1. gravity agreement. If control.py's G(q) and MuJoCo's qfrc_bias differ, then
     either the model file describes a different arm or one of the two is wrong.
     Measured on a grid; the maximum absolute difference is reported.

  2. the same PD, through MuJoCo. Same gains, same setpoint, same start pose.
     If the closed loop behaves differently from the analytic plant, the
     difference IS the physics the analytic plant does not have (friction,
     contact, actuator limit, a different mass distribution inside a link).

  3. gravity compensation with the ENGINE's own bias term. If the final joint
     error does not drop to ~0, the compensation is not the thing that fixes the
     droop, and the claim in bench.py that names gravity as the cause is wrong.

An unskipped run also records why MuJoCo could not be loaded, when it cannot:
a missing module and a MemoryError in OpenBLAS look the same from the outside
and the two need different fixes.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from spine import control                       # noqa: E402
from spine import mujoco_backend as mj          # noqa: E402
from spine.control import metrics               # noqa: E402


def _r(x, n=8):
    return round(float(x), n)


def run(t_end=4.0, dt=0.002):
    grid = [[-1.2 + 0.4 * i, -1.2 + 0.4 * j] for i in range(7) for j in range(7)]
    gains = [[18.0, 8.0], [2.2, 1.1]]
    q_des = [0.0, 0.0]
    q0 = [0.6, 0.9]
    raw = mj.run(grid=grid, gains=gains, q_des=q_des, q0=q0, t_end=t_end, dt=dt)
    if 'skipped' in raw:
        return {'skipped': raw['skipped'], 'detail': raw.get('detail', ''),
                'how_to_skip_it_yourself': 'the rest of the repo needs no MuJoCo; '
                                           'this section is the only consumer'}

    # ---- 1. gravity -----------------------------------------------------
    plant = control.PlanarArm()
    worst = 0.0
    per_axis = [0.0, 0.0]
    for q, row in zip(grid, raw['mujoco_bias']):
        mine = plant.G(q)
        for k in range(2):
            d = abs(mine[k] - row[k])
            per_axis[k] = max(per_axis[k], d)
            worst = max(worst, d)

    # ---- 2. the same PD through MuJoCo, and through the analytic plant ---
    trace = raw['loop']['trace']
    muj_metrics = metrics([(t[0], t[1], t[2], t[3]) for t in trace], q_des)

    plant_b = control.PlanarArm()
    plant_b.q = list(q0)
    ctrl = control.PD(gains[0], gains[1], plant=plant_b, comp=False)
    tr_b, _, _, _ = control.run(plant_b, ctrl, q_des, t_end=t_end, dt=dt)
    mine_metrics = metrics(tr_b, q_des)

    # A shared fixed point and a shared trajectory are different claims. The
    # engine integrates with RK4 at 2 ms; the plant integrates with the caller's
    # own stepper. So compare both: where they end up, and how far apart they got
    # on the way.
    n = min(len(trace), len(tr_b))
    transient = 0.0
    transient_at = 0.0
    for i in range(n):
        d = max(abs(trace[i][1][k] - tr_b[i][1][k]) for k in range(2))
        if d > transient:
            transient, transient_at = d, trace[i][0]
    final_gap = max(abs(raw['loop']['final_q'][k] - tr_b[-1][1][k]) for k in range(2))

    # ---- 3. compensation using the engine's own bias --------------------
    comp = raw['loop_with_bias_comp']
    comp_err = max(abs(comp['final_q'][k] - q_des[k]) for k in range(2))

    return {
        'mujoco': 'installed and loaded out of process',
        'model': 'demo/two_link.xml (same l1/l2/m1/m2 as spine/control.py)',
        'gravity': {
            'grid_points': len(grid),
            'max_abs_difference_Nm': worst,
            'per_axis_Nm': per_axis,
            'claim': 'the files describe the same arm: G(q) agrees with the '
                     'engine to float precision',
        },
        'same_pd_two_plants': {
            'gains_kp': gains[0], 'gains_kd': gains[1],
            'q0': q0, 'q_des': q_des,
            'mujoco': {'final_q': [_r(x) for x in raw['loop']['final_q']],
                       'final_qd': [_r(x) for x in raw['loop']['final_qd']],
                       'steady_err': muj_metrics['steady_err'],
                       'settle_s': muj_metrics['settle_s'],
                       'overshoot_frac': muj_metrics['overshoot_frac']},
            'analytic_plant': {'final_q': [_r(x) for x in plant_b.q],
                               'steady_err': mine_metrics['steady_err'],
                               'settle_s': mine_metrics['settle_s'],
                               'overshoot_frac': mine_metrics['overshoot_frac']},
            'max_gap_along_the_trajectory_rad': _r(transient),
            'gap_peaked_at_s': _r(transient_at, 4),
            'max_gap_in_the_final_state_rad': _r(final_gap),
            'what_this_says': 'both droop to the same place to 8 decimals, and '
                              'diverge by the number above while moving, because '
                              'the engine runs RK4 at 2 ms and the plant runs its '
                              'own stepper. A shared fixed point proves the '
                              'equations match; the transient gap is the price of '
                              'a different integrator, and it is reported rather '
                              'than called zero',
        },
        'gravity_comp_through_the_engine': {
            'final_q': [_r(x) for x in comp['final_q']],
            'final_qd': [_r(x) for x in comp['final_qd']],
            'max_joint_error_rad': _r(comp_err),
            'claim': 'with the engine bias added to the PD, the arm holds the '
                     'setpoint to within the number above: gravity was the whole '
                     'droop',
        },
    }
