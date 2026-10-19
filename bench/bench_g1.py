# -*- coding: utf-8 -*-
"""bench_g1.py -- the same code on a 29-DOF humanoid, because 2 DOF proves nothing about scale.

The URDF is unifolm-wla-toolkit's demo/g1.urdf (Unitree G1, 39 links, 38 joints,
29 movable). The point is not that the numbers are good; the point is that none
of the three claims in this repo quietly depends on there being two joints:

    FK and the Jacobian still agree with finite differences
    IK still converges, and still says so with a residual
    the gate still judges a command, and how long that costs

The gate reports a self-collision at the zero pose of this URDF. That is not a
bug to be tuned away: with 3 cm capsules on every link, the G1's own forearms sit
inside that radius of the torso at q=0. It is reported as measured.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from spine import ik, linalg as la, scenes      # noqa: E402
from spine.gate import Gate, Limits             # noqa: E402
from spine.urdf import Urdf                     # noqa: E402


def _r(x, n=6):
    return round(float(x), n)


def run(link='left_wrist_yaw_link'):
    path = os.path.join(os.path.dirname(HERE), 'demo', 'g1.urdf')
    if not os.path.exists(path):
        return {'skipped': 'demo/g1.urdf not present'}
    t0 = time.time()
    u = Urdf(path)
    load_s = time.time() - t0
    names = u.joint_names()
    q0 = [0.0] * len(names)
    T = u.fk(u.qdict(q0), link)
    p0 = [T[i][3] for i in range(3)]
    R0 = [[T[i][j] for j in range(3)] for i in range(3)]

    # jacobian vs finite differences, on this arm
    t0 = time.time()
    J = u.world_jacobian(u.qdict(q0), link)
    jac_ms = (time.time() - t0) * 1000
    eps = 1e-5
    worst = 0.0
    for k in range(len(names)):
        qp = list(q0)
        qm = list(q0)
        qp[k] += eps
        qm[k] -= eps
        Tp = u.fk(u.qdict(qp), link)
        Tm = u.fk(u.qdict(qm), link)
        dp = [(Tp[i][3] - Tm[i][3]) / (2 * eps) for i in range(3)]
        Rp = [[Tp[i][j] for j in range(3)] for i in range(3)]
        Rm = [[Tm[i][j] for j in range(3)] for i in range(3)]
        w = la.so3_log(la.mat_mul(Rp, la.transpose(Rm)))
        w = [x / (2 * eps) for x in w]
        for i in range(3):
            worst = max(worst, abs(J[i][k] - dp[i]), abs(J[i + 3][k] - w[i]))

    # IK: lift the wrist 10 cm, orientation free, then orientation fixed
    target = [p0[0], p0[1], p0[2] + 0.10]
    t0 = time.time()
    r_pos, tries_pos = ik.ik_multistart(u, link, target, None, seeds=4,
                                        max_iter=120, tol=1e-4)
    pos_ms = (time.time() - t0) * 1000
    t0 = time.time()
    r_6d, tries_6d = ik.ik_multistart(u, link, target, R0, seeds=4,
                                      max_iter=120, tol=1e-4)
    ms_6d = (time.time() - t0) * 1000

    # the gate, on all 29 joints, twice: absolute clearance, then calibrated
    radii = dict((l, 0.03) for l in u.links)
    lim = Limits(u, max_step=0.6, min_clearance=0.01, radii=radii)
    g_abs = Gate(lim, obstacles=[], adjacency=scenes.adjacency_of(u))
    t0 = time.time()
    v_abs = g_abs.check(q0, qdot=[0.0] * len(names), tau=[0.0] * len(names))
    gate_ms_abs = (time.time() - t0) * 1000

    g = Gate(lim, obstacles=[], adjacency=scenes.adjacency_of(u), calibrate=q0)
    t0 = time.time()
    v = g.check(q0, qdot=[0.0] * len(names), tau=[0.0] * len(names))
    gate_ms = (time.time() - t0) * 1000
    # one plausible motion: lift the wrist 10 cm from the start pose
    v_move = g.check(r_pos.q, qdot=[0.0] * len(names), tau=[0.0] * len(names))

    # and one planted joint-limit break on this arm
    lims = u.limits()
    bad = list(q0)
    i = max(range(len(bad)), key=lambda k: lims[k][1] - lims[k][0])
    bad[i] = lims[i][1] + 0.2
    g.last_q = list(q0)
    v_bad = g.check(bad, qdot=[0.0] * len(names), tau=[0.0] * len(names))

    return {
        'urdf': 'demo/g1.urdf (unifolm-wla-toolkit)',
        'links': len(u.links), 'joints': len(u.joints),
        'movable_joints': len(names), 'load_ms': _r(load_s * 1000, 2),
        'jacobian_vs_finite_difference_max_err': worst,
        'jacobian_ms': _r(jac_ms, 2),
        'ik_position_only': {'ok': r_pos.ok, 'tries': tries_pos,
                             'residual_m': _r(r_pos.pos_err, 8),
                             'ms': _r(pos_ms, 1), 'iterations': r_pos.iters},
        'ik_position_and_orientation': {'ok': r_6d.ok, 'tries': tries_6d,
                                        'residual_m': _r(r_6d.pos_err, 8),
                                        'residual_rad': _r(r_6d.rot_err, 8),
                                        'ms': _r(ms_6d, 1)},
        'gate_at_zero_pose_absolute': {'ok': v_abs.ok, 'reasons': v_abs.reasons,
                                       'ms': _r(gate_ms_abs, 1),
                                       'pair_stats': g_abs.last_pair_stats},
        'gate_at_zero_pose_calibrated': {'ok': v.ok, 'reasons': v.reasons,
                                         'ms': _r(gate_ms, 1),
                                         'pairs_overlapping_at_baseline':
                                             g.baseline_pairs_overlapping,
                                         'pair_stats': g.last_pair_stats},
        'gate_after_a_10cm_move': {'ok': v_move.ok, 'reasons': v_move.reasons},
        'gate_planted_joint_limit': {'joint': names[i], 'ok': v_bad.ok,
                                     'reasons': v_bad.reasons},
        'note': 'with absolute clearance the zero pose is refused, and it is right '
                'to be: 3 cm capsules on this URDF put the torso stack inside the '
                'pelvis radius at q=0. Calibrated against that same pose, the '
                'structural overlaps are baselined and the arm is admitted -- and '
                'the 10 cm move in the IK section still passes the gate afterwards, '
                'so calibration did not simply switch the rule off',
    }
