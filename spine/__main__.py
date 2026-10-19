# -*- coding: utf-8 -*-
"""python -m spine -- one command that exercises all four stops on a URDF.

    python -m spine                                   # the built-in 2-link arm
    python -m spine demo/g1.urdf --link left_wrist_yaw_link --dz 0.10
    python -m spine demo/g1.urdf --link left_wrist_yaw_link --dz 0.10 --json

Prints, in order: the model, the target, what IK returned (including its residual),
and what the gate says about the answer. The gate is asked LAST and its answer is
printed even when IK failed, because "the solver could not" and "the solution is
unsafe" are different results and must not be collapsed into one line.
"""
import argparse
import json
import sys
import time

from . import ik, linalg as la, scenes
from .gate import Gate, Limits
from .urdf import Urdf


def _fmt(v, n=6):
    return ('%.' + str(n) + 'f') % v


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m spine')
    ap.add_argument('urdf', nargs='?', help='URDF path (omit for the built-in 2-link arm)')
    ap.add_argument('--link', help='the link to place (default: the tip / last link)')
    ap.add_argument('--dz', type=float, default=0.0, help='move the target up by this many metres')
    ap.add_argument('--dx', type=float, default=0.0, help='move the target right by this many metres')
    ap.add_argument('--dy', type=float, default=0.0, help='move the target forward by this many metres')
    ap.add_argument('--orient', action='store_true', help='also hold the tool orientation')
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--tol', type=float, default=1e-6)
    ap.add_argument('--radius', type=float, default=0.03, help='capsule radius for every link')
    ap.add_argument('--clearance', type=float, default=0.02)
    ap.add_argument('--no-calibrate', action='store_true',
                    help='absolute clearance only, no comparison to the start pose')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)

    t0 = time.time()
    if a.urdf:
        robot = Urdf(a.urdf)
        radii = dict((l, a.radius) for l in robot.links)
        radii[robot.root] = max(a.radius, 0.10)
    else:
        robot, radii = scenes.two_link()
    link = a.link or [l for l in robot.links if l in robot.by_child][-1]

    names = robot.joint_names()
    q0 = [0.0] * len(names)
    T = robot.fk(robot.qdict(q0), link)
    p0 = [T[i][3] for i in range(3)]
    R0 = [[T[i][j] for j in range(3)] for i in range(3)]
    target = [p0[0] + a.dx, p0[1] + a.dy, p0[2] + a.dz]

    res, tries = ik.ik_multistart(robot, link, target, R0 if a.orient else None,
                                  seeds=a.seeds, tol=a.tol, max_iter=200)

    lim = Limits(robot, min_clearance=a.clearance, radii=radii)
    # a real arm is healthy at the pose it is started from; that pose is also the
    # only ground truth about how fat the hand-set capsules have to be
    gate = Gate(lim, obstacles=[], adjacency=scenes.adjacency_of(robot),
                calibrate=(q0 if a.urdf and not a.no_calibrate else None))
    verdict = gate.check(res.q, qdot=[0.0] * len(names), tau=[0.0] * len(names))
    ms = (time.time() - t0) * 1000

    out = {
        'robot': {'name': robot.name, 'links': len(robot.links),
                  'joints': len(robot.joints), 'movable': len(names),
                  'root': robot.root},
        'link': link,
        'target': [round(x, 6) for x in target],
        'start': [round(x, 6) for x in p0],
        'ik': {'ok': res.ok, 'tries': tries, 'iterations': res.iters,
               'residual_m': round(res.pos_err, 9),
               'residual_rad': round(res.rot_err, 9),
               'q': [round(x, 6) for x in res.q]},
        'gate': {'ok': verdict.ok, 'reasons': verdict.reasons, 'detail': verdict.detail},
        'ms': round(ms, 1),
    }
    if a.json:
        print(json.dumps(out, indent=2))
        return 0 if (res.ok and verdict.ok) else 1

    print('robot     %s  %d links / %d joints (%d movable), root=%s'
          % (robot.name, len(robot.links), len(robot.joints), len(names), robot.root))
    print('link      %s' % link)
    print('target    [%s]  (start [%s])'
          % (', '.join(_fmt(x, 4) for x in target), ', '.join(_fmt(x, 4) for x in p0)))
    print('ik        %s  tries=%d  iterations=%d  residual %.2e m / %.2e rad'
          % ('converged' if res.ok else 'NOT converged', tries, res.iters,
             res.pos_err, res.rot_err))
    print('gate      %s  %s' % ('admitted' if verdict.ok else 'REFUSED',
                                ', '.join(verdict.reasons) if verdict.reasons else 'no rule fired'))
    for rule, why in verdict.detail.items():
        print('          %-14s %s' % (rule, why))
    print('time      %.1f ms' % ms)
    return 0 if (res.ok and verdict.ok) else 1


if __name__ == '__main__':
    sys.exit(main())
