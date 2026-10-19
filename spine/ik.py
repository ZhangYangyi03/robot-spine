# -*- coding: utf-8 -*-
"""ik.py -- numerical inverse kinematics, damped least squares with a joint limit
barrier. No solver library: this is 60 lines of the thing itself.

The loop, and why each part is there:

    e = [p_des - p(q) ; log(R(q)^T R_des)]      task error, 6-vector, world frame
    dq = (J^T J + lam^2 I)^-1 J^T e             damped least squares
    lam grows with the error, shrinks as it converges
                                                (large error -> small steps, so the
                                                 arm does not fling itself; small
                                                 error -> full Newton-ish speed)
    dq clipped to +-step                        a single call cannot teleport
    q clamped into [lower, upper]               the first line of defence
    residual re-measured every iteration        convergence is measured, not assumed

The residual is returned as data. Nothing here decides that the answer is good;
ik_solve reports how close it got and the caller (or the gate) judges.
"""
import math

from . import linalg as la


class IKResult:
    def __init__(self, q, iters, pos_err, rot_err, ok, trace):
        self.q = q
        self.iters = iters
        self.pos_err = pos_err        # metres
        self.rot_err = rot_err        # radians
        self.ok = ok                  # converged under the caller's tolerance
        self.trace = trace            # error per iteration -- the evidence

    def __repr__(self):
        return '<IKResult ok=%s it=%d pos=%.2e rot=%.2e>' % (
            self.ok, self.iters, self.pos_err, self.rot_err)


def pose_error(T_cur, p_des, R_des):
    p = [T_cur[i][3] for i in range(3)]
    R = [[T_cur[i][j] for j in range(3)] for i in range(3)]
    dp = la.sub(p_des, p)
    dr = la.rot_err(R, R_des)
    return [dp[0], dp[1], dp[2], dr[0], dr[1], dr[2]]


def _merit(robot, link, q, p_des, R_des, rot_weight):
    T = robot.fk(robot.qdict(q), link)
    R = R_des if R_des is not None else [[T[i][j] for j in range(3)] for i in range(3)]
    e = pose_error(T, p_des, R)
    if R_des is None:
        e[3:] = [0.0, 0.0, 0.0]
    else:
        e[3:] = [x * rot_weight for x in e[3:]]
    pe, re = la.norm(e[:3]), la.norm(e[3:]) / max(rot_weight, 1e-9)
    return e, pe, re


def ik_solve(robot, link, q0, p_des, R_des=None, max_iter=200, tol=1e-6,
             step=0.3, lam0=0.02, rot_weight=0.5, gate=None):
    """Solve for the joint vector that puts `link` at (p_des, R_des).

    Every step is accepted only if it (a) stays inside the joint limits, (b)
    passes `gate` when one is given, and (c) lowers the task error. A step that
    fails any of the three is backtracked; when no fraction of it works, the
    start is declared dead and the residual is returned.

    That line search is not decoration. Without it, a DLS step that pushes a
    joint INTO its limit gets clamped, the solver keeps pushing, and it ends up
    parked on the limit with a large residual -- the failure this repo's own
    bench found on 23% of reachable targets. The clamp was the bug; refusing the
    step is the fix.
    """
    names = robot.joint_names()
    lims = robot.limits()
    n = len(names)
    q = [float(x) for x in q0]
    trace = []
    e, pe, re = _merit(robot, link, q, p_des, R_des, rot_weight)
    trace.append((pe, re))
    it = 0
    for it in range(1, max_iter + 1):
        if pe < tol and re < tol:
            return IKResult(q, it - 1, pe, re, True, trace)
        J = robot.world_jacobian(robot.qdict(q), link)
        if R_des is None:
            for r in range(3, 6):
                J[r] = [0.0] * n
        else:
            for r in range(3, 6):
                J[r] = [x * rot_weight for x in J[r]]
        lam = lam0 * (1.0 + 4.0 * min(1.0, pe + re))
        dq = la.dls(J, e, lam)
        m = max([abs(x) for x in dq] or [0.0])
        if m > step:
            dq = [x * step / m for x in dq]
        merit_now = pe + re
        accepted = None
        alpha = 1.0
        for _ in range(24):
            trial = [q[k] + alpha * dq[k] for k in range(n)]
            if any(trial[k] < lims[k][0] or trial[k] > lims[k][1] for k in range(n)):
                alpha *= 0.5
                continue
            if gate is not None:
                gok, _why = gate(trial)
                if not gok:
                    alpha *= 0.5
                    continue
            e2, pe2, re2 = _merit(robot, link, trial, p_des, R_des, rot_weight)
            if pe2 + re2 < merit_now - 1e-15:
                accepted = (trial, e2, pe2, re2)
                break
            alpha *= 0.5
        if accepted is None:
            return IKResult(q, it, pe, re, False, trace)      # this start is dead
        q, e, pe, re = accepted
        trace.append((pe, re))
    return IKResult(q, it, pe, re, pe < tol and re < tol, trace)


def qlim_violation(robot, qvec):
    """How far outside the joint limits a vector is, in radians/metres (>0 = illegal)."""
    worst = 0.0
    which = None
    for (name, v), (lo, hi) in zip(zip(robot.joint_names(), qvec), robot.limits()):
        over = max(lo - v, v - hi, 0.0)
        if over > worst:
            worst, which = over, name
    return worst, which


def _halton(i, base):
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def starts_from_limits(robot, n_starts):
    """A deterministic spread of starting guesses: zero, then a Halton lattice.

    Deterministic on purpose -- a benchmark whose result changes between runs
    cannot be quoted, and a solver that only works on the lucky random seeds is
    not a solver. The lattice is over the URDF's own limits, so it adapts to the
    arm without being told anything about it.
    """
    lims = robot.limits()
    n = len(lims)
    out = [[0.0] * n]
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for k in range(1, n_starts):
        q = []
        for d in range(n):
            u = _halton(k, primes[d % len(primes)])
            lo, hi = lims[d]
            q.append(lo + u * (hi - lo) * 0.98 + 0.01 * (hi - lo))
        out.append(q)
    return out


def ik_multistart(robot, link, p_des, R_des=None, seeds=8, **kw):
    """Try ik_solve from a deterministic set of starts; keep the best converged.

    Single-start DLS fails on this repo's own bench for a reason the bench names:
    the 2-DOF arm has two elbow branches, and from a zero start one of them is
    unreachable without passing a joint limit. The fix is more guesses, and the
    honest reporting is the single-start rate next to the multi-start rate,
    because the difference is what the extra attempts cost and buy.
    """
    best = None
    tried = 0
    for q0 in starts_from_limits(robot, seeds):
        tried += 1
        r = ik_solve(robot, link, q0, p_des, R_des, **kw)
        if best is None or (r.pos_err + r.rot_err) < (best.pos_err + best.rot_err):
            best = r
        if r.ok:
            break
    return best, tried
