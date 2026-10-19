# -*- coding: utf-8 -*-
"""gate.py -- the safety gate, as a thing that can be measured rather than trusted.

This is autoforge's shape moved from tools to torque: a command is admitted only
if an independent check says yes, and the check is judged by fault injection
rather than by assertion. The claim "the gate works" is replaced by two numbers:

    legal commands admitted   / legal commands       (must be all of them)
    planted breaks caught     / planted breaks       (must also be all of them,
                                                     and by the RIGHT rule)

A gate that refuses everything scores 100% on the second number and 0% on the
first, which is why both are reported together and why a test asserts the second.

Rules, each returning a reason string, so a refusal says which line it was:

    joint_limit     q outside the URDF's <limit>
    step_jump       one command moves a joint further than max_step rad
    velocity        |qdot| above the servo's rating
    torque          |tau| above the joint's rating
    self_collision  two non-adjacent links of the same arm closer than clearance
    obstacle        a link closer than clearance to an obstacle in the scene

Nothing here knows about the controller. It is handed a command and a state, and
it answers. That is the whole point: the thing being checked does not get a vote.
"""
import math

from . import collision as col
from . import ik


class Limits:
    def __init__(self, robot, max_step=None, vel=None, tau=None,
                 min_clearance=0.02, radii=None):
        self.robot = robot
        names = robot.joint_names()
        lims = robot.limits()
        self.q_min = [lo for lo, _ in lims]
        self.q_max = [hi for _, hi in lims]
        self.vel = dict(vel or {n: 2.0 for n in names})
        self.tau = dict(tau or {n: 20.0 for n in names})
        self.max_step = max_step if max_step is not None else 0.35
        self.min_clearance = min_clearance
        self.radii = radii or {}
        self.names = names


class Verdict:
    __slots__ = ('ok', 'reasons', 'detail')

    def __init__(self, ok, reasons, detail):
        self.ok, self.reasons, self.detail = ok, reasons, detail

    def __bool__(self):
        return self.ok

    def __repr__(self):
        return '<Verdict %s %s>' % ('OK' if self.ok else 'REFUSED', ','.join(self.reasons))


class Gate:
    """A command gate with an obstacle scene and a remembered last command."""

    def __init__(self, limits, obstacles=None, adjacency=(), rules=None,
                 calibrate=None):
        self.lim = limits
        self.robot = limits.robot
        self.obstacles = list(obstacles or [])      # [(centre, radius)] in world
        self.adjacency = set(adjacency)
        self.last_q = None
        # calibrate: a joint vector the machine is known to be healthy in. With
        # it, a pair of capsules is flagged only if it comes closer than it
        # already is at that pose, or inside the absolute clearance if it was
        # outside it. Without it, the rule is absolute. Calibration is what makes
        # a hand-set capsule model usable on a real URDF: the G1's own torso
        # stack overlaps its pelvis with 3 cm capsules at q=0, and that is a
        # property of the model, not of the pose.
        self.calibrate = calibrate
        self._baseline = None
        self.baseline_pairs_overlapping = None
        # rules can be switched off, on purpose, so that the bench can show what
        # each one is actually buying instead of assuming it buys something.
        self.rules = set(rules or
                         ('joint_limit', 'step_jump', 'velocity', 'torque',
                          'self_collision', 'obstacle'))

    # -- individual rules ------------------------------------------------
    def _joint_limit(self, q):
        over, which = ik.qlim_violation(self.robot, q)
        return ({'joint_limit': 'q[%s] outside limit by %.4f' % (which, over)}
                if over > 1e-12 else {})

    def _step_jump(self, q):
        if self.last_q is None:
            return {}
        worst = max([abs(a - b) for a, b in zip(q, self.last_q)] or [0.0])
        return ({'step_jump': 'largest joint move %.4f > %.4f' % (worst, self.lim.max_step)}
                if worst > self.lim.max_step else {})

    def _velocity(self, qdot):
        if qdot is None:
            return {}
        bad = {n: v for n, v in zip(self.lim.names, qdot)
               if abs(v) > self.lim.vel.get(n, 1e9)}
        if not bad:
            return {}
        n, v = max(bad.items(), key=lambda kv: abs(kv[1]))
        return {'velocity': '|qdot[%s]|=%.3f above %.3f' % (n, abs(v), self.lim.vel[n])}

    def _torque(self, tau):
        if tau is None:
            return {}
        bad = {n: t for n, t in zip(self.lim.names, tau)
               if abs(t) > self.lim.tau.get(n, 1e9)}
        if not bad:
            return {}
        n, t = max(bad.items(), key=lambda kv: abs(kv[1]))
        return {'torque': '|tau[%s]|=%.2f above %.2f' % (n, abs(t), self.lim.tau[n])}

    def _pairs(self, q):
        """Distance for every capsule pair that a bounding sphere says can matter."""
        caps = col.robot_capsules(self.robot, self.robot.qdict(q),
                                  self.lim.radii, self.adjacency)
        out = {}
        self.last_pair_stats = {
            'capsules': len(caps),
            'pairs_total': len(caps) * (len(caps) - 1) // 2,
        }
        n_exact = 0
        margin = max(self.lim.min_clearance, 0.05)
        for i, j in col.broad_phase_pairs(caps, margin):
            A, B = caps[i], caps[j]
            if A.name == B.name:
                continue
            if (A.name, B.name) in self.adjacency or (B.name, A.name) in self.adjacency:
                continue
            n_exact += 1
            d, _, _ = col.capsule_capsule(A.p, A.q, A.r, B.p, B.q, B.r,
                                          A.r_end, B.r_end)
            key = (A.name, B.name) if A.name < B.name else (B.name, A.name)
            if key not in out or d < out[key]:
                out[key] = d
        self.last_pair_stats['pairs_exact'] = n_exact
        return out

    def calibrate_at(self, q):
        """Measure every capsule pair at a pose the machine is healthy in."""
        base = self._pairs(q)
        self._baseline = base
        self.baseline_pairs_overlapping = sum(1 for d in base.values()
                                              if d < self.lim.min_clearance)
        return base

    def _self_collision(self, q):
        now = self._pairs(q)
        if self.calibrate is not None and self._baseline is None:
            self.calibrate_at(self.calibrate)
        base = self._baseline or {}
        worst = None
        for key, d in now.items():
            thr = self.lim.min_clearance
            if key in base:
                thr = min(thr, base[key])
            if d < thr - 1e-9 and (worst is None or d - thr < worst[0]):
                worst = (d - thr, key, d, thr)
        if worst is None:
            return {}
        _, key, d, thr = worst
        return {'self_collision': '%s vs %s at %.4f m (limit %.4f)'
                % (key[0], key[1], d, thr)}

    def _obstacle(self, q):
        if not self.obstacles:
            return {}
        caps = col.robot_capsules(self.robot, self.robot.qdict(q),
                                  self.lim.radii, self.adjacency)
        for c, r in self.obstacles:
            for cap in caps:
                d, _, _ = col.capsule_sphere(cap.p, cap.q, min(cap.r, cap.r_end), c, r)
                if d < self.lim.min_clearance:
                    return {'obstacle': '%s at %.4f m from an obstacle'
                            % (cap.name, d)}
        return {}

    # -- the one entry point --------------------------------------------
    def check(self, q, qdot=None, tau=None, commit=False):
        """Judge one command. commit=True remembers it as the last ADMITTED one."""
        reasons, detail = {}, {}
        for rule in ('joint_limit', 'step_jump', 'self_collision', 'obstacle'):
            if rule in self.rules:
                got = getattr(self, '_' + rule)(q)
                if got:
                    reasons.update(got)
                    detail[rule] = list(got.values())[0]
        if 'velocity' in self.rules:
            got = self._velocity(qdot)
            reasons.update(got)
            if got:
                detail['velocity'] = list(got.values())[0]
        if 'torque' in self.rules:
            got = self._torque(tau)
            reasons.update(got)
            if got:
                detail['torque'] = list(got.values())[0]
        ok = not reasons
        if ok and commit:
            self.last_q = list(q)
        return Verdict(ok, sorted(reasons), detail)
