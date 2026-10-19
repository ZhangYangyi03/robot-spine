# -*- coding: utf-8 -*-
"""bench_gate.py -- two numbers about the safety gate, and how each was obtained.

    admitted / legal commands     a gate that refuses is not safe, it is broken
    caught   / planted breaks     each break planted by hand, per rule

Plus the mutation control, which is the only part that makes the second number
mean anything: for every rule, the same planted faults are re-run with THAT RULE
SWITCHED OFF, and the bench requires the catch rate to collapse. A gate that
still catches a joint-limit break with the joint-limit rule disabled is catching
it by accident, and the number it reports is not about that rule.
"""
import math
import random

from spine import scenes
from spine.gate import Gate, Limits

ALL_RULES = ('joint_limit', 'step_jump', 'velocity', 'torque',
             'self_collision', 'obstacle')


def _legal_walk(robot, lim, n, rng):
    """A walk inside 80% of the limits whose steps are all below max_step."""
    lims = robot.limits()
    n_j = len(lims)
    lo_ok = [0.5 * (lo + hi) - 0.4 * (hi - lo) for lo, hi in lims]
    hi_ok = [0.5 * (lo + hi) + 0.4 * (hi - lo) for lo, hi in lims]
    q = [0.5 * (lo + hi) for lo, hi in lims]
    out = []
    for _ in range(n):
        last = list(q)
        nxt = []
        for i in range(n_j):
            d = rng.uniform(-0.9, 0.9) * lim.max_step
            v = q[i] + d
            if v < lo_ok[i] or v > hi_ok[i]:          # reflect, keep it legal
                v = q[i] - d
            nxt.append(min(hi_ok[i], max(lo_ok[i], v)))
        out.append((last, nxt))
        q = nxt
    return out


def _plant(rule, robot, lim, rng, last):
    """One command that breaks exactly `rule`."""
    lims = robot.limits()
    i = rng.randrange(len(lims))
    lo, hi = lims[i]
    q = list(last)
    if rule == 'joint_limit':
        q[i] = hi + rng.uniform(0.01, 0.30)
    elif rule == 'step_jump':
        d = lim.max_step + rng.uniform(0.02, 0.20)
        q[i] = min(hi - 1e-3, max(lo + 1e-3, q[i] + d))
        if abs(q[i] - last[i]) <= lim.max_step:
            q[i] = max(lo + 1e-3, min(hi - 1e-3, last[i] - d))
    else:
        raise ValueError(rule)
    return q


def run(n_legal=200, n_faults=40, seed=7):
    rng = random.Random(seed)
    robot, radii = scenes.two_link()
    names = robot.joint_names()
    n_j = len(names)
    adj = scenes.adjacency_of(robot)
    lim = Limits(robot, max_step=0.35, min_clearance=0.02, radii=radii)

    # ---- 1. legal commands, empty table (no obstacle, no self-collision rule:
    #         an arm folded onto itself really is a collision, so that rule is
    #         measured in its own scene below rather than smuggled in here) ----
    gate = Gate(lim, obstacles=[], adjacency=adj,
                rules=('joint_limit', 'step_jump', 'velocity', 'torque'))
    walk = _legal_walk(robot, lim, n_legal, rng)
    admitted = 0
    refusals = []
    for last, q in walk:
        gate.last_q = last
        v = gate.check(q, qdot=[0.0] * n_j, tau=[0.0] * n_j)
        if v.ok:
            admitted += 1
        elif len(refusals) < 3:
            refusals.append({'q': [round(x, 4) for x in q], 'reasons': v.reasons})
    legal = {'commands': len(walk), 'admitted': admitted,
             'false_refusal_rate': round(1.0 - admitted / max(1, len(walk)), 6),
             'examples_refused': refusals}

    # ---- 2. planted faults ----------------------------------------------
    faults = []
    for rule in ('joint_limit', 'step_jump'):
        for _ in range(n_faults):
            last = [0.0] * n_j
            faults.append((rule, last, _plant(rule, robot, lim, rng, last)))

    per_rule = {}
    for rule in ('joint_limit', 'step_jump'):
        caught = by_intended = 0
        for r, last, q in faults:
            if r != rule:
                continue
            gate.last_q = last
            v = gate.check(q, qdot=[0.0] * n_j, tau=[0.0] * n_j)
            if not v.ok:
                caught += 1
                if rule in v.reasons:
                    by_intended += 1
        per_rule[rule] = {'planted': n_faults, 'caught': caught,
                          'caught_by_intended_rule': by_intended}

    # velocity / torque: one planted fault each, on a still arm
    gate.last_q = [0.0] * n_j
    qdot_bad = [lim.vel[names[0]] * 1.5] + [0.0] * (n_j - 1)
    v_vel = gate.check([0.0] * n_j, qdot=qdot_bad, tau=[0.0] * n_j)
    tau_bad = [lim.tau[names[0]] * 1.5] + [0.0] * (n_j - 1)
    v_tau = gate.check([0.0] * n_j, qdot=[0.0] * n_j, tau=tau_bad)
    per_rule['velocity'] = {'planted': 1, 'caught': int(not v_vel.ok),
                            'caught_by_intended_rule': int('velocity' in v_vel.reasons),
                            'reasons': v_vel.reasons}
    per_rule['torque'] = {'planted': 1, 'caught': int(not v_tau.ok),
                          'caught_by_intended_rule': int('torque' in v_tau.reasons),
                          'reasons': v_tau.reasons}

    # ---- 3. mutation control --------------------------------------------
    mutation = {}
    for rule in ('joint_limit', 'step_jump'):
        g2 = Gate(lim, obstacles=[], adjacency=adj,
                  rules=tuple(r for r in ALL_RULES if r != rule))
        caught = 0
        for r, last, q in faults:
            if r != rule:
                continue
            g2.last_q = last
            if not g2.check(q, qdot=[0.0] * n_j, tau=[0.0] * n_j).ok:
                caught += 1
        mutation[rule] = {'planted': n_faults, 'caught_with_rule_off': caught}

    # ---- 4. the geometry rules, in a scene built to trigger them ---------
    # obstacle at the arm's forward reach; q=0 reaches it, q=pi points away
    gate_o = Gate(lim, obstacles=[([0.75, 0.0, 0.1], 0.10)], adjacency=adj)
    reach = gate_o.check([0.0, 0.0], qdot=[0.0, 0.0], tau=[0.0, 0.0])
    away = gate_o.check([3.0, 0.0], qdot=[0.0, 0.0], tau=[0.0, 0.0])
    # folded: the forearm lies back along the upper arm
    gate_s = Gate(lim, obstacles=[], adjacency=adj)
    folded = gate_s.check([0.0, 3.0], qdot=[0.0, 0.0], tau=[0.0, 0.0])
    straight = gate_s.check([0.0, 0.0], qdot=[0.0, 0.0], tau=[0.0, 0.0])
    geometry = {
        'obstacle_at_reach': {'admitted': reach.ok, 'reasons': reach.reasons},
        'obstacle_away': {'admitted': away.ok, 'reasons': away.reasons},
        'folded_self_collision': {'admitted': folded.ok, 'reasons': folded.reasons},
        'straight_clear': {'admitted': straight.ok, 'reasons': straight.reasons},
    }
    per_rule['obstacle'] = {'planted': 1, 'caught': int(not reach.ok),
                            'caught_by_intended_rule': int('obstacle' in reach.reasons),
                            'reasons': reach.reasons}
    per_rule['self_collision'] = {'planted': 1, 'caught': int(not folded.ok),
                                  'caught_by_intended_rule':
                                      int('self_collision' in folded.reasons),
                                  'reasons': folded.reasons}

    return {'legal': legal, 'per_rule': per_rule, 'mutation_control': mutation,
            'geometry_scene': geometry}
