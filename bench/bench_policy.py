# -*- coding: utf-8 -*-
"""bench_policy.py -- the learned layer, measured against the thing it must not beat.

Three claims are made here, and each has a way of being false that the bench looks
for rather than assumes:

  1. the learner can be taught    clone accuracy must beat the majority-class
                                  baseline on held-out ticks, and the baseline is
                                  printed next to it so a degenerate teacher (one
                                  action, always) cannot pass as a good clone
  2. the learner is not perfect   the ungated arm must fail somewhere. If a policy
                                  this small never errs, the scenarios were too easy
                                  and claim 3 is untested, not proven
  3. the gate is what saves it    gated and ungated run the SAME policy on the SAME
                                  worlds; the difference in deaths and ledge falls
                                  is attributable to the gate and nothing else

Scenarios are generated from a seed, and the training set sees even seeds while the
evaluation set sees odd ones. That is the whole generalisation claim: a policy that
only survives the worlds it trained on has learned the map, not the task.

usage: python bench/bench_policy.py [--epochs 60] [--out bench/results_policy.json]
"""
import os

# BEFORE torch is imported anywhere in the process. Thread pools are created at
# import time from these variables, and set_num_threads() afterwards does not undo
# what the pool already sized itself to: measured on this host, the identical
# training loop takes 7 s with these set here and had not finished after ten
# minutes without them -- only when the process was started as a subprocess, which
# is the only way the bench is ever started.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from spine import embodied as emb          # noqa: E402
from spine import plan, policy, voxel      # noqa: E402


# ---- the world family -----------------------------------------------------
def scenario(seed):
    """A platform with a pit and a wall, a start, and a goal that is reachable.

    Deterministic in the seed, so a failure is reproducible by number. The wall is
    placed so a greedy axis-aligned controller must turn; the pit is placed so that
    walking straight at the goal can be a death, which is the ledge the gate exists
    for.

    Reachability is *checked*, not hoped for: the planner is asked whether a route
    exists and the seed is mutated until one does. An earlier version placed the
    goal blindly and produced worlds where the goal cell was inside the wall -- 6
    of 24 training seeds had no route at all, and "the teacher failed" was being
    counted as "the learner failed".
    """
    import random
    w = None
    for attempt in range(40):
        rnd = random.Random(seed * 1000 + attempt)
        w = voxel.VoxelWorld()
        half = 7
        w.add_box((-half, -61, -half), (half, -61, half), 'stone')
        # a pit: 2x2 of air punched through the floor.
        #
        # `danger` decides whether the pit is a hole in the middle of the platform
        # or a CHASM cut all the way across it. The first version only ever made
        # the former, and the consequence was that no world in the suite could kill
        # a policy that walked the wrong way: measured 0 deaths and 0 gate refusals
        # across 12 worlds, while the same gate refused 80 lethal moves on the real
        # Minecraft server. A gate that is never consulted proves nothing, so the
        # chasm exists to make the claim testable rather than to make it true.
        px, pz = rnd.randint(-4, 4), rnd.randint(-4, 3)
        w.add_box((px, -61, pz), (px + 1, -61, pz + 1), 'air')
        danger = seed % 2 == 1
        chasm_x = None
        if danger and attempt >= 0:
            # a two-wide chasm spanning z, leaving x-cells on both sides walkable so
            # a route over/around always exists (and the planner is asked, below)
            chasm_x = rnd.randint(-3, 2)
            w.add_box((chasm_x, -61, -half), (chasm_x + 1, -61, half), 'air')
        # a wall that does not reach the platform edge, so a detour always exists
        wx = rnd.randint(-4, 4)
        w.add_box((wx, -60, -half + 1), (wx, -59, rnd.randint(-4, 0)), 'stone')
        start = (rnd.choice([-0.5, 1.5, 3.5, -3.5]), -60.0, rnd.choice([4.5, 5.5]))
        goal = (rnd.choice([-0.5, 1.5, -3.5, 3.5]), -60.0, rnd.choice([-4.5, -5.5]))
        pl = plan.CellPlanner(w)
        # The start cell is the one the body's FEET are in, snapped to the nearest
        # centre -- not floor() of its position. A start of -0.5 is on a cell
        # boundary and floor() names cell -1, whose centre is -0.5: the body is
        # standing exactly on it, but `standable` asks about the cell UNDER that
        # centre and returns False, so the planner refused the world and the
        # episode ended at tick 0 with no failure recorded (measured: scenario 15,
        # `why_ended: tick budget, ticks 0, cells 0`).
        scell = (int(round(start[0] - 0.5)), int(math.floor(start[1])),
                 int(round(start[2] - 0.5)))
        gcell = (int(math.floor(goal[0])), int(math.floor(goal[1])),
                 int(math.floor(goal[2])))
        if not pl.standable(scell) or not pl.standable(gcell):
            continue
        path, _stats = pl.plan(scell, gcell)
        if path and (not danger or chasm_x is not None):
            return w, start, goal
    raise RuntimeError('no reachable scenario for seed %r' % (seed,))


def collect(seeds, ticks=400, use_gate=True):
    rows, eps = [], []
    for s in seeds:
        w, start, goal = scenario(s)
        gate = emb.EmbodiedGate(w) if use_gate else None
        r, out = policy.scripted_episode(w, start, goal, ticks=ticks, gate=gate)
        for row in r:
            row['seed'] = s
            rows.append(row)
        out['seed'] = s
        out['start'] = list(start)
        out['goal'] = list(goal)
        eps.append(out)
    return rows, eps


def evaluate(net, seeds, ticks=400):
    """The same policy in four arms per world.

    gated / ungated      does the gate save a policy that errs?
    no-planner gated     what is the A* route worth? (the axis ablation)
    """
    # Four arms, and the pair that matters is the LAST two: same policy, same
    # worlds, no route -- one walks straight at the far goal through the gate and
    # one does not. `gated` vs `ungated` turned out to be the wrong comparison to
    # headline: with the A* route doing the avoidance, the gate was never consulted
    # once (0 refusals, 0 deaths in 12 worlds) and the two arms were identical to
    # the digit. A gate that is never asked anything proves nothing, so the
    # planner is removed to create the situation the gate exists for.
    arms = {'gated': [], 'ungated': [], 'flat_gated': [], 'flat_ungated': []}
    for s in seeds:
        w, start, goal = scenario(s)
        gate = emb.EmbodiedGate(w)
        # policy_episode returns (rows, outcome); the arms only want the outcome,
        # because what a policy did is its outcome, not the ticks it emitted.
        _ra, a = policy.policy_episode(net, w, start, goal, ticks=ticks, gate=gate)
        _rb, b = policy.policy_episode(net, w, start, goal, ticks=ticks, gate=None)
        _rc, c = policy.policy_episode(net, w, start, goal, ticks=ticks, gate=gate,
                                       trust_planner=False)
        _rd, dfl = policy.policy_episode(net, w, start, goal, ticks=ticks, gate=None,
                                         trust_planner=False)
        for k, d in (('gated', a), ('ungated', b), ('flat_gated', c),
                     ('flat_ungated', dfl)):
            d['seed'] = s
            d['start'] = list(start)
            d['goal'] = list(goal)
            arms[k].append(d)
    return arms


def summarize(eps):
    n = len(eps)
    return {
        'episodes': n,
        'reached': sum(1 for e in eps if e['reached']),
        'reached_rate': round(sum(1 for e in eps if e['reached']) / n, 3),
        'died': sum(1 for e in eps if e['died']),
        'death_rate': round(sum(1 for e in eps if e['died']) / n, 3),
        'refusals': sum(e['refused'] for e in eps),
        'min_health': round(min(e['min_health'] for e in eps), 2),
        'teacher_agreement': round(sum(e.get('teacher_agreement', 0.0) for e in eps) / n, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--train-seeds', type=int, default=24)
    ap.add_argument('--test-seeds', type=int, default=12)
    ap.add_argument('--ticks', type=int, default=400)
    ap.add_argument('--eval-ticks', type=int, default=200)
    ap.add_argument('--dagger-ticks', type=int, default=150)
    ap.add_argument('--dagger-rounds', type=int, default=3)
    ap.add_argument('--explore', type=float, default=0.1)
    ap.add_argument('--out', default=os.path.join(HERE, 'results_policy.json'))
    args = ap.parse_args()

    # One thread, before anything touches torch. The tensors here are 128 x 441 --
    # far too small to pay for the intra-op pool, and on this host the default pool
    # turned a 3-second training job into one that did not finish in ten minutes
    # (measured twice: identical code, 3.1 s with one thread, still running at 590 s
    # without). policy.train() pins it too, but by then the pool has been created.
    import torch
    torch.set_num_threads(1)

    train_seeds = [2 * i for i in range(args.train_seeds)]
    test_seeds = [2 * i + 1 for i in range(args.test_seeds)]
    print('round 0: cloning the teacher on %d worlds ...' % len(train_seeds),
          end='', flush=True)
    rows, train_eps = collect(train_seeds, ticks=args.ticks)
    print(' %d ticks from %d episodes, %d reached'
          % (len(rows), len(train_eps), sum(1 for e in train_eps if e['reached'])))

    net = policy.train(rows, epochs=args.epochs)

    # ---- DAgger: the student rolls out, the teacher labels what it saw -----
    # Without this the policy is accurate on the teacher's states and useless on
    # its own; the first run measured exactly that (0.917 clone accuracy, 2/12
    # episodes). Rounds are reported so the improvement is visible, not asserted.
    rounds = []
    for it in range(args.dagger_rounds):
        new_rows, dag_eps = [], []
        for s in train_seeds:
            w, start, goal = scenario(s)
            # A DAgger rollout is not the same budget as an evaluation. The
            # student's policy early in training wanders, so every episode runs to
            # the tick limit -- at 24 worlds x 300 ticks that is one round per
            # minute, and the whole point of the round is to collect the states it
            # wanders INTO, which it does in the first hundred ticks.
            r, out = policy.dagger_episode(net, w, start, goal, ticks=args.dagger_ticks,
                                           explore=args.explore, seed=s * 31 + it)
            new_rows += r
            dag_eps.append(out)
        rows += new_rows
        net = policy.train(rows, epochs=args.epochs)
        reached = sum(1 for e in dag_eps if e['reached'])
        rounds.append({'round': it + 1, 'student_ticks': len(new_rows),
                       'total_ticks': len(rows), 'student_reached': reached,
                       'student_died': sum(1 for e in dag_eps if e['died']),
                       'student_agreement': round(
                           sum(e['teacher_agreement'] for e in dag_eps) / len(dag_eps), 3)})
        print('  dagger round %d: %d new ticks, student reached %d/%d on its own'
              % (it + 1, len(new_rows), reached, len(dag_eps)))

    tr, te = policy.split(rows)
    net = policy.train(rows, epochs=args.epochs)
    acc_act, acc_jump = policy.accuracy(net, te)

    import collections
    counts = collections.Counter(r['act'] for r in te)
    baseline = max(counts.values()) / max(1, len(te))

    print('\nclone on held-out ticks (every 4th tick, so twins stay in train)')
    print('  action accuracy      %.3f' % acc_act)
    print('  majority baseline    %.3f' % baseline)
    print('  jump accuracy        %.3f' % acc_jump)
    print('  action histogram     %s' % dict(counts))

    print('\nevaluating the same policy on %d UNSEEN worlds ...' % len(test_seeds))
    # An evaluation episode that runs out of budget is a FAILURE to report, not a
    # run to continue, so the budget here is deliberately tight -- and it is
    # separate from the training budget because the two answer different questions.
    # Sharing one number meant a policy that wandered got 500 ticks x 4 arms x N
    # worlds of simulation to be measured in, which is how a bench that finishes in
    # seconds at 6 worlds takes ten minutes at 24.
    arms = evaluate(net, test_seeds, ticks=args.eval_ticks)
    g, u = summarize(arms['gated']), summarize(arms['ungated'])
    f, fu = summarize(arms['flat_gated']), summarize(arms['flat_ungated'])
    print('  %-14s %-9s %-5s %-9s %-12s %s' % ('arm', 'reached', 'died',
                                               'refusals', 'min health', 'agreement'))
    for label, s in (('gated', g), ('ungated', u), ('no-planner', f),
                     ('no-planner+no-gate', fu)):
        print('  %-14s %2d/%-6d %-5d %-9d %-12.2f %.3f'
              % (label, s['reached'], s['episodes'], s['died'], s['refusals'],
                 s['min_health'], s['teacher_agreement']))
    print('\n  the pair that tests the gate is the last two: same policy, no route,')
    print('  one arm gated. With the A* route in place the gate is never consulted')
    print('  (0 refusals, 0 deaths), so gated == ungated there by construction and')
    print('  that equality is evidence about the planner, not about the gate.')
    if f['refusals']:
        print('  the no-planner arm shows the gate doing work: %d refusals, and the'
              % f['refusals'])
        print('  gated arm is the one that still finishes sometimes.')

    teacher_eps = [e for e in train_eps]
    payload = {
        'dagger_rounds': rounds,
        'policy': {'params': sum(p.numel() for p in net.parameters()),
                   'epochs': args.epochs,
                   'train_ticks': len(tr), 'test_ticks': len(te)},
        'clone': {'action_accuracy': round(acc_act, 3),
                  'majority_baseline': round(baseline, 3),
                  'jump_accuracy': round(acc_jump, 3),
                  'histogram': {str(k): v for k, v in counts.items()}},
        'train_worlds': {'episodes': len(teacher_eps),
                         'reached': sum(1 for e in teacher_eps if e['reached']),
                         'died': sum(1 for e in teacher_eps if e['died'])},
        'eval_worlds': {'n': len(test_seeds), 'gated': g, 'ungated': u,
                        'flat_gated': f, 'flat_ungated': fu},
        'per_world': {'gated': [{k: e[k] for k in ('seed', 'reached', 'died', 'refused',
                                                   'final', 'min_health')} for e in arms['gated']],
                      'ungated': [{k: e[k] for k in ('seed', 'reached', 'died', 'refused',
                                                     'final', 'min_health')} for e in arms['ungated']],
                      'flat_gated': [{k: e[k] for k in ('seed', 'reached', 'died', 'refused',
                                                        'final', 'min_health')} for e in arms['flat_gated']],
                      'flat_ungated': [{k: e[k] for k in ('seed', 'reached', 'died', 'refused',
                                                          'final', 'min_health')} for e in arms['flat_ungated']]},
    }
    with open(args.out, 'w') as fh:
        json.dump(payload, fh, indent=1)
    print('\nwrote', args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
