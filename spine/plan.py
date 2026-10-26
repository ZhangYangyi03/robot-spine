# -*- coding: utf-8 -*-
"""plan.py -- a route through a voxel world, and the executor that spends it.

This is the layer between the two ends of spine-agent:

    a plan is cells,          [ (x, y, z), (x, y, z), ... ] meaning "stand here"
    an intent is one tick,    {'dv': ..., 'jump': ..., 'sneak': ...}
    a tick is judged,         EmbodiedGate.check(state, intent)
    and then it is spent,     Env.tick(intent)

The planner is A* over a graph whose edges are a body's legitimate moves: walk to
a neighbouring cell, step up one, drop up to `max_drop`. It is where "no route" is
decided, and it is decided by the same constants the gate uses, so a route the
planner emits and a route the gate refuses cannot both be right -- the bench counts
both and prints the difference.

Deliberate limits, because a planner that quietly cheats here would make every
number downstream meaningless:

  * it does not plan jumps over gaps. A gap either has a bridge or the route does
    not exist; the executor, not the planner, is where an attempt to jump a canyon
    would be ignored. (Jumping gaps is worth having and is not here.)
  * it does not know about monsters, water flows, or falling blocks.
  * diagonal moves are not in the graph. The game allows them; this does not, and
    the routes are 41% longer for it -- measured in the bench, not estimated.
"""
import heapq
import math

from . import embodied as emb
from . import voxel

STILL = (0.0, 0.0, 0.0)


class CellPlanner:
    """A* over cells a body can stand in."""

    def __init__(self, world, max_step_up=1, max_drop=3):
        self.world = world
        self.max_step_up = max_step_up
        self.max_drop = max_drop
        self.expansions = 0
        self.cache = {}

    # -- what counts as standable ----------------------------------------
    def standable(self, cell):
        """A cell is standable when the block UNDER it is solid, the two cells the
        body occupies (feet and head) are empty, and neither of them hurts.

        The cell convention is the feet's cell: a body standing on the top block at
        y=64 has its feet at y=65.0, which is cell 65, and the Env agrees -- the
        first version of this function used "the cell itself must be solid", which
        made the feet-cell unwalkable and every plan empty.
        """
        hit = self.cache.get(cell)
        if hit is not None:
            return hit
        x, y, z = cell
        ok = (self.world.solid((x, y - 1, z))
              and not self.world.solid((x, y, z))
              and not self.world.solid((x, y + 1, z))
              and voxel.hazard_of(self.world.blocks.get((x, y - 1, z))) == 0.0
              and voxel.hazard_of(self.world.blocks.get((x, y, z))) == 0.0
              and voxel.hazard_of(self.world.blocks.get((x, y + 1, z))) == 0.0)
        self.cache[cell] = ok
        return ok

    def neighbours(self, cell):
        x, y, z = cell
        out = []
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            # same level
            same = (x + dx, y, z + dz)
            if self.standable(same):
                out.append((same, 1.0))
                continue
            # step up exactly one
            up = (x + dx, y + 1, z + dz)
            # standable(up) already requires the head cell (up_y + 1) to be empty,
            # which is the whole check; an earlier copy of it here said the same
            # thing twice and would have hidden it if one of them changed.
            if self.max_step_up >= 1 and self.standable(up):
                out.append((up, 1.3))
                continue
            # step down 1..max_drop
            for d in range(1, self.max_drop + 1):
                down = (x + dx, y - d, z + dz)
                if self.standable(down):
                    out.append((down, 1.0 + 0.1 * d))
                    break               # the first floor is the one you land on
        return out

    # -- A* --------------------------------------------------------------
    def plan(self, start, goal, max_expansions=200000):
        """Cells from `start` to `goal` (both standable), or ([], stats)."""
        self.expansions = 0
        if not self.standable(start):
            return [], {'ok': False, 'why': 'start cell %s is not standable' % (start,),
                        'expansions': 0}
        if not self.standable(goal):
            return [], {'ok': False, 'why': 'goal cell %s is not standable' % (goal,),
                        'expansions': 0}

        def h(c):
            return abs(c[0] - goal[0]) + abs(c[1] - goal[1]) + abs(c[2] - goal[2])

        openq = [(h(start), 0.0, start)]
        g = {start: 0.0}
        came = {}
        while openq:
            _, gc, cur = heapq.heappop(openq)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return list(reversed(path)), {'ok': True, 'expansions': self.expansions}
            if gc > g.get(cur, float('inf')):
                continue
            self.expansions += 1
            if self.expansions > max_expansions:
                return [], {'ok': False, 'why': 'expansion budget exhausted',
                            'expansions': self.expansions}
            for nxt, cost in self.neighbours(cur):
                ng = gc + cost
                if ng < g.get(nxt, float('inf')) - 1e-12:
                    g[nxt] = ng
                    came[nxt] = cur
                    heapq.heappush(openq, (ng + h(nxt), ng, nxt))
        return [], {'ok': False, 'why': 'no route: the graph has no path',
                    'expansions': self.expansions}


def intents_for(path, walk=emb.WALK):
    """Turn a cell path into the intents the executor will try, one per move.

    Kept separate from the executor so that a plan can be inspected (and gated)
    without a plant, which is how the bench measures "the planner proposed X and
    the gate refused Y" without running the body.
    """
    out = []
    for a, b in zip(path, path[1:]):
        dy = b[1] - a[1]
        dx, dz = b[0] - a[0], b[2] - a[2]
        n = math.hypot(dx, dz) or 1.0
        dv = [dx / n * walk, 0.0, dz / n * walk]
        yaw = math.atan2(dz, dx)
        if dy == 1:
            out.append({'kind': 'step_up', 'from': a, 'to': b, 'yaw': yaw,
                        'act': {'dv': dv, 'jump': True, 'yaw': yaw}})
        elif dy == 0:
            out.append({'kind': 'walk', 'from': a, 'to': b, 'yaw': yaw,
                        'act': {'dv': dv, 'yaw': yaw}})
        else:
            out.append({'kind': 'descend', 'from': a, 'to': b, 'yaw': yaw,
                        'act': {'dv': dv, 'yaw': yaw}})
    return out


def execute(env, gate, path, planner=None, max_ticks_per_move=40, refuse_stops=True):
    """Spend a plan one intent at a time, with every tick gated first.

    Returns a record with per-intent verdicts and what the plant actually did, so
    that the two can be compared afterwards rather than assumed to agree.

    A refusal does not retry: a gate that is asked again after saying no is a gate
    whose answer did not matter. `refuse_stops` makes that explicit -- with it off,
    the executor is the ungated baseline the bench uses to show what the gate is
    worth.
    """
    planner = planner or CellPlanner(env.world)
    rec = {'moves': [], 'refused': 0, 'ticks': 0, 'ok': True, 'why': ''}
    for mv in intents_for(path):
        act = dict(mv['act'])
        if gate is not None and refuse_stops:
            v = gate.check(env.st, act)
            if not v.ok:
                rec['moves'].append({'kind': mv['kind'], 'to': mv['to'],
                                     'verdict': 'REFUSED', 'reasons': v.reasons,
                                     'detail': v.detail})
                rec['refused'] += 1
                rec['ok'] = False
                rec['why'] = 'refused at %s: %s' % (mv['kind'], ','.join(v.reasons))
                break
        target = mv['to']
        done = False
        for _ in range(max_ticks_per_move):
            rec['ticks'] += 1
            o = env.tick(act)
            cx, cz = math.floor(o['pos'][0]), math.floor(o['pos'][2])
            if (cx, o['pos'][1] if False else int(math.floor(o['pos'][1] + 1e-6)), cz) == target:
                done = True
                break
            if env.dead:
                rec['ok'] = False
                rec['why'] = 'the body died'
                break
        rec['moves'].append({'kind': mv['kind'], 'to': target,
                             'verdict': 'done' if done else 'stalled',
                             'pos': [round(x, 3) for x in env.obs()['pos']]})
        if env.dead:
            break
        if not done:
            rec['ok'] = False
            rec['why'] = 'stalled at %s -> %s' % (mv['kind'], target)
            break
    return rec
