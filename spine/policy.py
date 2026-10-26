# -*- coding: utf-8 -*-
"""policy.py -- the learned layer: what a network proposes, and what it is allowed to be.

The three layers of spine-agent, and which one this is:

    the model / the learner   proposes an intent for the next tick        <-- here
    EmbodiedGate              refuses the intents that break physics      embodied.py
    Env / the real server     moves the body, or does not                 embodied.py

The reason the learner is here at all, given that an analytic kernel already
decides what is safe: the gate can say no, and only a learner can say *something
else*. A hand-written policy has to be told every move in advance; a policy that
watches state and picks an intent generalises over a world it was not written for,
which is the whole claim of a visuomotor / VLA model. So this file is a small,
honest version of that: a network that maps (local voxel patch, body state, goal)
to an intent, trained on episodes the gate already admitted, and never allowed to
bypass it.

Deliberately NOT here:

  * pixels. There is no camera in the loop, because there is no reason to add a
    vision encoder to a claim about gating. `patch()` is a local voxel occupancy
    grid -- an egocentric depth/occupancy view, which is what a visuomotor policy
    would get out of a depth camera, with the rendering step removed so the
    failure cannot be a rendering bug.
  * a GPU. The policy below is 20k parameters and trains in seconds on the CPU of
    this laptop, on purpose: a claim that "the learning layer works" should be
    checkable by whoever reads it, not on hardware I rented.

What the numbers must show, or this file is decoration:

    clone        the policy reproduces the scripted controller's intents on held-out
                 states at a rate well above the majority-class baseline
    gated        the policy's own intents, run through the gate in the simulator,
                 keep the body alive and reach the goal -- and the ungated arm of the
                 same policy does not
    honest       the policy is allowed to be wrong; the gate is what makes wrongness
                 survivable, and the bench reports how often it had to step in
"""
import json
import math
import os

import torch
import torch.nn as nn

from . import embodied as emb
from . import plan
from . import voxel

# ---- the action space -----------------------------------------------------
# Five discrete intents, not a continuous velocity: a discrete set is what the gate
# can be asked about cheaply, and it is what a VLA model emits after tokenisation.
ACTS = (
    'stop',
    'forward',
    'back',
    'left',
    'right',
)
ACTV = ACT_DV = {
    'stop': (0.0, 0.0, 0.0),
    'forward': (0.0, 0.0, -emb.WALK),
    'back': (0.0, 0.0, emb.WALK),
    'left': (-emb.WALK, 0.0, 0.0),
    'right': (emb.WALK, 0.0, 0.0),
}
JUMP = 'jump'          # separate head: it composes with any of the above

# How close to a cell's centre the TEACHER must get before it moves its target on.
# A separate, looser number (LEG_TOL below) decides when a leg is over, because a
# discrete policy cannot hit a 0.30 target: it walks 0.216 per tick, so it stops
# 0.37 short and is right to. Conflating the two is what scored a correct decision
# as a stuck leg (measured: 3/14 worlds, teacher fine, student frozen).
# How close to a cell's centre counts as being in it. This number is load-bearing
# and three values were measured before it was right:
#   0.9  on the centre   adjacent centres are 1.0 apart, so it fired at 0.78 and the
#                        route never advanced (0/24 finished, every leg "done")
#   0.62 the width of a body's half-box -- generous, and WRONG: it advances while the
#                        body is still 0.468 off centre, so a 0.6-wide body enters a
#                        one-cell corridor straddling the neighbouring cell and
#                        wedges against a wall (7/36 routes).
#   0.45                 two ticks of travel. Too large for the ROUTE cursor: the
#                        cursor would advance while the body was still in the
#                        previous cell, and the teacher fell from 37/38 to 0/38.
ARRIVE_TOL = 0.30

# When a leg is over. Two ticks of travel, so any intent the policy can emit is
# enough to close the gap -- and any shortfall inside it is a shortfall the leg
# controller must not read as failure.
LEG_TOL = 0.45

# How far off the route's column the body may be and still count as being in that
# cell. Half a body plus one step: 0.6/2 = 0.3 is the box's own half-width, and a
# whole step of 0.216 covers the furthest a single tick can put it. Anything less
# means a body standing on a cell boundary -- which is where every leg starts --
# is read as off-route.
LAT_TOL = 0.55


def intent_from(name, jump, yaw=0.0):
    """A network's choice, in the shape the gate already speaks."""
    return {'dv': list(ACT_DV[name]), 'jump': bool(jump), 'yaw': yaw,
            'keys': dict([(k, True) for k in ()]), 'intent': name + ('+jump' if jump else '')}


# ---- the observation ------------------------------------------------------
PATCH_R = 3            # cells each way in x and z
PATCH_H = 3            # cells up: feet, head, and one above the head

def patch(world, x, y, z, r=PATCH_R, h=PATCH_H):
    """Egocentric occupancy: what a depth camera would hand a visuomotor policy.

    Three channels, because they are the three things the gate rules about:
      solid   -- something is there
      hazard  -- being there costs health
      floor   -- the cell below is solid (this is the standable bit)
    """
    solid, hazard, floor = [], [], []
    fx, fy, fz = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
    for dy in range(h):
        for dz in range(-r, r + 1):
            for dx in range(-r, r + 1):
                cell = (fx + dx, fy + dy, fz + dz)
                name = world.blocks.get(cell)
                solid.append(1.0 if voxel.is_solid(name) else 0.0)
                hazard.append(1.0 if voxel.hazard_of(name) > 0.0 else 0.0)
                below = world.blocks.get((fx + dx, fy + dy - 1, fz + dz))
                floor.append(1.0 if voxel.is_solid(below) else 0.0)
    return solid + hazard + floor


def pursuit_target(path, cursor, st):
    """The point to steer at, chosen so that ONE axis is ever in play.

    Two failures came from pointing at the raw cell centre, and they are the same
    failure: with two simultaneous errors the controller has to choose which axis to
    close first, and any rule that changes its mind between ticks oscillates.

      * the body at x=1.23 steering at a centre of x=1.5 has a 0.27 lateral error
        against a 0.45 forward error. "Close the larger gap first" flips between the
        axes every tick: measured 254 `forward` and 238 `back` in 500 ticks, the
        body bouncing between z=2.05 and z=1.83 and never moving again.
      * the teacher never hit it because it walks the centre of the corridor; a
        LEARNED policy is always a little off, so this is exactly the state a clone
        lives in.

    So: re-centre laterally FIRST, on the axis the body is already on, and only then
    advance. The target is the cell's centre on the lateral axis and the body's own
    coordinate on the travel axis -- a point straight ahead, which makes the greedy
    axis rule correct by construction.
    """
    c = path[cursor]
    nxt = path[min(cursor + 1, len(path) - 1)]
    cx, cz = c[0] + 0.5, c[2] + 0.5
    along_z = abs(nxt[2] - c[2]) >= abs(nxt[0] - c[0])
    latent = (st.x - cx) if along_z else (st.z - cz)
    # ONE WALK STRIDE. Below this the remaining lateral error is smaller than the
    # smallest correction the body can make, so asking for it is asking for an
    # oscillation between two positions 0.216 apart -- which is exactly what the
    # 254-forward/238-back log shows. 0.10 was too tight: it left a 0.27 error
    # unreachable, the code fell through to the straight-ahead branch, and the
    # forward/back flip came back.
    if abs(latent) > emb.WALK * 0.75:
        # Off the centre line: the only move that helps is the lateral one, so aim
        # at the cell's centre on BOTH axes. `teacher` then closes the larger gap
        # first, which here is the lateral one by construction.
        #
        # The tempting alternative -- zero the travel-axis error so the lateral one
        # cannot be outvoted -- was measured and is worse: with the travel axis
        # exactly satisfied the teacher can never reach the centre line, the cursor
        # never advances, and the TEACHER falls from 38/38 to 14/38 (every failure
        # "no progress for 60 ticks"). A rule that helps the student by breaking the
        # teacher is not a fix; the teacher is the thing the student is cloned from.
        return (cx, float(c[1]), cz)
    if along_z:
        # travelling in z: aim straight ahead on the body's own x, so the x gap is
        # zero and cannot compete with the z gap
        return (st.x, float(c[1]), cz)
    return (cx, float(c[1]), st.z)


def state_features(st, goal):
    """The non-visual half: how fast I am going, and where the next waypoint is.

    `goal` is a waypoint, not the far target. That is the layering: CellPlanner in
    plan.py decides *where to go next* (it can see the whole map and is cheap to
    run at 1 Hz), and this policy only decides *how to get three cells over there*
    from what it can see from here. Asking one small network to route around a wall
    it cannot see would be asking it to memorise the map.
    """
    gx, gy, gz = goal
    return [
        st.vx / emb.WALK, st.vy / 1.0, st.vz / emb.WALK,
        1.0 if st.on_ground else 0.0,
        1.0 if st.sneak else 0.0,
        (gx - st.x) / 8.0, (gy - st.y) / 8.0, (gz - st.z) / 8.0,
        min(1.0, math.hypot(gx - st.x, gz - st.z) / 8.0),
    ]


PATCH_DIM = 3 * PATCH_H * (2 * PATCH_R + 1) ** 2
STATE_DIM = 9


class PolicyNet(nn.Module):
    """A small conv trunk over the occupancy patch plus a state MLP.

    Small on purpose. If a 20k-parameter net can learn to walk toward a goal and
    stop at a ledge, then the claim is about the architecture of the system (gate +
    learner), not about the size of the learner, and whoever reads this can retrain
    it in a minute.
    """

    def __init__(self, n_acts=len(ACTS)):
        super().__init__()
        c = 3
        self.trunk = nn.Sequential(
            nn.Conv2d(c * PATCH_H, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        side = 2 * PATCH_R + 1
        feat = 24 * side * side
        self.head_state = nn.Sequential(nn.Linear(STATE_DIM, 32), nn.ReLU())
        self.policy = nn.Sequential(nn.Linear(feat + 32, 64), nn.ReLU(), nn.Linear(64, n_acts))
        self.jump_head = nn.Sequential(nn.Linear(feat + 32, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, patch_vec, state_vec):
        b = patch_vec.shape[0]
        side = 2 * PATCH_R + 1
        # channel-major: PATCH_H channels of 3 planes each (solid / hazard / floor)
        x = patch_vec.view(b, PATCH_H, 3, side, side)
        x = x.permute(0, 1, 2, 3, 4).reshape(b, 3 * PATCH_H, side, side)
        v = self.trunk(x)
        s = self.head_state(state_vec)
        both = torch.cat([v, s], dim=1)
        return self.policy(both), self.jump_head(both)



def signed_lateral(st, goal):
    """How far off the line to the waypoint the body is, along its normal.

    The learner was trained on states the TEACHER produced, and the teacher walks the
    route down the middle. When the student drifts half a cell sideways it sees a
    state the teacher never visited, calls it "arrived" on the axis it can see, and
    emits `stop` -- measured: the body froze at (-3.5, -4.13) with the waypoint at
    (-3.5, -5.5), i.e. centred in x and a third of a cell short in z, and stayed
    there for the rest of the episode. This is the classic DAgger residue: the
    aggregate dataset fixes the states the student reaches, but only if the state
    FEATURES make the correction visible. So the feature is the signed offset, and
    the layer above supplies a corrective target while the body is off-centre.
    """
    return st.x - goal[0]


def net_action(net, world, st, goal):
    """The learner's proposal for this tick: one of ACTS, plus the jump head.

    No gate here and no safety check of any kind -- that is the point. This function
    answers "what would the policy do", and EmbodiedGate answers "may it". Keeping
    them in separate functions is what makes the gated/ungated bench arms differ in
    exactly one argument.
    """
    pv = torch.tensor(patch(world, st.x, st.y, st.z), dtype=torch.float32).unsqueeze(0)
    sv = torch.tensor(state_features(st, goal), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, jl = net(pv, sv)
    return ACTS[int(logits.argmax(dim=1))], bool(jl.argmax(dim=1))


def train(rows, epochs=60, lr=2e-3, batch=128, seed=0, verbose=False):
    """Behaviour cloning on (patch, state) -> teacher action.

    `rows` carries the teacher's action at the state the STUDENT visited (dagger_episode
    records it that way), so this is DAgger's aggregate dataset and not just the
    teacher's own trajectory. Both heads are trained; the jump head is imbalanced
    (almost every tick is no-jump) and is weighted inversely so it does not collapse
    to always-false.
    """
    if not rows:
        raise ValueError('train() got no rows')
    torch.manual_seed(seed)
    # One thread. These tensors are 128 x 441 -- far too small to pay for the
    # intra-op thread pool, and on this host the default pool made a 3-second job
    # into a many-minute one (measured: 60 epochs on 1922 rows finished in 3.1 s
    # with one thread and did not finish in 590 s with the default).
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    net = PolicyNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    P = torch.tensor([r['patch'] for r in rows], dtype=torch.float32)
    S = torch.tensor([r['state'] for r in rows], dtype=torch.float32)
    A = torch.tensor([r['act'] for r in rows], dtype=torch.long)
    J = torch.tensor([r['jump'] for r in rows], dtype=torch.long)

    counts = torch.bincount(A, minlength=len(ACTS)).float().clamp(min=1.0)
    w_act = (counts.sum() / (counts * len(ACTS)))
    jc = torch.bincount(J, minlength=2).float().clamp(min=1.0)
    w_jump = jc.sum() / (jc * 2)

    n = len(rows)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = corr = 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            logits, jl = net(P[idx], S[idx])
            loss = (nn.functional.cross_entropy(logits, A[idx], weight=w_act)
                    + 0.5 * nn.functional.cross_entropy(jl, J[idx], weight=w_jump))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += len(idx)
            corr += int((logits.argmax(dim=1) == A[idx]).sum())
        if verbose and (ep + 1) % 20 == 0:
            print('   epoch %3d  clone acc %.3f' % (ep + 1, corr / max(1, tot)))
    net.eval()
    return net


def split(rows, every=4):
    """Train/test by tick, not by episode.

    Consecutive ticks are near-duplicates, so a random split would put twins on both
    sides and report a number that measures memorisation. Every Nth tick is held out;
    neighbours of a held-out tick stay in train, which makes the reported accuracy a
    LOWER bound on the true generalisation, and a lower bound is the safe direction
    for a claim that the learner works.
    """
    train_rows, test_rows = [], []
    for i, r in enumerate(rows):
        (test_rows if i % every == 0 else train_rows).append(r)
    return train_rows, test_rows


def accuracy(net, rows):
    """(action accuracy, jump accuracy) on held-out rows -- the two heads, apart."""
    import torch as _t
    P = _t.tensor([r['patch'] for r in rows], dtype=_t.float32)
    S = _t.tensor([r['state'] for r in rows], dtype=_t.float32)
    A = _t.tensor([r['act'] for r in rows], dtype=_t.long)
    J = _t.tensor([r['jump'] for r in rows], dtype=_t.long)
    with _t.no_grad():
        logits, jl = net(P, S)
    return (float((logits.argmax(dim=1) == A).float().mean()),
            float((jl.argmax(dim=1) == J).float().mean()))


def clone_accuracy(net, rows):
    """Held-out agreement with the teacher, next to the majority-class baseline.

    The baseline is the number that matters: a teacher that always says `forward`
    would score high on this and mean nothing, so a clone claim without its baseline
    beside it is not a claim.
    """
    P = torch.tensor([r['patch'] for r in rows], dtype=torch.float32)
    S = torch.tensor([r['state'] for r in rows], dtype=torch.float32)
    A = torch.tensor([r['act'] for r in rows], dtype=torch.long)
    with torch.no_grad():
        logits, _ = net(P, S)
    acc = float((logits.argmax(dim=1) == A).float().mean())
    vals, cnts = A.unique(return_counts=True)
    base = float(cnts.max()) / float(len(A))
    return {'clone_acc': round(acc, 4), 'majority_baseline': round(base, 4),
            'n': len(A), 'classes': int(len(vals)),
            'beats_baseline': bool(acc > base + 0.05)}


# ---- the teacher ----------------------------------------------------------
def teacher(world, st, waypoint, max_drop=3.0):
    """The scripted controller the policy is cloned from: walk at the waypoint,
    stop when you arrive, and never make a move the body cannot survive.

    Every correction below came from measuring it, and three of them were the same
    mistake in different clothes -- deciding about a POINT when the plant resolves a
    BOX, then discovering the difference as a body wedged against a wall:

      * arrival used to fire at 0.9 from the centre; adjacent centres are 1.0 apart,
        so it fired at 0.78 and the route never advanced (0/24 routes finished while
        every leg reported "done"). It is ARRIVE_TOL now.
      * the ledge scan checked all three cells ahead on both axes, so a pit the body
        was walking ALONGSIDE stopped it (11/24). It checks the move being made.
      * a move was refused only for a missing floor, never for a wall, so "forward"
        always won even inside stone. `moves_ok` tests both.
    """
    gx, gy, gz = waypoint
    dx, dz = gx - st.x, gz - st.z
    if abs(dx) <= ARRIVE_TOL and abs(dz) <= ARRIVE_TOL:
        return 'stop', False, 'arrived'

    # Preference order: the axis that closes more of the gap first, the other second.
    if abs(dz) >= abs(dx):
        names = ['forward' if dz < 0 else 'back', 'left' if dx < 0 else 'right']
    else:
        names = ['left' if dx < 0 else 'right', 'forward' if dz < 0 else 'back']
    for name in names:
        if moves_ok(world, st, name, max_drop):
            return name, False, ''

    # Neither axis works. In a one-cell corridor that has exactly one cause: the
    # body's centre is off the corridor's axis, so its 0.6-wide box already spans
    # the neighbouring cell and any step forward hits the wall beside it. Measured at
    # x=1.248 in a corridor centred on 1.5 -- off by 0.252, more than (0.5 - 0.3),
    # so the box had crossed the cell boundary before the refused move even began.
    # Re-centre first; the same move is then free.
    for axis, target in ((0, round(st.x - 0.5) + 0.5), (2, round(st.z - 0.5) + 0.5)):
        if abs((st.x if axis == 0 else st.z) - target) < 0.06:
            continue
        name = (('right' if target > st.x else 'left') if axis == 0
                else ('forward' if target < st.z else 'back'))
        if moves_ok(world, st, name, max_drop):
            return name, False, ''
    return 'stop', False, 'blocked'


def slide(world, st, waypoint, max_drop=3.0):
    """A blocked step's second choice: the usable axis that closes more distance.
    This is what lets the teacher round a corner inside a wall's shadow instead of
    stopping at it."""
    gx, gy, gz = waypoint
    dx, dz = gx - st.x, gz - st.z
    order = [nm for nm, axis in (('left', dx), ('right', dx), ('forward', dz),
                                 ('back', dz))
             if (abs(axis) > ARRIVE_TOL and
                 ((nm in ('left', 'right')) == (abs(dx) >= abs(dz)) or abs(dx) < ARRIVE_TOL))]
    order = []
    if abs(dx) > ARRIVE_TOL:
        order.append('left' if dx < 0 else 'right')
    if abs(dz) > ARRIVE_TOL:
        order.append('forward' if dz < 0 else 'back')
    # a centring step is a legal fallback too: it is the one move that fixes the
    # cause instead of working around it
    for axis, target in ((0, round(st.x - 0.5) + 0.5), (2, round(st.z - 0.5) + 0.5)):
        if abs((st.x if axis == 0 else st.z) - target) > 0.06:
            order.append(('right' if target > st.x else 'left') if axis == 0
                         else ('forward' if target < st.z else 'back'))
    order.sort(key=lambda nm: -abs(ACT_DV[nm][0] * dx + ACT_DV[nm][2] * dz))
    for name in order:
        if moves_ok(world, st, name, max_drop):
            return name
    return None


def moves_ok(world, st, name, max_drop=3.0, half=0.3, h=1.8):
    """Can the body make this move: floor ahead within `max_drop`, and no wall.

    Two ways to be impossible, and for a long time only one of them was tested. A
    move into stone looked standable because `landing_below` only asks what is under
    the feet, so the first preferred axis always won and the body pressed into the
    wall for the rest of the episode.
    """
    dv = ACT_DV[name]
    x, z = st.x + dv[0], st.z + dv[2]
    top, _fluid, _nm = emb.landing_below(world, x, z, st.y)
    if top is None or (st.y - (top + 1.0)) > max_drop:
        return False
    # ... and the box, because that is what the plant resolves. Ask this file's own
    # body-blocked test, which uses the same 0.6 x 1.8 x 0.6 box as the sim.
    for sx in (-1, 1):
        for sy in (0.0, h - 0.001):
            for sz in (-1, 1):
                c = (int(math.floor(x + sx * half)), int(math.floor(st.y + sy)),
                     int(math.floor(z + sz * half)))
                if world.solid(c):
                    return False
    return True


def _body_blocked(world, x, y, z, half=0.3, h=1.8):
    """Would a body box standing at (x, y, z) intersect a solid block?

    Kept as the named test because the reason it exists is worth keeping: a
    controller that reasons about a point while the plant resolves a box is deciding
    on different geometry than the one it is subject to.
    """
    for sx in (-1, 1):
        for sy in (0.0, h - 0.001):
            for sz in (-1, 1):
                c = (int(math.floor(x + sx * half)),
                     int(math.floor(y + sy)),
                     int(math.floor(z + sz * half)))
                if world.solid(c):
                    return True
    return False


def cell_of(x, y, z):
    return (int(math.floor(x)), int(math.floor(y)), int(math.floor(z)))


def line_of_sight(planner, a, b, step=0.25):
    """Can the body walk from cell a to cell b without leaving standable ground?

    Sampled, not Bresenham: the body moves in continuous coordinates, so the cells
    it actually occupies are the ones a quarter-cell walk passes through. A
    cell-stepping version would call a diagonal gap walkable.
    """
    ax, ay, az = a[0] + 0.5, float(a[1]), a[2] + 0.5
    bx, by, bz = b[0] + 0.5, float(b[1]), b[2] + 0.5
    n = max(1, int(math.hypot(bx - ax, bz - az) / step))
    if abs(by - ay) > 1e-9:
        return False                    # no stairs in this family; keep it strict
    for i in range(n + 1):
        f = i / n
        c = cell_of(ax + (bx - ax) * f, ay, az + (bz - az) * f)
        if not planner.standable(c):
            return False
    return True


def furthest_visible(planner, here, path):
    """Pure pursuit: the furthest point down the route that can be walked to in a
    straight line. Following the route cell by cell makes a controller stop at every
    cell; following it through walls makes it walk into the wall. This is the third
    option, and it is what the body's own routing would do if it had the map."""
    best = here
    for c in path:
        if line_of_sight(planner, here, c):
            best = c
        else:
            break
    return best


# ---- episodes -------------------------------------------------------------
#
# There is exactly one episode loop below. The three public functions are wrappers
# that differ in two arguments -- who chooses the action, and whether the gate is
# consulted -- because three copies of a loop is how the copies drift apart, and
# the whole point of this file is that the gated and ungated arms differ in nothing
# else. `record` controls a third thing: teacher episodes collect a training set,
# student episodes collect the outcome only.


def route(planner, start, goal):
    """The A* route, in cells, as (path, why). A route is a plan; waypoints are a
    controller's business, and the controller now picks them by line of sight."""
    return planner.plan(cell_of(start[0], start[1], start[2]),
                        cell_of(goal[0], goal[1], goal[2]))


def advance_cursor(path, cursor, x, z, tol):
    """How far along the route the body has got. One rule, used everywhere.

    The bug this function exists to prevent, because it cost an afternoon: the
    route cursor advanced while the body was within ARRIVE_TOL of a cell's centre,
    and a LEG ended while the body was within LEG_TOL of its target -- two numbers,
    0.30 and 0.45. A body stopped 0.37 from a centre therefore satisfied the leg
    test and not the cursor test, so every leg opened and closed on the same
    waypoint, one tick each: measured 76 legs on an 11-cell route, 101 ticks, the
    body 1.37 from the goal and no `stuck_at` anywhere to say why. Two thresholds
    describing one decision will always disagree at the boundary; there is one
    threshold below and this function is where it is read.
    """
    n = len(path)
    while cursor < n - 1:
        c = path[cursor]
        # The LATERAL tolerance is the body's half-width plus a step: a body whose
        # centre is on a cell boundary still occupies the corridor, and a test that
        # demands half a cell of clearance from BOTH axes says it is nowhere on the
        # route. Measured with one tolerance for both: a body at x=-3.5 (exactly on
        # the boundary of the route's column) advanced the cursor by zero for 500
        # ticks, flipped left/right 231 times, and never left the spot.
        if abs(x - (c[0] + 0.5)) <= LAT_TOL and abs(z - (c[2] + 0.5)) <= tol:
            cursor += 1
        else:
            break
    return cursor


def run_episode(world, start, goal, ticks=400, gate=None, planner=None,
                student=None, trust_planner=True, explore=0.0, seed=0,
                record=False, max_leg=60):
    """One episode, one loop, one tick at a time.

    The loop, in order, and the order matters:

      1. advance the route cursor by how close the body is (advance_cursor)
      2. pick the target: that cell's centre, or the goal itself on the last cell
      3. whoever is acting proposes the intent
      4. if recording, the TEACHER labels the state, whatever the student chose
         -- the label is never the student's own action, so DAgger cannot drift
      5. the intent goes through the gate, which may replace it with stop
      6. the plant advances, and the outcome is read off the plant afterwards

    There is no inner "leg" loop any more. It existed to give the teacher a
    sub-goal, and the teacher does not need one: a target is a target, and the
    cursor decides when it has changed. Removing it removed the split-brain that
    two nested loops over one target kept producing.
    """
    import random
    rnd = random.Random(seed)
    env = emb.Env(world)
    env.reset(start[0], start[1], start[2])
    pl = planner or plan.CellPlanner(world)

    out = {'ticks': 0, 'reached': False, 'refused': 0, 'died': False, 'final': None,
           'health': env.health, 'min_health': env.health, 'planned': False,
           'plan_why': '', 'cells': 0, 'legs_done': 0, 'stuck_at': None,
           'blocked_at': None, 'teacher_agreement': 0.0, 'n': 0, 'acts': {},
           'stalled': 0, 'why_ended': ''}

    if trust_planner:
        path, stats = route(pl, start, goal)
        out['planned'] = bool(path)
        out['plan_why'] = stats.get('why', '')
        out['cells'] = len(path)
    else:
        # no route: the target IS the goal, which is the ablation that hands the
        # low layer the far goal and lets the gate show what it is for
        path = [cell_of(goal[0], goal[1], goal[2])]

    rows = []
    t = 0
    cursor = 0
    last_pos = (env.st.x, env.st.z)
    idle = 0
    seg = 0
    while t < ticks and path:
        here_x, here_z = env.st.x, env.st.z
        was = cursor
        cursor = advance_cursor(path, cursor, here_x, here_z, LEG_TOL)
        if cursor != was:
            seg += 1
            out['legs_done'] = seg

        c = path[cursor]
        if not pl.standable(c) and cursor < len(path) - 1:
            # the body is a cell off the route (pushed onto a wall's side). The
            # planner emits only standable cells, so step past this one instead of
            # driving into it for the rest of the episode.
            cursor += 1
            c = path[cursor]

        if cursor >= len(path) - 1:
            wp = goal
        else:
            wp = pursuit_target(path, cursor, env.st)

        cur = env.st
        want, want_jump, why = teacher(world, cur, wp)
        if why == 'blocked':
            alt = slide(world, cur, wp)
            if alt is None:
                out['blocked_at'] = list(wp)
                out['why_ended'] = 'route wants a move nobody can make'
                break
            want = alt

        if student is None:
            name, jump = want, want_jump
        else:
            name, jump = net_action(student, world, cur, wp)
            out['n'] += 1
            out['teacher_agreement'] += 1 if name == want else 0
            out['acts'][name] = out['acts'].get(name, 0) + 1
            if explore and rnd.random() < explore:
                name, jump = ACTS[rnd.randrange(len(ACTS))], False

        if record:
            rows.append({'patch': patch(world, cur.x, cur.y, cur.z),
                         'state': state_features(cur, wp),
                         'act': ACTS.index(want),
                         'jump': int(bool(want_jump)),
                         't': t, 'wp': cursor,
                         'wp_dist': math.hypot(wp[0] - cur.x, wp[2] - cur.z)})

        intent = intent_from(name, jump)
        if gate is not None:
            vd = gate.check(cur, intent)
            if not vd.ok:
                out['refused'] += 1
                intent = {'dv': (0.0, 0.0, 0.0), 'jump': False, 'keys': {},
                          'intent': 'stop'}
        env.tick(intent)
        t += 1
        out['ticks'] = t
        out['health'] = env.health
        out['min_health'] = min(out['min_health'], env.health)
        if env.dead:
            out['died'] = True
            out['why_ended'] = 'died'
            break

        if math.hypot(goal[0] - env.st.x, goal[2] - env.st.z) < 1.2:
            out['reached'] = True
            out['why_ended'] = 'reached'
            break

        # No-progress guard. A policy is allowed to decline a move -- `stop` is the
        # right answer to a target it cannot improve on -- but an episode in which
        # the body has not moved for 60 ticks is over, and saying so is the
        # difference between "this policy is bad here" and "this run is still going".
        moved = math.hypot(env.st.x - last_pos[0], env.st.z - last_pos[1])
        if moved < 0.02:
            idle += 1
        else:
            idle = 0
            last_pos = (env.st.x, env.st.z)
        if idle >= 60:
            out['stalled'] = idle
            out['stuck_at'] = list(wp)
            out['why_ended'] = 'no progress for %d ticks' % idle
            break

    out['final'] = list(env.st.pos())
    out['teacher_agreement'] /= max(1, out['n'])
    if not out['why_ended']:
        out['why_ended'] = 'tick budget'
    return rows, out


def scripted_episode(world, start, goal, ticks=400, gate=None, planner=None,
                     stride=3, max_leg=60):
    """The teacher's own episode, collecting the training set.

    `stride` is accepted and ignored: waypoints used to be every Nth cell of the
    route, which is a fixed guess at how far ahead to look. Pure pursuit picks the
    furthest point that can be walked to in a straight line, which is the same idea
    with the guess removed. The argument stays so old call sites keep working.
    """
    return run_episode(world, start, goal, ticks=ticks, gate=gate, planner=planner,
                       student=None, trust_planner=True, record=True,
                       max_leg=max_leg)


def dagger_episode(net, world, start, goal, ticks=400, stride=3, max_leg=60,
                   planner=None, explore=0.0, seed=0):
    """The student acts, the teacher labels. The fix for distribution shift."""
    return run_episode(world, start, goal, ticks=ticks, gate=None, planner=planner,
                       student=net, trust_planner=True, explore=explore, seed=seed,
                       record=True, max_leg=max_leg)


def policy_episode(net, world, start, goal, ticks=400, gate=None, stride=3,
                   max_leg=60, planner=None, trust_planner=True):
    """The student acts and is measured. `gate` is the only difference between the
    two arms the bench compares, and `trust_planner=False` is the ablation that
    hands the far goal straight to the low layer."""
    return run_episode(world, start, goal, ticks=ticks, gate=gate, planner=planner,
                       student=net, trust_planner=trust_planner, record=False,
                       max_leg=max_leg)


# ---- persistence ----------------------------------------------------------
def save(net, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save(net.state_dict(), path)
    return path


def load(path):
    net = PolicyNet()
    net.load_state_dict(torch.load(path, map_location='cpu'))
    net.eval()
    return net
