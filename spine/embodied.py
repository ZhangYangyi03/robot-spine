# -*- coding: utf-8 -*-
"""embodied.py -- the plant, and the gate that stands in front of it.

This is the merge seam of spine-agent, in the same shape as gate.py in front of
control.py:

    a controller (any controller -- a scripted walk, or a language model's plan
    turned into per-tick intents) proposes one tick of motion,
    EmbodiedGate judges it against the voxel world by arithmetic,
    Env integrates what survives.

Neither half knows the other. The gate never runs the physics and the plant never
knows a gate exists, so "the gate admits it" and "the body survives it" are two
numbers produced by two different pieces of code, and the bench compares them
instead of assuming one implies the other.

What the body can be hurt by, and therefore what the gate has five rules for:

    void_fall    the column under the target has no floor at all
    drop_height  the fall the plan starts lands further below than a body survives
    hazard       lava, fire, cactus, magma where the body would be
    speed        one tick of intent that moves faster than the body can move
    pose_reach   a crouch the leg cannot reach, from the closed-form IK

`wall` and `step_up` are deliberately NOT rules. Walking into a wall is not
dangerous, it is just slow; the plant stalls and the controller tries again. A
gate that refuses everything unusual scores 100% on "faults caught" and is
useless, which is why the bench reports false refusals on legal plans next to the
catch rate -- exactly as bench_gate.py does for the joint gate.

Constants are Minecraft's, not fitted to this bench: 20 ticks/s, gravity 0.08 per
tick damped by 0.98, jump velocity 0.42, walk 0.2158 blocks/tick (4.317 m/s).
They are quoted because a simulator invented to make the gate look right would
prove nothing; these are the numbers the game itself uses, and the resulting
flight arc is checked against the closed form for a projectile in the bench.
"""
import math

from . import body as bodymod
from . import voxel
from .gate import Verdict

TICK_S = 0.05
GRAVITY = 0.08
DRAG = 0.98
JUMP_V = 0.42
WALK = 0.2158            # blocks per tick, i.e. 4.317 m/s
SNEAK_SCALE = 0.3
MAX_STEP_UP = 0.6        # the game's automatic step
GROUND_FRICTION = 0.546  # horizontal velocity kept per tick with input, on ground
GROUND_DRAG = 0.6        # ... and with no input: the body coasts, it does not stop
AIR_DRAG = 0.91          # horizontal velocity kept per tick in the air
FALL_DAMAGE_FROM = 3.0   # damage = floor(fall - 3)
HAZARD_COOLDOWN = 10     # ticks between hazard hits, so lava is not 80 hp/s
VOID_Y = -64.0           # below this the world has no bottom, and the game hurts you
VOID_DAMAGE = 4.0


def landing_below(world, x, z, y_from):
    """First solid block below `y_from` in column (x, z), and the fluid met on the way.

    Arithmetic on the grid, no integration: (top + 1) is where a body standing on
    that block has its feet. Both the plant and the gate call this -- the plant to
    decide where the floor is, the gate to decide whether the floor is far enough
    away to hurt -- and they call the same function on purpose. What makes the
    gate's verdict worth something is that it is computed BEFORE the tick runs and
    compared afterwards in the bench, not that it re-derives the floor twice.
    """
    cx, cz = int(math.floor(x)), int(math.floor(z))
    fluid = None
    for cy in range(int(math.floor(y_from)), -320, -1):
        name = world.blocks.get((cx, cy, cz))
        if name in ('water', 'lava') and fluid is None:
            fluid = name
        if voxel.is_solid(name):
            return cy, fluid, name
    return None, fluid, None


class Env:
    """A deterministic voxel plant: gravity, collision, fall damage, hazards.

    Nothing in here consults a gate. It is the thing the gate is checked against.
    """

    def __init__(self, world, body=None, health=20.0, tick_s=TICK_S,
                 gravity=GRAVITY, drag=DRAG, walk=WALK, jump_v=JUMP_V):
        self.world, self.body = world, body or bodymod.Body()
        self.health, self.max_health = float(health), float(health)
        self.tick_s, self.g, self.drag = tick_s, gravity, drag
        self.walk, self.jump_v = walk, jump_v
        self.st = bodymod.State()
        self.fall_from = None          # feet height where the current fall began
        self.hits = []                 # (tick, kind, amount, cell)
        self.tick_n = 0
        self.hazard_cd = 0

    # -- setup -----------------------------------------------------------
    def reset(self, x, y, z, yaw=0.0):
        self.st = bodymod.State(x, y, z, yaw)
        self.body.stand(self.st)
        self.health, self.hits, self.tick_n, self.hazard_cd = self.max_health, [], 0, 0
        self.fall_from = None
        self.st.on_ground = self.supported(self.st.pos())
        return self.obs()

    # -- queries ---------------------------------------------------------
    def obs(self):
        return {'t': self.tick_n, 'pos': self.st.pos(), 'yaw': self.st.yaw,
                'vel': [self.st.vx, self.st.vy, self.st.vz],
                'on_ground': self.st.on_ground, 'health': self.health,
                'legs': [list(q) for q in self.st.legs]}

    def cells_of(self, p, w=0.3, h=1.8):
        return [(int(math.floor(p[0] + sx * w)), int(math.floor(p[1] + sy)),
                 int(math.floor(p[2] + sz * w)))
                for sx in (-1, 1) for sy in (0, h - 0.001) for sz in (-1, 1)]

    def solid_at(self, p):
        return any(self.world.solid(c) for c in self.cells_of(p))

    def supported(self, p):
        """Is there a solid block directly under the feet at this position?"""
        probe = [p[0], p[1] - 0.02, p[2]]
        return any(self.world.solid(c) for c in self.cells_of(probe))

    def in_water(self, p):
        return any(self.world.blocks.get(c) in ('water', 'lava')
                   for c in self.cells_of(p))

    # -- one tick --------------------------------------------------------
    def tick(self, act):
        """act: {'dv': [dx,dy,dz], 'jump': bool, 'sneak': bool, 'yaw': float|None}

        Horizontal motion is resolved one axis at a time, which is what the game
        does and what lets a body slide along a wall instead of sticking to it.
        Vertical motion is resolved AGAINST the column below (arithmetic on the
        grid), not by probing the body box, because the body box is 1.8 tall and
        probing it leaves the feet parked a fraction of a block above the floor --
        the first version of this file did exactly that and the jump arc never
        came back down. The order inside a tick is the game's: move by the current
        velocity, then apply gravity, which is why the arc peaks at 1.25 blocks.
        """
        st = self.st
        self.tick_n += 1
        if act.get('yaw') is not None:
            st.yaw = float(act['yaw'])
        st.sneak = bool(act.get('sneak'))
        dv = list(act.get('dv') or (0.0, 0.0, 0.0))
        limit = self.walk * (SNEAK_SCALE if st.sneak else 1.0)
        h = math.hypot(dv[0], dv[2])
        if h > limit:                       # the plant clamps, like a real drive
            dv[0] *= limit / h
            dv[2] *= limit / h

        # ---- horizontal, axis by axis ----
        for axis in (0, 2):
            d = dv[axis]
            if d == 0.0:
                continue
            n = max(1, int(math.ceil(abs(d) / 0.2)))
            for _ in range(n):
                trial = list(st.pos())
                trial[axis] += d / n
                if self.solid_at(trial):
                    # Step UP TO the contact, don't abandon the whole step.
                    #
                    # This is what the game does and what this plant did not: the
                    # body box is 0.6 wide, so in a one-cell corridor its centre has
                    # only 0.4 of usable range, and a step of 0.216 from z=-2.59 to
                    # -2.70 leaves the corridor even though the body was standing
                    # comfortably inside it. Refusing the step parked the body at a
                    # corridor mouth forever (measured: 7 of 36 routes died there,
                    # every one of them with a fixed position and a nonzero intent).
                    # A bisection onto the contact point costs five probes and makes
                    # the plant's "walls are sticky" behaviour match the engine's.
                    lo, hi = 0.0, d / n
                    for _ in range(5):
                        mid = 0.5 * (lo + hi)
                        probe = list(st.pos())
                        probe[axis] += mid
                        if self.solid_at(probe):
                            hi = mid
                        else:
                            lo = mid
                    if lo > 1e-9:
                        moved = list(st.pos())
                        moved[axis] += lo
                        st.x, st.z = moved[0], moved[2]
                    break
                st.x, st.z = trial[0], trial[2]

        # ---- vertical: move by the current velocity, then gravity ----
        if act.get('jump') and st.on_ground:
            st.vy = self.jump_v
            st.on_ground = False
            self.fall_from = st.y
        d = st.vy
        if d > 0.0:
            n = max(1, int(math.ceil(d / 0.2)))
            for _ in range(n):
                trial = [st.x, st.y + d / n, st.z]
                if self.solid_at(trial):
                    st.vy = 0.0
                    break
                st.y += d / n
        elif d < 0.0:
            top, _fluid, _name = landing_below(self.world, st.x, st.z, st.y)
            floor_y = None if top is None else top + 1.0
            if floor_y is not None and st.y + d <= floor_y:
                st.y = floor_y
                self._land()
            else:
                st.y += d
        if not st.on_ground:
            water = self.in_water(st.pos())
            st.vy = (st.vy - self.g * (0.25 if water else 1.0)) * (0.8 if water else self.drag)
            if self.fall_from is None:
                self.fall_from = st.y
        else:
            # walked off an edge? then the body is airborne from this height
            top, _fluid, _name = landing_below(self.world, st.x, st.z, st.y + 1e-9)
            if top is None or (top + 1.0) < st.y - 1e-9:
                st.on_ground = False
                self.fall_from = st.y
        # ---- keep the legs where the pelvis now is ----
        self.body.stand(st)
        # ---- hazards ----
        self._hazard_tick()
        self._void_tick()
        return self.obs()

    def _land(self):
        st = self.st
        if self.fall_from is None:
            self.fall_from = st.y
        fall = max(0.0, self.fall_from - st.y)
        st.vy = 0.0
        st.on_ground = True
        self.fall_from = None
        if fall > FALL_DAMAGE_FROM and not self.in_water(st.pos()):
            dmg = float(math.floor(fall - FALL_DAMAGE_FROM))
            self._hurt('fall', dmg, (int(st.x), int(st.y) - 1, int(st.z)), fall)

    def _hazard_tick(self):
        if self.hazard_cd > 0:
            self.hazard_cd -= 1
            return
        for c in self.cells_of(self.st.pos()):
            name = self.world.blocks.get(c)
            hz = voxel.hazard_of(name)
            if hz > 0.0:
                self._hurt('hazard:%s' % name, hz, c, 0.0)
                self.hazard_cd = HAZARD_COOLDOWN
                return

    def _void_tick(self):
        """Past the bottom of the world there is nothing to land on, so the fall
        never ends. The game charges for that; a plant that lets a body drop
        forever and reports full health would make void_fall look harmless."""
        if self.st.y < VOID_Y and self.tick_n % 10 == 0:
            self._hurt('void', VOID_DAMAGE,
                       (int(self.st.x), int(self.st.y), int(self.st.z)), 0.0)

    def _hurt(self, kind, amount, cell, fall):
        self.health -= amount
        self.hits.append({'tick': self.tick_n, 'kind': kind, 'amount': amount,
                          'cell': list(cell), 'fall': round(fall, 3),
                          'health_after': round(self.health, 3)})
        if not self.st.on_ground:
            self.fall_from = self.st.y

    @property
    def dead(self):
        return self.health <= 0.0


class EmbodiedGate:
    """Judge one tick of intent against the world. Arithmetic, not simulation.

    The prediction is deliberately NOT the plant: `_fall` scans a column and
    subtracts, while Env integrates gravity tick by tick. If the gate's prediction
    and the plant's outcome ever disagree, bench_embodied.py sees it in the
    refusals that were wrong, which is the only reason to have two of them.
    """

    RULES = ('void_fall', 'drop_height', 'hazard', 'speed', 'pose_reach')

    def __init__(self, world, body=None, rules=None, max_drop=FALL_DAMAGE_FROM,
                 walk=WALK, sneak_scale=SNEAK_SCALE, max_step_up=MAX_STEP_UP,
                 clearance=0.0):
        self.world = world
        self.body = body or bodymod.Body()
        self.rules = set(rules or self.RULES)
        self.max_drop = max_drop
        self.walk, self.sneak_scale = walk, sneak_scale
        self.max_step_up = max_step_up
        self.clearance = clearance
        self.last = None

    # -- one rule at a time, each naming itself ---------------------------
    def _speed(self, st, act):
        dv = list(act.get('dv') or (0.0, 0.0, 0.0))
        limit = self.walk * (self.sneak_scale if act.get('sneak') else 1.0)
        h = math.hypot(dv[0], dv[2])
        if h > limit + 1e-9:
            return {'speed': 'one tick asks for %.3f blocks, the body moves %.3f'
                             % (h, limit)}
        return {}

    def _landing(self, x, z, y_from):
        return landing_below(self.world, x, z, y_from)[:2]

    def _fall_check(self, st, act):
        """One column scan, two rules: is there a floor at all (void_fall), and is
        the floor far enough below to hurt (drop_height). They are separate rules
        because they can be switched off separately -- falling into the void and
        stepping off a 5-block ledge are different mistakes -- and a mutation
        control has to be able to turn one off without the other.
        """
        dv = list(act.get('dv') or (0.0, 0.0, 0.0))
        # A body in the game cannot stop inside one tick: it keeps a fraction of its
        # horizontal velocity (GROUND_DRAG with no input, GROUND_FRICTION with input,
        # AIR_DRAG in the air) and slides. A gate that predicts only the intended
        # velocity therefore refuses one tick too late -- measured on the live bench,
        # where the refusal came at z=-30.23 and the body coasted off the edge at
        # z=-30.34 anyway. So the column is checked at the coast position too: where
        # momentum alone will carry the body if the intent is obeyed and then dropped.
        coast = self.coast_offset(st, act)
        for fx, fz in ((dv[0], dv[2]), (coast[0], coast[1])):
            got = self._column_verdict(st, fx, fz)
            if got:
                return got
        return {}

    def coast_offset(self, st, act, ticks=None, v_min=1e-3):
        """How far momentum alone carries the body before it actually stops.

        The game integrates: move by velocity, then damp it. So the stopping
        distance is the whole geometric series, v * keep / (1 - keep), not one
        tick of it. Measured at walk speed on flat ground that is 0.32 blocks with
        no input and 0.26 with input -- and a gate that looks one tick ahead
        (0.22) is short by a tenth of a block, which is the difference between
        refusing at the edge and refusing already over it.

        `ticks` is capped only so a body in free fall cannot integrate forever.
        """
        keep = AIR_DRAG if not st.on_ground else (GROUND_DRAG if not act.get('dv')
                                                  else GROUND_FRICTION)
        cap = 400 if ticks is None else max(1, int(ticks))
        dx = dz = 0.0
        vx, vz = st.vx, st.vz
        for _ in range(cap):
            dx += vx
            dz += vz
            vx *= keep
            vz *= keep
            if abs(vx) < v_min and abs(vz) < v_min:
                break
        return dx, dz

    def _column_verdict(self, st, dx, dz):
        x, z = st.x + dx, st.z + dz
        top, fluid, name = landing_below(self.world, x, z, st.y)
        if top is None:
            if 'void_fall' in self.rules:
                return {'void_fall': 'no floor under (%.1f, %.1f) down to y=-320'
                                     % (x, z)}
            return {}
        if fluid in ('water', 'lava'):
            return {}                      # the fluid is the hazard rule's business
        drop = st.y - (top + 1.0)
        if drop > self.max_drop + 1e-9 and 'drop_height' in self.rules:
            return {'drop_height': 'a %.2f block fall onto %s, the body survives %.2f'
                                   % (drop, name, self.max_drop)}
        return {}

    def _hazard(self, st, act):
        dv = list(act.get('dv') or (0.0, 0.0, 0.0))
        x, z = st.x + dv[0], st.z + dv[2]
        y = st.y + dv[1]
        # standing in something that hurts, or landing in it
        for cy in (int(math.floor(y)), int(math.floor(y)) - 1):
            name = self.world.blocks.get((int(math.floor(x)), cy, int(math.floor(z))))
            if voxel.hazard_of(name) > 0.0:
                return {'hazard': '%s at (%d, %d, %d)' % (name, int(math.floor(x)), cy,
                                                          int(math.floor(z)))}
        return {}

    def _pose_reach(self, st, act):
        crouch = act.get('crouch')
        if crouch is None:
            return {}
        height = self.body.pelvis_h - float(crouch)
        if height <= 0.0:
            return {'pose_reach': 'crouch %.3f m leaves the pelvis below the feet' % crouch}
        q1, q2, ok, why = bodymod.leg_ik(0.0, height, self.body.leg_l, self.body.leg_l)
        if not ok:
            return {'pose_reach': why}
        return {}

    # -- the one entry point ---------------------------------------------
    def check(self, st, act):
        reasons, detail = {}, {}
        if 'void_fall' in self.rules or 'drop_height' in self.rules:
            got = self._fall_check(st, act)
            reasons.update(got)
            for k, v in got.items():
                detail[k] = v
        for rule in ('hazard', 'speed', 'pose_reach'):
            if rule not in self.rules:
                continue
            got = getattr(self, '_' + rule)(st, act)
            if got:
                reasons.update(got)
                detail[rule] = list(got.values())[0]
        ok = not reasons
        if ok:
            self.last = act
        return Verdict(ok, sorted(reasons), detail)
