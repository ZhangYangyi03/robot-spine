# -*- coding: utf-8 -*-
"""body.py -- the body that walks in a voxel world, and the legs that get it there.

The kinematic half of spine-agent. It is deliberately small and fully written out:

    pelvis at a fixed height above the feet, a torso capsule, a head sphere, and
    two 2-DOF legs modelled by a URDF (hip + knee, thigh = shin = 0.45 m)

The legs are the reason this file is not just a hitbox. Placing a foot at a point
is an inverse-kinematics problem, and robot-spine already has two independent
answers for it (a closed form written here, and the damped-least-squares solver in
ik.py). Keeping both means the closed form can be checked against a solver that did
not come from the same derivation, which is the repo's usual standard: a number
produced by the same person on the same day is not a witness.

So the embodied gate has something to refuse that a hitbox check cannot: a foot
target the leg cannot reach, a knee that would pass through a block, a hip outside
its limit. Those are "the spine cannot take this pose", which is a different
refusal from "the world is in the way" -- and the bench keeps the two apart.
"""
import math

from . import ik, scenes, voxel
from .urdf import Urdf

# -- proportions, all in metres, chosen so the numbers in the bench are human --
PELVIS_H = 0.88          # pelvis centre above the feet (legs are 0.90 m, so it cracks the knee)
NECK_H = 1.51
HEAD_H = 1.63
HEAD_R = 0.12
TORSO_R = 0.19
THIGH_R0, THIGH_R1 = 0.095, 0.075
SHIN_R0, SHIN_R1 = 0.080, 0.065
LEG_L = 0.45             # thigh and shin are the same length
HIP_SEP = 0.11           # half the distance between the two hips
HITBOX_W, HITBOX_H = 0.60, 1.80     # Minecraft's player box, for the engine checks

LEG_URDF = (
    '<robot name="leg">'
    '<link name="hip"/><link name="thigh"/><link name="shin"/><link name="foot"/>'
    '<joint name="knee" type="revolute"><parent link="thigh"/><child link="shin"/>'
    '<origin xyz="{L} 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>'
    '<limit lower="0.0" upper="2.70" effort="120" velocity="8"/></joint>'
    '<joint name="hip" type="revolute"><parent link="hip"/><child link="thigh"/>'
    '<origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>'
    '<limit lower="-2.20" upper="2.20" effort="200" velocity="10"/></joint>'
    '<joint name="ankle" type="fixed"><parent link="shin"/><child link="foot"/>'
    '<origin xyz="{L} 0 0" rpy="0 0 0"/></joint>'
    '</robot>').format(L='%.6f' % LEG_L)

_LEG_ROBOT = None


def leg_robot():
    """The URDF, built once. Its FK is the witness for the closed form below."""
    global _LEG_ROBOT
    if _LEG_ROBOT is None:
        import tempfile
        d = tempfile.mkdtemp(prefix='spine_leg_')
        p = d + '\\leg.urdf'
        with open(p, 'w', encoding='utf-8') as fh:
            fh.write(LEG_URDF)
        _LEG_ROBOT = Urdf(p)
    return _LEG_ROBOT


def leg_fk(q):
    """Foot position in the leg plane (x = forward, y = down), via the URDF."""
    T = leg_robot().fk({'hip': q[0], 'knee': q[1]}, 'foot')
    return [T[0][3], T[1][3]]


def leg_ik(fwd, down, l1=LEG_L, l2=LEG_L, branch='knee_forward'):
    """Closed-form 2-link IK in the leg plane. Returns (q1, q2, ok, why).

    `down` is positive downwards, so a foot hanging under the hip is (0, 0.9).
    Out of reach is reported, not raised: a controller that cannot reach a target
    is a normal event, and the gate needs to name it (`ik_unreachable`) rather
    than crash on it.

    The first version of this returned pi - interior for the knee angle, which is
    the angle between the two links rather than the joint angle between them; a
    straight leg came out as a knee folded 180 degrees. The check that caught it is
    leg_fk() below, which is the URDF's own forward kinematics -- the arithmetic
    says the answer, the URDF says whether the arithmetic means what I think.
    """
    d2 = fwd * fwd + down * down
    d = math.sqrt(d2)
    reach = l1 + l2
    if d > reach * 0.999999:
        return 0.0, 0.0, False, 'unreachable: %.4f m from hip, leg is %.4f m' % (d, reach)
    if d < 1e-6:
        return 0.0, 0.0, False, 'degenerate: target at the hip'
    cos_knee = (d2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    interior = math.acos(max(-1.0, min(1.0, cos_knee)))   # 0 = straight leg
    cos_hip = (l1 * l1 + d2 - l2 * l2) / (2.0 * l1 * d)
    alpha = math.acos(max(-1.0, min(1.0, cos_hip)))        # thigh offset from the hip-foot line
    theta = math.atan2(down, fwd)                          # direction of the target
    candidates = []
    for q2 in (interior, -interior):
        q1 = theta - alpha if q2 >= 0.0 else theta + alpha
        knee = [l1 * math.cos(q1), l1 * math.sin(q1)]
        foot = [knee[0] + l2 * math.cos(q1 + q2), knee[1] + l2 * math.sin(q1 + q2)]
        err = math.hypot(foot[0] - fwd, foot[1] - down)
        candidates.append((err, q1, q2, knee[0]))
    exact = [c for c in candidates if c[0] < 1e-9]
    if not exact:
        err, q1, q2, _ = min(candidates)
        return q1, q2, False, 'closed form misses the target by %.3e m' % err
    if branch == 'knee_forward':
        err, q1, q2, _ = max(exact, key=lambda t: t[3])     # knee leads
    elif branch == 'knee_back':
        err, q1, q2, _ = min(exact, key=lambda t: t[3])
    else:
        err, q1, q2, _ = exact[0]
    return q1, q2, True, 'ok'


def leg_dls(fwd, down, q0=None, tol=1e-8):
    """The same foot target through ik.py's damped least squares. An independent witness.

    Position only, on purpose. Asked to hold the foot's ORIENTATION as well, the
    solver declares the start dead immediately: a 2-DOF planar leg cannot hold a
    position and an orientation at the same time, so the merit function it is
    minimising has no descent direction. That is a fact about the leg, not a bug in
    the solver, and the bench reports both rates instead of only the good one. The
    foot's orientation is set by the ground it stands on, so position is the
    constraint the controller actually has.
    """
    robot = leg_robot()
    q0 = list(q0) if q0 else [0.0, 0.0]
    res, tries = ik.ik_multistart(robot, 'foot', [fwd, down, 0.0], None,
                                  seeds=4, tol=tol, max_iter=300)
    return res, tries


class State:
    """Where the body is: feet position, yaw, stance, and the velocity that got it here."""
    __slots__ = ('x', 'y', 'z', 'yaw', 'vx', 'vy', 'vz', 'on_ground', 'sneak', 't', 'legs')

    def __init__(self, x=0.5, y=65.0, z=0.5, yaw=0.0, vx=0.0, vy=0.0, vz=0.0,
                 on_ground=True, sneak=False, t=0.0, legs=None):
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.yaw = float(yaw)
        self.vx, self.vy, self.vz = float(vx), float(vy), float(vz)
        self.on_ground, self.sneak, self.t = on_ground, sneak, t
        self.legs = legs or [[0.0, 0.0], [0.0, 0.0]]

    def copy(self):
        return State(self.x, self.y, self.z, self.yaw, self.vx, self.vy, self.vz,
                     self.on_ground, self.sneak, self.t, [list(q) for q in self.legs])

    def pos(self):
        return [self.x, self.y, self.z]

    def __repr__(self):
        return '<State (%.3f, %.3f, %.3f) yaw %.2f v(%.2f, %.2f, %.2f) %s>' % (
            self.x, self.y, self.z, self.yaw, self.vx, self.vy, self.vz,
            'ground' if self.on_ground else 'air')


class Body:
    """The articulated body: proportions fixed here, pose supplied by a State."""

    def __init__(self, leg_l=LEG_L, pelvis_h=PELVIS_H, hip_sep=HIP_SEP,
                 torso_r=TORSO_R, head_r=HEAD_R):
        self.leg_l, self.pelvis_h, self.hip_sep = leg_l, pelvis_h, hip_sep
        self.torso_r, self.head_r = torso_r, head_r

    # -- frames ----------------------------------------------------------
    @staticmethod
    def forward(yaw):
        return [math.cos(yaw), 0.0, math.sin(yaw)]

    @staticmethod
    def right(yaw):
        return [math.sin(yaw), 0.0, -math.cos(yaw)]

    # -- legs ------------------------------------------------------------
    def solve_legs(self, st, stride=0.0, lift=0.0):
        """Place both feet for a step of `stride`; returns (legs, ok, why).

        The stance foot is under the hip; the swing foot is `stride` ahead and
        `lift` above the ground. Standing still is stride=0, lift=0, which is the
        target both solvers are checked against in the bench.
        """
        out, problems = [], []
        for side in (-1, 1):
            if stride == 0.0 or (side < 0) != (stride < 0.0):
                q, ok, why = self.step_foot(st, side, 0.0, 0.0)
            else:
                q, ok, why = self.step_foot(st, side, abs(stride), lift)
            if not ok:
                problems.append('leg%d: %s' % (0 if side < 0 else 1, why))
            out.append(q)
        return out, not problems, '; '.join(problems)

    def step_foot(self, st, side, stride, lift=0.0):
        """One leg's joint angles for a foot `stride` ahead and `lift` above ground."""
        down = self.pelvis_h - lift
        q1, q2, ok, why = leg_ik(stride, down, self.leg_l, self.leg_l)
        return [q1, q2], ok, why

    def stand(self, st):
        """Put the body in a legal standing pose, and say whether the legs reached it."""
        legs, ok, why = self.solve_legs(st)
        st.legs = legs
        return ok, why

    def leg_points(self, st, side):
        """(hip, knee, foot) in world for one leg. One place, so the capsule and
        the FK cannot disagree about where a leg is."""
        f = self.forward(st.yaw)
        hip = [st.x + side * self.hip_sep * self.right(st.yaw)[0],
               st.y + self.pelvis_h,
               st.z + side * self.hip_sep * self.right(st.yaw)[2]]
        q1, q2 = st.legs[0 if side < 0 else 1]
        knee_xy = [self.leg_l * math.cos(q1), self.leg_l * math.sin(q1)]
        foot_xy = [knee_xy[0] + self.leg_l * math.cos(q1 + q2),
                   knee_xy[1] + self.leg_l * math.sin(q1 + q2)]
        knee = [hip[0] + knee_xy[0] * f[0], hip[1] - knee_xy[1], hip[2] + knee_xy[0] * f[2]]
        foot = [hip[0] + foot_xy[0] * f[0], hip[1] - foot_xy[1], hip[2] + foot_xy[0] * f[2]]
        return hip, knee, foot

    def capsules(self, st):
        """World capsules for the whole body at pose `st`: [(name, p, q, r0, r1)]."""
        f, r = self.forward(st.yaw), self.right(st.yaw)

        def at(fwd, up, lat=0.0):
            return [st.x + fwd * f[0] + lat * r[0],
                    st.y + up,
                    st.z + fwd * f[2] + lat * r[2]]

        caps = [('torso', at(0.0, 0.02), at(0.0, NECK_H - 0.02), self.torso_r,
                 self.torso_r * 0.85),
                ('head', at(0.0, NECK_H + 0.02), at(0.0, NECK_H + 0.10),
                 self.head_r, self.head_r)]
        for i, side in enumerate((-1, 1)):
            hip, knee, foot = self.leg_points(st, side)
            caps.append(('thigh%d' % i, hip, knee, THIGH_R0, THIGH_R1))
            caps.append(('shin%d' % i, knee, foot, SHIN_R0, SHIN_R1))
        return caps

    def hitbox(self, st):
        hw = HITBOX_W / 2.0
        h = HITBOX_W / 2.0
        return ([-hw, st.y, -h], [hw, st.y + HITBOX_H, h]), (st.x, st.y, st.z)
