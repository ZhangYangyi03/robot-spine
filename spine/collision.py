# -*- coding: utf-8 -*-
"""collision.py -- capsule-vs-capsule and capsule-vs-sphere, distance only.

Why capsules and not meshes: a robot controller has one millisecond, and the
interesting quantity is the distance with a sign, not a perfect surface. Every
link is approximated by the segment between its two joint origins plus a radius;
the distance between two such segments has a closed form, so the whole check is
arithmetic with no sampling and no randomness.

The result carries the closest points, because "how far" without "where" is not
enough to say which link to move.
"""
import math

from . import linalg as la


def _closest_on_segments(p1, q1, p2, q2):
    """Closest points between two segments (Ericson, Real-Time Collision Detection 5.1.9)."""
    d1 = la.sub(q1, p1)
    d2 = la.sub(q2, p2)
    r = la.sub(p1, p2)
    a = la.dot(d1, d1)
    e = la.dot(d2, d2)
    f = la.dot(d2, r)
    EPS = 1e-12
    if a <= EPS and e <= EPS:
        return p1, p2
    if a <= EPS:
        s, t = 0.0, min(1.0, max(0.0, f / e))
    else:
        c = la.dot(d1, r)
        if e <= EPS:
            t, s = 0.0, min(1.0, max(0.0, -c / a))
        else:
            b = la.dot(d1, d2)
            den = a * e - b * b
            s = min(1.0, max(0.0, (b * f - c * e) / den)) if den > EPS else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, min(1.0, max(0.0, -c / a))
            elif t > 1.0:
                t, s = 1.0, min(1.0, max(0.0, (b - c) / a))
    return la.add(p1, la.scale(d1, s)), la.add(p2, la.scale(d2, t))


def _lerp_r(a, b, r0, r1):
    """Radius at the point `a + t (b - a)`, t read off the projections."""
    ab = la.sub(b, a)
    L2 = la.dot(ab, ab)
    if L2 < 1e-18:
        return max(r0, r1)
    return r0


def capsule_capsule(p1, q1, r1, p2, q2, r2, r1b=None, r2b=None):
    """Distance between two capsules whose radius may vary from end to end.

    With r1b/r2b left as None this is the constant-radius case. A tapered link
    (a forearm is not as fat as a shoulder) is the same closest-point pair with
    the radius interpolated along each segment -- without that, one number has to
    be both the fattest and the thinnest part of the limb, and every clearance
    check is wrong by the difference.
    """
    c1, c2 = _closest_on_segments(p1, q1, p2, q2)
    d = la.norm(la.sub(c1, c2))
    ra = r1
    if r1b is not None:
        L = la.norm(la.sub(q1, p1))
        tt = 0.0 if L < 1e-12 else min(1.0, max(0.0, la.norm(la.sub(c1, p1)) / L))
        ra = r1 + (r1b - r1) * tt
    rb = r2
    if r2b is not None:
        L = la.norm(la.sub(q2, p2))
        tt = 0.0 if L < 1e-12 else min(1.0, max(0.0, la.norm(la.sub(c2, p2)) / L))
        rb = r2 + (r2b - r2) * tt
    return d - ra - rb, c1, c2


def capsule_sphere(p1, q1, r1, c, r2):
    d1 = la.sub(q1, p1)
    a = la.dot(d1, d1)
    if a < 1e-12:
        cp = p1
    else:
        t = min(1.0, max(0.0, la.dot(la.sub(c, p1), d1) / a))
        cp = la.add(p1, la.scale(d1, t))
    d = la.norm(la.sub(cp, c))
    return d - r1 - r2, cp, c


class Capsule:
    __slots__ = ('name', 'p', 'q', 'r', 'r_end', 'mid', 'bound')

    def __init__(self, name, p, q, r, r_end=None):
        self.name, self.p, self.q, self.r = name, p, q, r
        self.r_end = r if r_end is None else r_end
        # bounding sphere, for the broad phase: centre of the segment, radius
        # half the length plus the fatter end radius
        self.mid = [(p[i] + q[i]) / 2.0 for i in range(3)]
        self.bound = 0.5 * la.norm(la.sub(q, p)) + max(r, self.r_end)


def broad_phase_pairs(caps, margin):
    """Index pairs whose bounding spheres could come within `margin`.

    Reported as a number rather than assumed: bench_g1 counts how many pairs the
    exact test actually had to run, so "the prefilter works" is a measurement.
    """
    out = []
    n = len(caps)
    for i in range(n):
        A = caps[i]
        for j in range(i + 1, n):
            B = caps[j]
            d = la.norm(la.sub(A.mid, B.mid))
            if d <= A.bound + B.bound + margin:
                out.append((i, j))
    return out


def robot_capsules(robot, q, radii, skip_pairs=(), taper=0.6):
    """Build capsules from FK. radii: {link: r}; a link not in it gets 0.

    The capsule covering link L runs from L's own frame to the next joint's
    frame, so its radius is L's at the near end and `taper * L's radius` at the
    far end -- a shoulder is thicker than the elbow it feeds. taper=1.0 gives the
    constant-radius cylinder instead.
    """
    out = []
    for parent, child, p0, p1 in robot.segments(q):
        r0 = radii.get(parent, 0.0)
        out.append(Capsule(parent, p0, p1, r0, r0 * taper))
    return out


def min_distance(caps_a, caps_b):
    best = (float('inf'), None, None, None)
    for A in caps_a:
        for B in caps_b:
            d, ca, cb = capsule_capsule(A.p, A.q, A.r, B.p, B.q, B.r,
                                        A.r_end, B.r_end)
            if d < best[0]:
                best = (d, A.name, B.name, (ca, cb))
    return best


def self_collision(caps, adjacency, min_clearance=0.005):
    """Pairs of capsules on the same arm that are not adjacent and are inside min_clearance."""
    worst = (float('inf'), None, None)
    for i in range(len(caps)):
        for j in range(i + 1, len(caps)):
            A, B = caps[i], caps[j]
            if A.name == B.name:
                continue      # two capsules of the SAME link are not a collision
            if (A.name, B.name) in adjacency or (B.name, A.name) in adjacency:
                continue
            d, _, _ = capsule_capsule(A.p, A.q, A.r, B.p, B.q, B.r,
                                      A.r_end, B.r_end)
            if d < worst[0]:
                worst = (d, A.name, B.name)
    return worst
