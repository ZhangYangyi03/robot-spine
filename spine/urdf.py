# -*- coding: utf-8 -*-
"""urdf.py -- URDF kinematic tree, forward kinematics, geometric Jacobian.

The URDF reader is unifolm-wla-toolkit/urdf_kin.py's, kept verbatim in spirit
(same joint model, same origin xyz+rpy, same fixed-axis convention) but on the
stdlib linalg here so this repo has no numpy dependency at all.

What is added to it, because the rest of this repo needs it:
  * world_jacobian() -- 6 x n geometric Jacobian by the standard z x (p - o) rule
  * joint metadata in movable-joint order, with limits, for the gate
"""
import math
import xml.etree.ElementTree as ET

from . import linalg as la


class Joint:
    def __init__(self, name, jtype, parent, child, origin, axis, limit):
        self.name, self.jtype = name, jtype
        self.parent, self.child = parent, child
        self.origin, self.axis, self.limit = origin, axis, limit

    def motion(self, q):
        T = la.eye(4)
        if self.jtype in ('revolute', 'continuous'):
            R = la.axis_angle_R(self.axis, q)
            for i in range(3):
                for j in range(3):
                    T[i][j] = R[i][j]
        elif self.jtype == 'prismatic':
            n = la.norm(self.axis) or 1.0
            for i in range(3):
                T[i][3] = self.axis[i] / n * q
        elif self.jtype != 'fixed':
            raise ValueError('joint type %r not supported' % self.jtype)
        return T

    def __repr__(self):
        return '<Joint %s %s %s->%s>' % (self.name, self.jtype, self.parent, self.child)


class Urdf:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.name = root.get('name')
        self.links = [l.get('name') for l in root.findall('link')]
        self.joints = []
        for j in root.findall('joint'):
            origin = la.eye(4)
            o = j.find('origin')
            if o is not None:
                xyz = [float(v) for v in (o.get('xyz') or '0 0 0').split()]
                rpy = [float(v) for v in (o.get('rpy') or '0 0 0').split()]
                R = la.rpy_to_R(rpy)
                for i in range(3):
                    for k in range(3):
                        origin[i][k] = R[i][k]
                    origin[i][3] = xyz[i]
            ax = j.find('axis')
            axis = [float(v) for v in (ax.get('xyz') or '1 0 0').split()] \
                if ax is not None else [1.0, 0.0, 0.0]
            lim = j.find('limit')
            limit = (float(lim.get('lower', -1e9)), float(lim.get('upper', 1e9))) \
                if lim is not None else None
            self.joints.append(Joint(j.get('name'), j.get('type'),
                                     j.find('parent').get('link'),
                                     j.find('child').get('link'),
                                     origin, axis, limit))
        self.by_child = {j.child: j for j in self.joints}
        self.by_name = {j.name: j for j in self.joints}
        children = set(self.by_child)
        self.root = next((l for l in self.links if l not in children), None)
        if self.root is None:
            raise ValueError('no root link found (cycle in the kinematic tree?)')

    # ---- topology -------------------------------------------------------
    def chain(self, link):
        out, cur = [], link
        while cur != self.root:
            if cur not in self.by_child:
                raise KeyError('%s is not connected to the root %s' % (link, self.root))
            j = self.by_child[cur]
            out.append(j)
            cur = j.parent
        return list(reversed(out))

    def movable_joints(self):
        return [j for j in self.joints if j.jtype != 'fixed']

    def joint_names(self):
        return [j.name for j in self.movable_joints()]

    def limits(self):
        out = []
        for j in self.movable_joints():
            out.append(j.limit if j.limit else (-math.pi, math.pi))
        return out

    # ---- kinematics -----------------------------------------------------
    def fk_all(self, q):
        """Every link frame. Document order is NOT assumed to be topological.

        The first version walked self.joints once and skipped any joint whose
        parent frame was not built yet, which silently drops links when a URDF
        lists a child joint before its parent -- and a missing link surfaces later
        as a KeyError somewhere else, or worse as a capsule built from a frame that
        does not exist. Now the pass repeats until it stops making progress, and a
        joint that never resolves is a hard error naming itself.
        """
        T = {self.root: la.eye(4)}
        pending = list(self.joints)
        while pending:
            deferred, progress = [], False
            for j in pending:
                if j.parent in T:
                    T[j.child] = la.mat_mul(la.mat_mul(T[j.parent], j.origin),
                                            j.motion(float((q or {}).get(j.name, 0.0))))
                    progress = True
                else:
                    deferred.append(j)
            if not progress:
                raise ValueError('kinematic tree is not rooted at %r: %s unreachable'
                                 % (self.root, [j.name for j in deferred]))
            pending = deferred
        return T

    def fk(self, q, link=None):
        if not link:
            return la.eye(4)
        return self.fk_all(q)[link]

    def world_jacobian(self, q, link):
        """6 x n geometric Jacobian, columns in movable-joint order.

        A joint outside the root->link chain contributes a zero column, so the
        matrix always has the full n columns and IK never has to renumber.
        """
        T = self.fk_all(q)
        p_ee = [T[link][i][3] for i in range(3)]
        on_chain = set(j.name for j in self.chain(link))
        cols = []
        for j in self.movable_joints():
            if j.name not in on_chain:
                cols.append([0.0] * 6)
                continue
            Tj = la.mat_mul(T[j.parent], j.origin)
            ax = la.scale(j.axis, 1.0 / (la.norm(j.axis) or 1.0))
            z = [sum(Tj[i][k] * ax[k] for k in range(3)) for i in range(3)]
            o = [Tj[i][3] for i in range(3)]
            if j.jtype == 'prismatic':
                cols.append(z + [0.0, 0.0, 0.0])
            else:
                cols.append(la.cross(z, la.sub(p_ee, o)) + z)
        return [[cols[c][r] for c in range(len(cols))] for r in range(6)]

    def qdict(self, qvec):
        return {n: float(v) for n, v in zip(self.joint_names(), qvec)}

    def segments(self, q, min_length=1e-6):
        """(link, child_link, p0, p1) in world: one capsule per link->child-joint.

        The span is the link's OWN frame to the next joint's frame. The earlier
        version spanned parent-frame to child-frame, which turned every decorative
        link sitting at a coincident origin (head_link, logo_link, imu_in_pelvis,
        pelvis_contour_link on the G1) into a fat phantom capsule around the
        pelvis and made the self-collision rule fire on the zero pose. A capsule
        derived from the wrong span is not a conservative approximation, it is a
        wrong model, and the bench caught it.
        """
        T = self.fk_all(q)
        out = []
        for j in self.joints:
            if j.parent not in T or j.child not in T:
                continue
            p0 = [T[j.parent][i][3] for i in range(3)]
            p1 = [T[j.child][i][3] for i in range(3)]
            if la.norm(la.sub(p1, p0)) < min_length:
                continue                       # zero-extent link: no capsule
            out.append((j.parent, j.child, p0, p1))
        return out
