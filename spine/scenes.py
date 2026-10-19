# -*- coding: utf-8 -*-
"""scenes.py -- the two robots this repo actually runs on, and their link radii.

    two_link    a 2-DOF planar arm as a URDF, with the same numbers the analytic
                plant in control.py uses, so a solver's answer can be checked
                against a formula rather than against another solver
    g1          demo/g1.urdf from unifolm-wla-toolkit, 29 movable joints, used to
                show the same code does not fall over at scale

Link radii are hand-set capsules, not measured from meshes: the URDFs here carry
no collision geometry, and pretending otherwise would be the wrong kind of tidy.
"""
import os
import tempfile

from .urdf import Urdf

TWO_LINK = '\n'.join([
    '<robot name="two_link">',
    '  <link name="base"/><link name="l1"/><link name="l2"/><link name="tip"/>',
    '  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>',
    '    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/>',
    '    <limit lower="-3.0" upper="3.0" effort="30" velocity="3"/></joint>',
    '  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>',
    '    <origin xyz="0.4 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>',
    '    <limit lower="-3.0" upper="3.0" effort="30" velocity="3"/></joint>',
    '  <joint name="jt" type="fixed"><parent link="l2"/><child link="tip"/>',
    '    <origin xyz="0.35 0 0" rpy="0 0 0"/></joint>',
    '</robot>'])

TWO_LINK_RADII = {'base': 0.05, 'l1': 0.05, 'l2': 0.05, 'tip': 0.03}


def two_link():
    d = tempfile.mkdtemp(prefix='robot_spine_')
    p = os.path.join(d, 'two_link.urdf')
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write(TWO_LINK)
    return Urdf(p), TWO_LINK_RADII


def g1():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(here, 'demo', 'g1.urdf')
    if not os.path.exists(p):
        return None, None
    return Urdf(p), G1_RADII


# rough capsule radii per link family; a real machine would carry these in the
# URDF, these two do not, and the bench says so in its own output
G1_RADII = {}


def adjacency_of(robot, depth=1):
    """Link pairs that are exempt from the self-collision rule.

    Two links that share a joint always touch, so they are exempt. Everything
    else is checked -- including a link and its grandchild -- and the structural
    overlaps that a fat hand-set capsule produces are removed by CALIBRATION
    (Gate.calibrate_at) instead of by widening the exemption. Widening the
    exemption until the alarm stops is the same move as widening a tolerance
    until the test passes; measuring a healthy pose and comparing to it is not.

    The measured consequence is in bench/results.json under
    self_collision_calibration.
    """
    # undirected link graph from the joints
    graph = {}
    for j in robot.joints:
        graph.setdefault(j.parent, set()).add(j.child)
        graph.setdefault(j.child, set()).add(j.parent)
    pairs = set()
    for start in graph:
        # breadth-first to `depth` joints away
        frontier, seen = {start}, {start}
        for _ in range(depth):
            nxt = set()
            for u in frontier:
                for v in graph.get(u, ()):
                    if v not in seen:
                        nxt.add(v)
            seen |= nxt
            frontier = nxt
        for v in seen:
            if v != start:
                pairs.add((start, v))
                pairs.add((v, start))
    return pairs
