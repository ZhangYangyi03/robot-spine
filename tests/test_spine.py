# -*- coding: utf-8 -*-
"""The claims in README.md, as assertions. If one of these fails the README is wrong.

    python -m unittest discover -s tests -v
"""
import math
import os
import re
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from spine import collision as col          # noqa: E402
from spine import control, ik, scenes       # noqa: E402
from spine import gate as gatemod           # noqa: E402
from spine import linalg as la              # noqa: E402
from bench import bench_gate, bench_ik      # noqa: E402


class TestLinalg(unittest.TestCase):
    def test_rpy_matches_closed_form(self):
        # rpy = (0, 90deg, 90deg) is Rz(90) @ Ry(90); x-hat goes to -z-hat
        R = la.rpy_to_R([0.0, math.pi / 2, math.pi / 2])
        v = la.mat_vec(R, [1.0, 0.0, 0.0])
        for a, b in zip(v, [0.0, 0.0, -1.0]):
            self.assertAlmostEqual(a, b, places=12)

    def test_inv_T_is_inverse(self):
        T = la.eye(4)
        R = la.axis_angle_R([0.3, 0.5, 0.9], 0.7)
        for i in range(3):
            for j in range(3):
                T[i][j] = R[i][j]
            T[i][3] = [0.4, -0.2, 1.1][i]
        P = la.mat_mul(T, la.inv_T(T))
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(P[i][j], 1.0 if i == j else 0.0, places=12)

    def test_rotation_6d_round_trip(self):
        rng = random.Random(5)
        for _ in range(20):
            R = la.axis_angle_R([rng.uniform(-1, 1) for _ in range(3)], rng.uniform(-3, 3))
            R2 = la.r6_to_rot(la.rot_to_r6(R))
            for i in range(3):
                for j in range(3):
                    self.assertAlmostEqual(R[i][j], R2[i][j], places=10)

    def test_small_rotation_is_not_rounded_to_zero(self):
        # the bug this test exists for: so3_log returning zeros for a genuine
        # 1e-7 rad rotation, which makes every finite-difference check wrong
        R = la.axis_angle_R([0.0, 0.0, 1.0], 1e-7)
        w = la.so3_log(R)
        self.assertGreater(abs(w[2]), 5e-8)


class TestFK(unittest.TestCase):
    def setUp(self):
        self.robot, self.radii = scenes.two_link()

    def test_fk_matches_closed_form(self):
        rng = random.Random(0)
        for _ in range(50):
            q = [rng.uniform(-3, 3), rng.uniform(-3, 3)]
            T = self.robot.fk(self.robot.qdict(q), 'tip')
            ref = [0.4 * math.cos(q[0]) + 0.35 * math.cos(q[0] + q[1]),
                   0.4 * math.sin(q[0]) + 0.35 * math.sin(q[0] + q[1]), 0.1]
            for i in range(3):
                self.assertAlmostEqual(T[i][3], ref[i], places=12)

    def test_jacobian_matches_finite_difference(self):
        q = [0.4, -0.7]
        J = self.robot.world_jacobian(self.robot.qdict(q), 'tip')
        eps = 1e-5
        for k in range(2):
            qp, qm = list(q), list(q)
            qp[k] += eps
            qm[k] -= eps
            Tp = self.robot.fk(self.robot.qdict(qp), 'tip')
            Tm = self.robot.fk(self.robot.qdict(qm), 'tip')
            for i in range(3):
                self.assertAlmostEqual(J[i][k], (Tp[i][3] - Tm[i][3]) / (2 * eps), places=6)
            Rp = [[Tp[i][j] for j in range(3)] for i in range(3)]
            Rm = [[Tm[i][j] for j in range(3)] for i in range(3)]
            w = la.so3_log(la.mat_mul(Rp, la.transpose(Rm)))
            for i in range(3):
                self.assertAlmostEqual(J[i + 3][k], w[i] / (2 * eps), places=6)

    def test_links_outside_the_chain_get_zero_columns(self):
        J = self.robot.world_jacobian(self.robot.qdict([0.0, 0.0]), 'l1')
        self.assertEqual(len(J[0]), 2)
        self.assertEqual(J[0][1], 0.0)       # j2 is not on the path to l1


class TestCollision(unittest.TestCase):
    def test_known_distances(self):
        d, _, _ = col.capsule_capsule([0, 0, 0], [1, 0, 0], 0.0, [0, 0.5, 0], [1, 0.5, 0], 0.0)
        self.assertAlmostEqual(d, 0.5, places=12)
        d, _, _ = col.capsule_capsule([0, 0, 0], [1, 0, 0], 0.1, [0, 0.5, 0], [1, 0.5, 0], 0.1)
        self.assertAlmostEqual(d, 0.3, places=12)

    def test_capsule_sphere_sign(self):
        d, _, _ = col.capsule_sphere([0, 0, 0], [1, 0, 0], 0.05, [0.5, 0.2, 0], 0.10)
        self.assertAlmostEqual(d, 0.05, places=12)
        d, _, _ = col.capsule_sphere([0, 0, 0], [1, 0, 0], 0.05, [0.5, 0.0, 0], 0.10)
        self.assertLess(d, 0.0)             # overlapping is negative, not clamped


class TestIK(unittest.TestCase):
    def setUp(self):
        self.robot, _ = scenes.two_link()

    def test_reachable_targets_are_solved(self):
        res = bench_ik.run(self.robot, n_targets=40)
        self.assertGreaterEqual(res['multi_start']['rate'], 0.97)
        self.assertLess(res['multi_start']['worst_residual_m'], 1e-6)

    def test_single_start_is_measurably_worse_than_multi_start(self):
        # the README claims this gap; if a future change closes it, the claim is stale
        res = bench_ik.run(self.robot, n_targets=40)
        self.assertLess(res['single_start']['rate'], 0.90)

    def test_unreachable_target_returns_data_and_does_not_raise(self):
        r = ik.ik_solve(self.robot, 'tip', [0.0, 0.0], [1.5, 0.0, 0.1], None, max_iter=100)
        self.assertFalse(r.ok)
        self.assertGreater(r.pos_err, 0.5)

    def test_joint_limits_are_respected(self):
        r = ik.ik_solve(self.robot, 'tip', [0.0, 0.0], [0.74, 0.0, 0.1], None, max_iter=200)
        over, _ = ik.qlim_violation(self.robot, r.q)
        self.assertEqual(over, 0.0)

    def test_gate_inside_the_solver_is_obeyed(self):
        # a gate that forbids j2 < 0 must never see a returned solution with j2 < 0
        def no_negative_j2(q):
            return (q[1] >= 0.0, 'j2 must stay positive')
        r = ik.ik_solve(self.robot, 'tip', [0.0, 0.0], [0.2, 0.3, 0.1], None,
                        max_iter=200, gate=no_negative_j2)
        self.assertGreaterEqual(r.q[1], 0.0)


class TestGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = bench_gate.run(n_legal=100, n_faults=20)

    def test_legal_commands_are_admitted(self):
        self.assertEqual(self.res['legal']['false_refusal_rate'], 0.0)

    def test_every_planted_fault_is_caught_by_its_own_rule(self):
        for rule, row in self.res['per_rule'].items():
            self.assertEqual(row['caught'], row['planted'], rule)
            self.assertEqual(row['caught_by_intended_rule'], row['planted'], rule)

    def test_mutation_control_collapses_for_step_jump(self):
        # with the step rule off, a step break must go through: that is what makes
        # "caught by the step rule" a measurement instead of a coincidence
        self.assertEqual(self.res['mutation_control']['step_jump']['caught_with_rule_off'], 0)

    def test_obstacle_scene_discriminates(self):
        g = self.res['geometry_scene']
        self.assertFalse(g['obstacle_at_reach']['admitted'])
        self.assertTrue(g['obstacle_away']['admitted'])
        self.assertFalse(g['folded_self_collision']['admitted'])
        self.assertTrue(g['straight_clear']['admitted'])


class TestControl(unittest.TestCase):
    def test_gravity_compensation_removes_the_steady_state_error(self):
        plant = control.PlanarArm()
        plant.q = [0.6, 0.9]
        ctrl = control.PD([18.0, 8.0], [2.2, 1.1], plant=plant, comp=False)
        tr, _, _, _ = control.run(plant, ctrl, [0.0, 0.0], t_end=3.0, dt=0.002)
        naive = control.metrics(tr, [0.0, 0.0])['steady_err']
        self.assertGreater(naive, 0.2)

        plant = control.PlanarArm()
        plant.q = [0.6, 0.9]
        ctrl = control.PD([18.0, 8.0], [2.2, 1.1], plant=plant, comp=True)
        tr, _, _, _ = control.run(plant, ctrl, [0.0, 0.0], t_end=3.0, dt=0.002)
        comp = control.metrics(tr, [0.0, 0.0])['steady_err']
        self.assertLess(comp, 1e-3)

    def test_joint_stops_hold(self):
        plant = control.PlanarArm()
        ctrl = control.PD([200.0, 200.0], [0.0, 0.0], plant=plant, comp=False)
        tr, _, _, extra = control.run(plant, ctrl, [3.0, 3.0], t_end=1.0, dt=0.002,
                                      q_min=[-3.0, -3.0], q_max=[3.0, 3.0])
        self.assertLessEqual(max(abs(x) for x in plant.q), 3.0 + 1e-9)
        self.assertGreater(extra['joint_stop_contacts'], 0)

    def test_gate_stops_a_rogue_controller_and_not_a_good_one(self):
        robot, radii = scenes.two_link()
        lims = gatemod.Limits(robot, max_step=0.35, min_clearance=0.02, radii=radii,
                              tau={n: 14.0 for n in robot.joint_names()},
                              vel={n: 6.0 for n in robot.joint_names()})
        adj = scenes.adjacency_of(robot)

        def make(rogue, gated):
            plant = control.PlanarArm()
            plant.q = [0.6, 0.9]
            c = (control.PD([200., 120.], [0., 0.], plant=plant, comp=True) if rogue
                 else control.PD([18., 8.], [2.2, 1.1], plant=plant, comp=True))
            g = gatemod.Gate(lims, obstacles=[], adjacency=adj) if gated else None
            qmin = [l[0] for l in robot.limits()]
            qmax = [l[1] for l in robot.limits()]
            return control.run(plant, c, [0.0, 0.0], t_end=2.0, dt=0.002, gate=g,
                               q_min=qmin, q_max=qmax), plant

        (tr0, r0, _, _), p0 = make(True, False)
        (tr1, r1, _, _), p1 = make(True, True)
        self.assertEqual(r0, 0)                 # ungated: the bad command gets through
        self.assertEqual(r1, len(tr1))          # gated: every tick refused
        self.assertLessEqual(max(abs(t[3][i]) for t in tr1 for i in range(2)), 0.0)

        (tr2, r2, _, _), p2 = make(False, True)
        self.assertEqual(r2, 0)                 # the good controller loses nothing


if __name__ == '__main__':
    unittest.main()


class TestBroadPhase(unittest.TestCase):
    """The prefilter must change the cost and not the answer. That is its whole job."""

    def test_prefilter_does_not_change_the_verdict(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, 'demo', 'g1.urdf')
        if not os.path.exists(path):
            self.skipTest('demo/g1.urdf not present')
        from spine.urdf import Urdf
        from spine import collision as col
        u = Urdf(path)
        radii = dict((l, 0.03) for l in u.links)
        rng = random.Random(11)
        names = u.joint_names()
        margin = 0.05
        for _ in range(10):
            q = [rng.uniform(-0.4, 0.4) for _ in names]
            caps = col.robot_capsules(u, u.qdict(q), radii)
            kept = set(col.broad_phase_pairs(caps, margin))
            for i in range(len(caps)):
                for j in range(i + 1, len(caps)):
                    A, B = caps[i], caps[j]
                    d, _, _ = col.capsule_capsule(A.p, A.q, A.r, B.p, B.q, B.r,
                                                  A.r_end, B.r_end)
                    # a pair the prefilter dropped must be further than the margin,
                    # which is exactly the promise the wrong answer would break
                    if (i, j) not in kept:
                        self.assertGreater(d, margin, (A.name, B.name))


class TestCalibration(unittest.TestCase):
    """Calibration must baseline the model's structural overlaps, not mute the rule."""

    def test_calibrated_gate_still_catches_a_real_fold(self):
        robot, radii = scenes.two_link()
        adj = scenes.adjacency_of(robot)
        lim = gatemod.Limits(robot, max_step=0.35, min_clearance=0.02, radii=radii)
        q0 = [0.0, 0.0]
        g = gatemod.Gate(lim, obstacles=[], adjacency=adj, calibrate=q0)
        self.assertTrue(g.check(q0, qdot=[0.0, 0.0], tau=[0.0, 0.0]).ok)
        v = g.check([0.0, 3.0], qdot=[0.0, 0.0], tau=[0.0, 0.0])
        self.assertFalse(v.ok)
        self.assertIn('self_collision', v.reasons)


class TestMuJoCoAgreement(unittest.TestCase):
    """The plant against an engine that did not write it.

    Skipped, loudly, when MuJoCo cannot be imported -- a skipped physics check and a
    passed one must never look the same, so the skip message names the reason.
    """

    @classmethod
    def setUpClass(cls):
        from spine import mujoco_backend as mj
        ok, why = mj.available()
        if not ok:
            raise unittest.SkipTest('mujoco not importable here: %s' % why)
        cls.mj = mj
        cls.out = mj.run(grid=[[0.0, 0.0], [0.6, 0.9], [-0.5, 0.2], [1.1, -0.7]])

    def test_gravity_agrees_with_the_engine(self):
        from spine import control
        plant = control.PlanarArm()
        for q, row in zip([[0.0, 0.0], [0.6, 0.9], [-0.5, 0.2], [1.1, -0.7]],
                          self.out['mujoco_bias']):
            mine = plant.G(q)
            for k in (0, 1):
                self.assertLess(abs(mine[k] - row[k]), 1e-9,
                                'G(q) and qfrc_bias differ at %r axis %d' % (q, k))

    def test_same_pd_reaches_the_same_fixed_point(self):
        a = self.out['loop']['final_q']
        self.assertLess(max(abs(x - y) for x, y in zip(a, [-0.41166928, -0.16216466])),
                        1e-6, 'the PD fixed point moved')

    def test_engine_bias_compensation_holds_the_setpoint(self):
        q = self.out['loop_with_bias_comp']['final_q']
        self.assertLess(max(abs(x) for x in q), 1e-4)

    def test_the_model_has_no_joint_limits(self):
        # the decision in docs/MUJOCO.md: limits belong to the gate, not the engine.
        # If someone adds a `range` back, this test names why it must not.
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'demo', 'two_link.xml')
        with open(path, encoding='utf-8') as fh:
            xml = fh.read()
        self.assertNotIn(None, re.findall(r'<joint[^>]*\brange="', xml),
                         'a MuJoCo joint range clamps at every pose in 3.2.7, so the '
                         'engine would solve a different problem from the plant; '
                         'joint limits are enforced by spine/gate.py')
