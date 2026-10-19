# -*- coding: utf-8 -*-
"""control.py -- a plant and a PD controller, small enough to reason about.

The plant is a planar 2-link arm with the standard Lagrangian equations (mass
matrix, Coriolis, gravity). It is the smallest plant that still punishes a naive
controller: gravity torque depends on configuration, and the mass matrix couples
the two joints, so a PD that ignores both shows a steady-state error that a PD
with gravity compensation does not -- and the bench measures exactly that.

Nothing in here is a black box: M, C, G are written out, so the numbers the bench
reports can be checked by hand.
"""
import math


class PlanarArm:
    """masses at the link centres, links in a vertical plane, g along -y."""

    def __init__(self, l1=0.40, l2=0.35, m1=1.6, m2=0.9, g=9.81):
        self.l1, self.l2, self.m1, self.m2, self.g = l1, l2, m1, m2, g
        self.l1c, self.l2c = l1 / 2.0, l2 / 2.0
        self.I1 = m1 * l1 ** 2 / 12.0
        self.I2 = m2 * l2 ** 2 / 12.0
        self.q = [0.0, 0.0]
        self.qd = [0.0, 0.0]

    # -- terms -----------------------------------------------------------
    def M(self, q=None):
        q = q or self.q
        c2 = math.cos(q[1])
        M11 = self.m1 * self.l1c ** 2 + self.m2 * (self.l1 ** 2 + self.l2c ** 2 +
                                                   2 * self.l1 * self.l2c * c2) + self.I1 + self.I2
        M12 = self.m2 * (self.l2c ** 2 + self.l1 * self.l2c * c2) + self.I2
        M22 = self.m2 * self.l2c ** 2 + self.I2
        return [[M11, M12], [M12, M22]]

    def C(self, qd=None, q=None):
        q = q or self.q
        qd = qd or self.qd
        h = -self.m2 * self.l1 * self.l2c * math.sin(q[1])
        return [[h * qd[1], h * (qd[0] + qd[1])], [-h * qd[0], 0.0]]

    def G(self, q=None):
        q = q or self.q
        G1 = (self.m1 * self.l1c + self.m2 * self.l1) * self.g * math.cos(q[0]) + \
             self.m2 * self.l2c * self.g * math.cos(q[0] + q[1])
        G2 = self.m2 * self.l2c * self.g * math.cos(q[0] + q[1])
        return [G1, G2]

    def fk(self, q=None):
        q = q or self.q
        return [self.l1 * math.cos(q[0]) + self.l2 * math.cos(q[0] + q[1]),
                self.l1 * math.sin(q[0]) + self.l2 * math.sin(q[0] + q[1])]

    # -- integration -----------------------------------------------------
    def step(self, tau, dt, q_min=None, q_max=None):
        """Explicit RK4 on the Lagrangian dynamics; stiff terms are not an issue here."""
        def deriv(q, qd):
            M = self.M(q)
            C = self.C(qd, q)
            G = self.G(q)
            rhs = [tau[i] - (C[i][0] * qd[0] + C[i][1] * qd[1]) - G[i] for i in range(2)]
            b = M[0][0] * M[1][1] - M[0][1] * M[1][0]
            if abs(b) < 1e-15:
                return qd, [0.0, 0.0]
            qdd = [(M[1][1] * rhs[0] - M[0][1] * rhs[1]) / b,
                   (-M[1][0] * rhs[0] + M[0][0] * rhs[1]) / b]
            return qd, qdd
        q, qd = list(self.q), list(self.qd)
        k1q, k1v = deriv(q, qd)
        k2q, k2v = deriv([q[i] + dt / 2 * k1q[i] for i in range(2)],
                         [qd[i] + dt / 2 * k1v[i] for i in range(2)])
        k3q, k3v = deriv([q[i] + dt / 2 * k2q[i] for i in range(2)],
                         [qd[i] + dt / 2 * k2v[i] for i in range(2)])
        k4q, k4v = deriv([q[i] + dt * k3q[i] for i in range(2)],
                         [qd[i] + dt * k3v[i] for i in range(2)])
        self.q = [q[i] + dt / 6 * (k1q[i] + 2 * k2q[i] + 2 * k3q[i] + k4q[i]) for i in range(2)]
        self.qd = [qd[i] + dt / 6 * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i]) for i in range(2)]
        # joint stops. A real arm has them, and without them the integration is
        # free to run an angle to 11 rad -- not a pose the URDF can be in, which
        # would make the ungated numbers below physically meaningless.
        self.contacts = 0
        if q_min is not None and q_max is not None:
            for i in range(2):
                if self.q[i] < q_min[i]:
                    self.q[i], self.qd[i] = q_min[i], max(0.0, self.qd[i])
                    self.contacts += 1
                elif self.q[i] > q_max[i]:
                    self.q[i], self.qd[i] = q_max[i], min(0.0, self.qd[i])
                    self.contacts += 1
        return self.q


class PD:
    """Joint-space PD, optionally with gravity compensation.

    comp=False is the naive controller: it lets the plant's own gravity torque
    stand, so the arm settles wherever PD torque balances gravity -- not at the
    setpoint. The gap between the two is the first number the bench reports.
    """

    def __init__(self, kp, kd, plant=None, comp=False):
        self.kp, self.kd = kp, kd
        self.plant, self.comp = plant, comp

    def torque(self, q, qd, q_des):
        out = []
        for i in range(2):
            t = self.kp[i] * (q_des[i] - q[i]) - self.kd[i] * qd[i]
            if self.comp:
                t += self.plant.G(q)[i]
            out.append(t)
        return out


def run(plant, ctrl, q_des, t_end=4.0, dt=0.002, gate=None, q_min=None, q_max=None):
    """Closed loop. Returns the trace, and how many commands the gate refused.

    When a gate is given, a refused torque command falls back to zero torque for
    that tick (a real drive holds or coasts; it does not run the bad command).
    """
    t, trace, refused, first_refusal, contacts = 0.0, [], 0, None, 0
    cmd_peak = 0.0
    n = int(t_end / dt)
    for _ in range(n):
        tau = ctrl.torque(plant.q, plant.qd, q_des)
        cmd_peak = max(cmd_peak, max([abs(x) for x in tau] or [0.0]))
        if gate is not None:
            v = gate.check(plant.q, plant.qd, tau)
            if not v.ok:
                refused += 1
                if first_refusal is None:
                    first_refusal = (round(t, 3), v.reasons, v.detail)
                tau = [0.0, 0.0]
        plant.step(tau, dt, q_min, q_max)
        contacts += getattr(plant, 'contacts', 0)
        t += dt
        trace.append((t, list(plant.q), list(plant.qd), list(tau)))
    # cmd_peak is what the CONTROLLER asked for; the tau inside the trace is
    # what actually reached the plant after the gate. Two different numbers, and
    # conflating them would let a refusing gate look like a controller.
    return trace, refused, first_refusal, {'joint_stop_contacts': contacts,
                                           'peak_command': cmd_peak}


def metrics(trace, q_des, band=0.02):
    """Settling time, overshoot, steady-state error -- the three everyone quotes."""
    err = [max(abs(t[1][i] - q_des[i]) for i in range(2)) for t in trace]
    ss = err[-1]
    settle = None
    for i in range(len(trace) - 1, -1, -1):
        if err[i] > band:
            settle = trace[min(i + 1, len(trace) - 1)][0]
            break
    settle = 0.0 if settle is None else settle
    e0 = err[0] if err[0] > 1e-12 else 1e-12
    over = max(0.0, max(err) / e0 - 1.0) if err else 0.0
    return {'settle_s': round(settle, 4), 'steady_err': round(ss, 5),
            'peak_ratio': round(max(err) / e0, 4), 'overshoot_frac': round(over, 4)}
