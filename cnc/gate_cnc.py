# -*- coding: utf-8 -*-
"""gate_cnc.py -- admit a G-code program, or say by which rule it would crash.

This is autoforge's gate shape, and robot-spine's, pointed at a machine tool. The
gate is arithmetic over the parsed program and the machine's geometry; it does not
run the program, and that is deliberate:

    a plant that runs the program can only tell you it CUT badly.
    a gate that checks the program can tell you WHICH LINE was going to crash it,
    before the spindle turns, which is the only version a shop will trust.

Six rules, each naming itself in the verdict:

    travel_limit    a move outside the machine's travel
    table_crash     the tool tip below the table's top face
    clamp_crash     the tool tip or the HOLDER inside a clamp
    holder_crash    the tool tip inside the stock while the holder is inside the
                    stock too -- i.e. the flutes ran out, which is how a tool snaps
    rapid_into_stock  a G0 that moves THROUGH material. A rapid is allowed to go
                    down to material it is ABOUT to cut only if the next move is a
                    feed at that level; the rule is about the rapid's own path
    unsupported     the parser refused something (arc, plane, tapping). Carried
                    through so a program cannot be "safe" by being unreadable

The one rule that is NOT here, and why: a feed move that removes more than the tool
can take. That is a cutting-force question -- chain load, radial engagement, the
material -- and this gate has no force model. It is measured as a separate number
(a signature in verify.py) so the file does not claim a physics it does not have.
"""
import math

from . import gcode as gc


class Verdict:
    """Same shape as spine.gate.Verdict, so one habit reads both."""
    __slots__ = ('ok', 'reasons', 'detail', 'line', 'worst', 'machine', 'partial')

    def __init__(self, ok, reasons, detail, line=None, worst=None):
        self.ok, self.reasons, self.detail = ok, reasons, detail
        self.line, self.worst = line, worst
        self.machine = None      # admitted: the stock as cut. refused: uncut.
        self.partial = None      # refused only: how far it got before the stop

    def __bool__(self):
        return self.ok

    def __repr__(self):
        if self.ok:
            return '<Verdict OK>'
        return '<Verdict REFUSED %s line %s>' % (','.join(self.reasons), self.line)


class CncGate:
    RULES = ('travel_limit', 'table_crash', 'clamp_crash', 'holder_crash',
             'rapid_into_stock', 'unsupported')

    def __init__(self, machine, rules=None, simulate=True):
        self.m = machine
        self.rules = set(rules or self.RULES)
        self.simulate = simulate

    # -- helpers -----------------------------------------------------------
    def _in_clamp(self, x, y, z):
        """Which clamp contains (x, y, z), or None. NaN sentinel: -1."""
        for k, (cx0, cy0, cx1, cy1, cz1) in enumerate(self.m.stock.clamps):
            if cx0 - 1e-9 <= x <= cx1 + 1e-9 and cy0 - 1e-9 <= y <= cy1 + 1e-9 \
                    and z <= cz1 + 1e-9:
                return k
        return None

    def _stock_columns(self, m, x, y, r):
        """(min height, max height) of the stock columns under a tool of radius r."""
        lo, hi = None, None
        st = m.stock
        n = max(1, int(math.ceil(2 * r / min(st.dx, st.dy))))
        for a in range(n + 1):
            for b in range(n + 1):
                px = x - r + 2 * r * a / n
                py = y - r + 2 * r * b / n
                h = st.height_at(px, py)
                if h is None:
                    continue
                lo = h if lo is None else min(lo, h)
                hi = h if hi is None else max(hi, h)
        return lo, hi

    # -- the check ---------------------------------------------------------
    def check(self, program_or_text):
        if isinstance(program_or_text, str):
            try:
                prog = gc.parse(program_or_text)
            except gc.GcodeError as e:
                return Verdict(False, ['unsupported'], {'unsupported': str(e)})
        else:
            prog = program_or_text

        # A fresh machine each time: a gate that mutates the caller's stock would
        # make the second check of the same program disagree with the first.
        m = self.m.clone_fresh()
        reasons, detail, worst = {}, {}, None
        first_bad_line = None

        tool = m.tool
        prev = None
        for mv in prog.moves:
            rr = {}

            # travel limit -- every axis, on the tip position
            if 'travel_limit' in self.rules:
                if not (m.x_min <= mv.x <= m.x_max and m.y_min <= mv.y <= m.y_max
                        and m.z_min <= mv.z <= m.z_max):
                    rr['travel_limit'] = ('line %d: (%.2f,%.2f,%.2f) is outside travel '
                                          'X[%g,%g] Y[%g,%g] Z[%g,%g]'
                                          % (mv.line, mv.x, mv.y, mv.z, m.x_min,
                                             m.x_max, m.y_min, m.y_max,
                                             m.z_min, m.z_max))

            # table -- the tip below the table's face, anywhere, any move
            if 'table_crash' in self.rules and mv.z < m.stock.table_z - 1e-9:
                rr['table_crash'] = ('line %d: Z %.3f is below the table face at %.3f'
                                     % (mv.line, mv.z, m.stock.table_z))

            # clamps -- the tip, and the holder, which is the half everybody forgets
            if 'clamp_crash' in self.rules:
                k = self._in_clamp(mv.x, mv.y, mv.z)
                if k is not None:
                    rr['clamp_crash'] = ('line %d: tool tip (%.1f,%.1f,%.1f) is inside '
                                         'clamp %d' % (mv.line, mv.x, mv.y, mv.z, k))
                else:
                    hz = tool.holder_bottom(mv.z)
                    k2 = self._in_clamp(mv.x, mv.y, hz)
                    if k2 is not None:
                        rr['clamp_crash'] = (
                            'line %d: HOLDER bottom Z=%.2f is inside clamp %d at '
                            '(%.1f,%.1f) -- the tool is out %.1f mm, less than the '
                            'clamp height' % (mv.line, hz, k2, mv.x, mv.y, tool.stickout))

            # holder into stock -- flutes ran out
            if 'holder_crash' in self.rules:
                lo, hi = self._stock_columns(m, mv.x, mv.y, tool.radius)
                if hi is not None and mv.z < hi - 1e-9:
                    if tool.holder_bottom(mv.z) < hi - 1e-9:
                        rr['holder_crash'] = (
                            'line %d: holder bottom Z=%.2f is inside stock (top %.2f) '
                            '-- stickout %.1f mm with %.1f mm of stock above the tip'
                            % (mv.line, tool.holder_bottom(mv.z), hi,
                               tool.stickout, hi - mv.z))

            # rapid through material
            if 'rapid_into_stock' in self.rules and mv.kind == 'rapid':
                if prev is not None:
                    # sample the rapid along its path and ask whether any sampled
                    # point is inside material. A rapid that starts above and ends
                    # above but dips through a boss is the case a start/end test
                    # misses, and it is a real crash.
                    L = math.sqrt((mv.x - prev[0]) ** 2 + (mv.y - prev[1]) ** 2
                                  + (mv.z - prev[2]) ** 2)
                    n = max(1, int(math.ceil(L / max(1e-6, min(m.stock.dx, m.stock.dy)))))
                    # A rapid is checked only where it moves LATERALLY. A pure
                    # plunge or lift along Z cannot gouge: the tool is a vertical
                    # cylinder, so moving it straight up or down leaves the
                    # material it is not touching alone -- and every program in
                    # the world ends with a lift out of the part (G0 Z20 from
                    # z=-2), which a naive path test calls "through material" and
                    # refuses. Measured: a legal facing program was refused at its
                    # final retract line until this split was made.
                    lateral = math.hypot(mv.x - prev[0], mv.y - prev[1])
                    if lateral < 1e-9:
                        pass
                    else:
                        hit = None
                        for q in range(1, n + 1):
                            t = q / n
                            x = prev[0] + (mv.x - prev[0]) * t
                            y = prev[1] + (mv.y - prev[1]) * t
                            z = prev[2] + (mv.z - prev[2]) * t
                            lo, hi = self._stock_columns(m, x, y, tool.radius)
                            if hi is not None and z < hi - 1e-9:
                                hit = (x, y, z, hi)
                                break
                        if hit is not None:
                            rr['rapid_into_stock'] = (
                                'line %d: rapid passes through material at '
                                '(%.2f,%.2f,%.2f) where the stock top is %.2f'
                                % (mv.line, hit[0], hit[1], hit[2], hit[3]))

            if rr and first_bad_line is None:
                # A REFUSED PROGRAM CUTS NOTHING. The simulation below only ever
                # ran for admitted moves, so the stock advanced up to the first bad
                # line and stopped -- which meant a refused program still reported
                # 1408 mm3 removed, and a caller comparing `removed` against a
                # target would have been told a program the gate rejected had done
                # most of the work. MEASURED, and it was my own bug: the gate said
                # no and the plant cut anyway, which is the one failure this whole
                # pattern exists to prevent.
                first_bad_line = mv.line
            reasons.update(rr)
            if rr and worst is None:
                worst = (mv.line, sorted(rr)[0])
            detail.update(rr)

            # the plant, when asked: only ever for admitted motion, and only to
            # keep the stock current so the NEXT move's checks see what is left.
            # `first_bad_line is None` is the important half: once the program has
            # been refused, nothing after it is simulated either, so a refused
            # program reports zero material removed. Without that, the gate refused
            # greedy's line 6 and its own plant had already cut 1408 mm3 -- a
            # rejected program that had done most of the work, which is the exact
            # failure this pattern exists to prevent.
            if self.simulate and not rr and first_bad_line is None:
                if mv.kind == 'feed':
                    a = prev if prev is not None else (mv.x, mv.y, mv.z)
                    m.stock.sweep(a, (mv.x, mv.y, mv.z), tool.radius,
                                  floor_z=m.z_min)
                elif mv.kind == 'drill':
                    cyc = mv.cycle or {}
                    r = cyc.get('r')
                    plunge_from = (mv.x, mv.y, r if r is not None else mv.z)
                    m.stock.sweep(plunge_from, (mv.x, mv.y, mv.z), tool.radius,
                                  floor_z=m.z_min)
            prev = (mv.x, mv.y, mv.z)

        ok = not reasons
        if not ok:
            # a refused program leaves the stock as it found it. The partial stock
            # is still useful for diagnosis, so it is exposed under a name that
            # cannot be mistaken for "what this program would cut".
            m_partial = m
            m = self.m.clone_fresh()
        v = Verdict(ok, sorted(reasons), detail, line=first_bad_line, worst=worst)
        v.machine = m            # uncut if refused, cut if admitted
        v.partial = m_partial if not ok else None
        return v


def planted_faults(machine):
    """Programs that a shop would call a crash, and the rule each must trip.

    This is the same idea as spine/gate.py's fault injection, and it is here for
    the same reason: the claim "the gate refuses crashes" is worth nothing unless
    the crashes are planted rather than imagined, and unless each one is caught by
    the rule that was supposed to catch it rather than by some other rule.
    """
    m = machine
    st = m.stock
    sz = m.safe_z()
    cx = st.clamps[0][0] + 2.0
    cy = st.clamps[0][1] + 2.0
    clamp_top = st.clamps[0][4]
    return [
        ('table', 'G21 G90\nG0 Z%.1f\nG1 Z%.1f F100\n' % (sz, st.table_z - 1.0),
         'table_crash'),
        ('clamp_tip', 'G21 G90\nG0 Z%.1f\nG0 X%.1f Y%.1f\nG1 Z%.1f F100\n'
         % (sz, cx, cy, clamp_top - 1.0), 'clamp_crash'),
        ('clamp_holder', 'G21 G90\nG0 Z%.1f\nG0 X%.1f Y%.1f\nG0 Z%.1f\n'
         % (sz, cx, cy, clamp_top - 3.0), 'clamp_crash'),
        # the flutes are shorter than the cut: the tip is inside the stock and the
        # holder is too. 22 mm below the top with 20 mm out of the holder means the
        # holder's bottom is 2 mm inside the material, and the table is 10 mm down,
        # so this is NOT a table crash -- it has to be caught as the holder.
        ('holder_in_stock', 'G21 G90\nG0 Z%.1f\nG0 X%.1f Y%.1f\nG1 Z%.1f F100\n'
         % (sz, 0.0, 0.0, st.top_z - (m.tool.stickout + 2.0)), 'holder_crash'),
        ('rapid_through', 'G21 G90\nG0 Z%.1f\nG0 X%.1f Y%.1f\nG1 Z%.1f F100\n'
         'G0 X%.1f Y%.1f\n'
         % (sz, st.x0 + 2, st.y0 + 2, st.top_z - 2.0, st.x1 - 2, st.y1 - 2),
         'rapid_into_stock'),
        ('travel', 'G21 G90\nG0 Z%.1f\nG0 X%.1f Y0\n' % (sz, m.x_max + 20.0),
         'travel_limit'),
        ('arc_unsupported', 'G21 G90\nG0 X0 Y0 Z5\nG2 X10 Y0 I5 J0 F100\n',
         'unsupported'),
    ]
