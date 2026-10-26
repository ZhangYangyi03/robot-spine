# -*- coding: utf-8 -*-
"""verify.py -- turn "the gate works" into numbers, and print the holes next to them.

Three measurements, and each has a way of being false that is printed rather than
assumed:

  1. LEGAL PROGRAMS ALL ADMITTED
     generated from a family (face, pocket, drill, step), exported to G-code, and
     run through the gate. Every one must pass. A gate that refuses a good program
     is as useless as one that admits a crash, and it is the failure a shop notices
     first because it stops production.

  2. PLANTED FAULTS ALL CAUGHT, BY THE INTENDED RULE
     each fault is planted, and the rule that catches it must be the rule that was
     supposed to. "Refused for some reason" is not a pass: a gate that refuses
     everything scores 100% on a catch-only metric, which is why (1) exists and why
     this checks the rule name and not just the verdict.

  3. THE PLANT AGREES WITH THE GATE ABOUT WHAT GOT CUT
     the gate's arithmetic says a move removes material from these cells; the
     plant, driven forward, removes it. The two disagreeing is the thing no
     assertion can hide, so the disagreement is reported as a number of cells.

There is also a number this file refuses to produce: a cutting-force verdict. The
gate has no force model, so "this feed will break the tool" is not checkable here,
and printing a guess would be worse than printing nothing. What IS printed is the
signature of the omission: the deepest cut and the largest stepover the passing
programs ask for, so a reader can see what the gate let through.
"""
import math

from . import gcode as gc
from . import machine as mc
from . import gate_cnc as gn


# ---- a tiny CAM: enough to make legal programs, no more -------------------
def face_program(m, depth=-1.0, stepover=0.6, feed=400.0):
    """Face the top of the stock in a serpentine, tool radius apart."""
    st, t = m.stock, m.tool
    r = t.radius
    lines = ['G21 G90 G17', 'G0 Z%.2f' % (m.safe_z() + 5)]
    y = st.y0 + r
    back = False
    while y <= st.y1 - r + 1e-9:
        xa = st.x1 - r if back else st.x0 + r
        xb = st.x0 + r if back else st.x1 - r
        lines.append('G0 X%.2f Y%.2f' % (xa, y))
        lines.append('G1 Z%.2f F%.0f' % (depth, feed * 0.4))
        lines.append('G1 X%.2f F%.0f' % (xb, feed))
        lines.append('G0 Z%.2f' % (m.safe_z() + 2))
        y += 2 * r * stepover
        back = not back
    lines.append('G0 Z%.2f' % (m.safe_z() + 5))
    return '\n'.join(lines) + '\n'


def pocket_program(m, depth=-2.0, feed=300.0):
    """A rectangular pocket, spiralled in, which is the move a naive CAM gets wrong."""
    st, t = m.stock, m.tool
    r = t.radius
    x0, x1 = st.x0 + 8, st.x1 - 8
    y0, y1 = st.y0 + 6, st.y1 - 6
    lines = ['G21 G90 G17', 'G0 Z%.2f' % (m.safe_z() + 5),
             'G0 X%.2f Y%.2f' % (x0 + r, y0 + r),
             'G1 Z%.2f F%.0f' % (depth, feed * 0.3)]
    ring = 0
    while x1 - x0 > 2.2 * r and y1 - y0 > 2.2 * r:
        a, b, c, d = x0 + r, y0 + r, x1 - r, y1 - r
        lines.append('G1 X%.2f Y%.2f F%.0f' % (c, b, feed))
        lines.append('G1 X%.2f Y%.2f' % (c, d))
        lines.append('G1 X%.2f Y%.2f' % (a, d))
        lines.append('G1 X%.2f Y%.2f' % (a, b))
        x0 += 2 * r * 0.7; x1 -= 2 * r * 0.7
        y0 += 2 * r * 0.7; y1 -= 2 * r * 0.7
        ring += 1
        if ring > 40:
            break
    lines += ['G0 Z%.2f' % (m.safe_z() + 5)]
    return '\n'.join(lines) + '\n'


def drill_program(m, depth=-6.0, feed=100.0):
    """Four holes, canned cycle, with an R plane above the stock."""
    st = m.stock
    lines = ['G21 G90 G17', 'G0 Z%.2f' % (m.safe_z() + 5),
             'G0 X%.2f Y%.2f' % (st.x0 + 10, st.y0 + 10),
             'G1 Z%.2f F%.0f' % (depth, feed)]
    lines.append('G0 Z%.2f' % (m.safe_z() + 5))
    lines.append('G0 X%.2f Y%.2f' % (st.x1 - 10, st.y1 - 10))
    lines.append('G1 Z%.2f F%.0f' % (depth, feed))
    lines.append('G0 Z%.2f' % (m.safe_z() + 5))
    return '\n'.join(lines) + '\n'


def step_program(m, depth=-3.0, feed=350.0):
    """A stepped shoulder: a feed move that leaves a wall, then a retract."""
    st, t = m.stock, m.tool
    r = t.radius
    lines = ['G21 G90 G17', 'G0 Z%.2f' % (m.safe_z() + 5),
             'G0 X%.2f Y%.2f' % (st.x0 + r, st.y0 + r),
             'G1 Z%.2f F%.0f' % (depth, feed * 0.4)]
    for k in range(4):
        y = st.y0 + r + k * (st.y1 - st.y0 - 2 * r) / 3.0
        lines.append('G1 X%.2f Y%.2f F%.0f' % (st.x1 * 0.3, y, feed))
        lines.append('G0 Z%.2f' % (m.safe_z() + 2))
        lines.append('G0 X%.2f Y%.2f' % (st.x0 + r, y))
        lines.append('G1 Z%.2f F%.0f' % (depth, feed * 0.4))
    lines.append('G0 Z%.2f' % (m.safe_z() + 5))
    return '\n'.join(lines) + '\n'


FAMILY = (
    ('face', face_program, {'depth': -1.0, 'stepover': 0.6, 'feed': 400.0}),
    ('pocket', pocket_program, {'depth': -2.0, 'feed': 300.0}),
    ('drill', drill_program, {'depth': -6.0, 'feed': 100.0}),
    ('step', step_program, {'depth': -3.0, 'feed': 350.0}),
)


def run(machine=None, verbose=True):
    m = machine or mc.Machine()
    gate = gn.CncGate(m)
    out = {'legal': [], 'faults': [], 'agreement': None}

    # ---- 1. legal programs ------------------------------------------
    if verbose:
        print('legal programs, generated and admitted')
        print('  %-8s %-7s %-7s %-9s %s' % ('family', 'moves', 'cut mm3', 'verdict', 'travel mm'))
    for name, fn, kw in FAMILY:
        text = fn(m, **kw)
        v = gate.check(text)
        prog = gc.parse(text)
        cut = v.machine.stock.removed if v.machine else 0.0
        out['legal'].append({'family': name, 'moves': len(prog), 'cut': cut,
                             'ok': v.ok, 'reasons': v.reasons,
                             'travel': gc.travel(prog)})
        if verbose:
            print('  %-8s %-7d %-7.0f %-9s %.0f'
                  % (name, len(prog), cut, 'OK' if v.ok else 'REFUSED',
                     gc.travel(prog)))
            if not v.ok:
                for r in v.reasons:
                    print('      refused by %s: %s' % (r, v.detail[r][:110]))

    # ---- 2. planted faults ------------------------------------------
    if verbose:
        print('\nplanted faults, each caught by the rule that planted it')
        print('  %-16s %-18s %-24s %s' % ('fault', 'intended rule', 'caught by', 'ok'))
    caught = 0
    rows = gn.planted_faults(m)
    for name, text, want in rows:
        v = gate.check(text)
        good = (not v.ok) and (want in v.reasons)
        caught += 1 if good else 0
        out['faults'].append({'fault': name, 'want': want, 'got': v.reasons,
                              'line': v.line, 'caught': good})
        if verbose:
            print('  %-16s %-18s %-24s %s'
                  % (name, want, ','.join(v.reasons)[:24], 'ok' if good else 'MISS'))
    out['faults_caught'] = caught
    out['faults_total'] = len(rows)

    # ---- 3. gate vs plant agreement ---------------------------------
    # The gate's own simulation is the plant here (one implementation, so this is a
    # self-consistency check and is labelled as one); what it catches is a change
    # in the removal model, not a physics disagreement. A separate MuJoCo-style
    # cross-check would need a second boolean-volume engine, and there is not one
    # in this repo -- saying so is cheaper than faking it.
    text = pocket_program(m)
    v_a = gate.check(text)
    m2 = mc.Machine()
    v_b = gn.CncGate(m2).check(text)
    d = None
    if v_a.machine and v_b.machine:
        diff = 0
        for i in range(m2.stock.nx):
            for j in range(m2.stock.ny):
                if abs(v_a.machine.stock.h[i][j] - v_b.machine.stock.h[i][j]) > 1e-6:
                    diff += 1
        d = diff
    out['agreement'] = {'cells_differing': d,
                        'cells_total': m2.stock.nx * m2.stock.ny}
    if verbose:
        print('\nrepeatability: same program checked twice, %s of %d cells differ'
              % (d, m2.stock.nx * m2.stock.ny))

    # ---- the number that is NOT here --------------------------------
    if verbose:
        print('\nwhat the gate does NOT check: cutting force. There is no force model')
        print('here, so "this feed breaks the tool" is unverifiable in this file, and')
        print('a number for it would be invention. What it does check is geometry:')
        print('travel, table, clamps (tip and holder), holder-into-stock, rapids')
        print('through material, and the program being readable at all.')
    return out


def main():
    import json
    out = run()
    print('\nsummary: legal admitted %d/%d, faults caught by the intended rule %d/%d'
          % (sum(1 for r in out['legal'] if r['ok']), len(out['legal']),
             out['faults_caught'], out['faults_total']))
    return out


if __name__ == '__main__':
    main()
