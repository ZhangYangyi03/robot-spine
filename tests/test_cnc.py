# -*- coding: utf-8 -*-
"""test_cnc.py -- the CNC gate's claims, as tests that fail when a claim stops being true.

Each test names the failure it exists to catch. The ones worth reading twice:

  test_refused_program_cuts_nothing   the gate said no and its own plant cut anyway.
                                      This was a real bug found while wiring the
                                      package up: a refused program reported 1408 mm3
                                      removed and the repair loop then reported the
                                      repaired program as having cut 704 when it had
                                      been credited with work a rejection should have
                                      prevented. Both directions are asserted below.

  test_rapid_lift_is_not_a_crash      every program in the world ends by lifting out
                                      of the part with G0. A gate that calls that
                                      "rapid through material" refuses every real
                                      program, and would have been wrong 100% of the
                                      time on the most common line in the file.

  test_arc_is_refused_not_sampled     the temptation is to sample an arc at a few
                                      points and call it checked. That lets a gouge
                                      through between samples, so it must refuse.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cnc import gcode as gc
from cnc import machine as mc
from cnc import gate_cnc as gn
from cnc import student as sd
from cnc import repair as rp
from cnc import verify as vf


def test_parser_reaches_millimetres_absolute():
    p = gc.parse('G20 G90\nG0 X1 Y2\nG21\nG0 X10\n')
    assert abs(p.moves[0].x - 25.4) < 1e-9, 'G20 must scale inches to mm'
    assert abs(p.moves[1].x - 10.0) < 1e-9, 'G21 must return to mm'


def test_parser_is_incremental_when_asked():
    p = gc.parse('G91\nG0 X5 Y5\nG0 X5\n')
    assert abs(p.moves[0].x - 5) < 1e-9
    assert abs(p.moves[1].x - 10) < 1e-9, 'G91 must accumulate'


def test_bare_coordinate_line_continues_modal_motion():
    p = gc.parse('G90\nG1 X10 F100\nY10\n')
    assert len(p.moves) == 2
    assert p.moves[1].kind == 'feed', 'a bare Y continues the modal motion'


def test_arc_is_refused_not_sampled():
    try:
        gc.parse('G2 X10 Y0 I5 J0\n')
    except gc.GcodeError:
        return
    raise AssertionError('an arc must be refused, not silently sampled')


def test_legal_family_all_admitted():
    out = vf.run(verbose=False)
    bad = [r for r in out['legal'] if not r['ok']]
    assert not bad, 'the gate refused a program from its own legal family: %r' % bad


def test_every_planted_fault_is_caught_by_its_intended_rule():
    m = mc.Machine()
    g = gn.CncGate(m)
    for name, text, want in gn.planted_faults(m):
        v = g.check(text)
        assert not v.ok, '%s was admitted' % name
        assert want in v.reasons, ('%s was caught by %r, not by %r -- "refused for '
                                  'some reason" is not the claim'
                                   % (name, v.reasons, want))


def test_refused_program_cuts_nothing():
    m = mc.Machine()
    g = gn.CncGate(m)
    prog = sd.greedy(m)
    v = g.check(prog)
    assert not v.ok
    assert v.machine.stock.removed == 0.0, (
        'a refused program must not report material removed; it reported %r'
        % v.machine.stock.removed)
    assert v.partial is not None and v.partial.stock.removed > 0, (
        'the diagnostic partial stock should still show how far it got')


def test_admitted_program_does_cut():
    m = mc.Machine()
    v = gn.CncGate(m).check(sd.careful(m))
    assert v.ok
    assert v.machine.stock.removed > 100.0, 'an admitted facing program must remove material'


def test_rapid_lift_is_not_a_crash():
    m = mc.Machine()
    sz = m.safe_z() + 2
    prog = ('G21 G90\nG0 Z%.1f\nG0 X0 Y0\nG1 Z-2 F100\nG0 Z%.1f\n' % (sz, sz))
    v = gn.CncGate(m).check(prog)
    assert v.ok, 'lifting straight out of the part at the end is legal: %r' % (v.reasons,)


def test_holder_is_checked_not_just_the_tip():
    m = mc.Machine()
    sz = m.safe_z() + 2
    deep = m.stock.top_z - (m.tool.stickout + 2.0)
    prog = 'G21 G90\nG0 Z%.1f\nG0 X0 Y0\nG1 Z%.1f F100\n' % (sz, deep)
    v = gn.CncGate(m).check(prog)
    assert 'holder_crash' in v.reasons, (
        'the holder entering the stock must be its own refusal: %r' % (v.reasons,))


def test_clamp_holder_crash_is_caught():
    m = mc.Machine()
    cx0, cy0, cx1, cy1, cz = m.stock.clamps[0]
    cx, cy = cx0 + 2, cy0 + 2
    prog = 'G21 G90\nG0 Z%.1f\nG0 X%.1f Y%.1f\nG0 Z%.1f\n' % (m.safe_z() + 2, cx, cy, cz - 3)
    v = gn.CncGate(m).check(prog)
    assert 'clamp_crash' in v.reasons


def test_repair_loop_fixes_the_reorderable_error_and_says_no_to_the_geometric_one():
    m = mc.Machine()
    ok = rp.loop(m, sd.greedy(m))
    assert ok['ok'] and ok['rounds'] <= 2, 'a rapid-through-material is fixable by reordering'
    bad = rp.loop(m, sd.shortsighted(m))
    assert not bad['ok'], 'a cut deeper than the tool reaches cannot be fixed by reordering, ' \
                          'and the loop must not pretend otherwise'
    assert 'unrepairable' in bad['why']


def test_repeatability_of_the_removal_model():
    out = vf.run(verbose=False)
    assert out['agreement']['cells_differing'] == 0
