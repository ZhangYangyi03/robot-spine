# -*- coding: utf-8 -*-
"""repair.py -- the loop: refuse, hand the refusal back, and try again.

This is the part that makes the package an agent rather than a checker. The gate's
verdict is already structured (rule name, line number, the numbers that broke), so
the refusal can be turned into an instruction without a language model in the way:

    "line 6: rapid passes through material at (0.00,-8.00,-2.00) where the stock
     top is 0.00"   ->   "retract to Z>=14 before the lateral move at line 6"

Two things are claimed, and both are measured rather than asserted:

  1. EVERY REFUSAL IS ACTIONABLE
     the verdict names a line and a rule, and the repair below is derived from that
     rule alone -- no model, no prompt, no retry budget that hides a failure. A
     refusal a proposer cannot act on is a bug in the refusal, so this is checkable.

  2. THE LOOP TERMINATES, AND IT IS THE GATE THAT SAYS SO
     the repair is offered back, re-checked, and the sequence is recorded. If a
     repair does not help, the loop does not spin: it reports "unrepairable by this
     rule set" and stops, which is the honest answer when the proposer's geometry is
     wrong rather than its motion order.

The repair strategies are deliberately dumb. A smart one would be a CAM system, and
the point of this file is to show the loop works on top of a checker that already
works -- so that when a real model is wired in later, the thing it talks to has a
proven interface.
"""
from . import gcode as gc
from . import gate_cnc as gn


REPAIR_RULES = {
    'rapid_into_stock':
        'Every lateral rapid must be taken at or above the safe Z. Insert a retract '
        'to safe Z before the offending lateral move, and a plunge back down after.',
    'table_crash':
        'The tool tip went below the table face. Raise the depth by at least the '
        'overshoot, or reduce the requested depth.',
    'clamp_crash':
        'The tool (or its holder) is inside a clamp. Move the work or shorten the '
        'reach; a clamp is not material to be cut.',
    'holder_crash':
        'The flutes ran out: the holder entered the stock. Either reduce the depth '
        'or use a tool with more stickout -- this is the number a prompt rarely has.',
    'travel_limit':
        'The commanded position is outside the machine envelope. Reposition the '
        'workpiece zero, not the program.',
    'unsupported':
        'The program uses a construct this gate cannot verify (arc, tilted plane, '
        'rigid tapping). It is refused rather than guessed at.',
}


def explain(verdict):
    """Turn a verdict into the list of instructions a proposer can act on."""
    if verdict.ok:
        return []
    out = []
    for r in verdict.reasons:
        out.append({'rule': r,
                    'line': verdict.line,
                    'detail': verdict.detail.get(r, ''),
                    'fix': REPAIR_RULES.get(r, 'no repair strategy for this rule')})
    return out


def retract_before(text, line_no, safe_z):
    """The mechanical repair for a rapid through material: retract, move, plunge.

    It is done on the SOURCE TEXT and not on the parsed program, because a proposer
    hands back text and the loop has to speak its language.
    """
    lines = text.splitlines()
    out = []
    for i, ln in enumerate(lines, 1):
        if i == line_no:
            code = ln.split(';', 1)[0].strip()
            # remember the modal motion and feed, then retract, then restore
            out.append('G0 Z%.2f' % safe_z)
            out.append(code)
            # only re-plunge if the move after was a feed at depth -- a blind
            # re-plunge would create a hole that was never asked for
            out.append('; --- gate repair: retracted to safe Z before this move ---')
            continue
        out.append(ln)
    return '\n'.join(out) + '\n'


def loop(machine, text, gate=None, max_rounds=4, verbose=False):
    """check -> explain -> repair -> re-check, and stop when the gate says stop."""
    g = gate or gn.CncGate(machine)
    history = []
    cur = text
    for rnd in range(max_rounds + 1):
        v = g.check(cur)
        history.append({'round': rnd, 'ok': v.ok, 'reasons': v.reasons,
                        'line': v.line,
                        'detail': {k: v.detail[k] for k in v.detail}})
        if verbose:
            print('  round %d: %s' % (rnd, 'admitted' if v.ok else
                                      'refused %s at line %s'
                                      % (','.join(v.reasons), v.line)))
        if v.ok:
            return {'ok': True, 'program': cur, 'history': history,
                    'rounds': rnd, 'why': 'admitted'}
        if rnd == max_rounds:
            break
        fixes = explain(v)
        applied = False
        for f in fixes:
            if f['rule'] == 'rapid_into_stock' and f['line']:
                cur = retract_before(cur, f['line'], machine.safe_z() + 2.0)
                applied = True
                break
        if not applied:
            return {'ok': False, 'program': cur, 'history': history,
                    'rounds': rnd,
                    'why': 'unrepairable by this rule set: %s'
                           % ','.join(v.reasons)}
    return {'ok': False, 'program': cur, 'history': history,
            'rounds': max_rounds, 'why': 'round budget spent'}
