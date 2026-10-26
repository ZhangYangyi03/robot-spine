# -*- coding: utf-8 -*-
"""bench_cnc.py -- the CNC half, measured the way the walking half is measured.

Run:  python -m bench.bench_cnc

Prints three things and refuses to print a fourth:

  1. legal programs admitted        (a gate that refuses good work is useless)
  2. planted faults caught, BY THE RULE THAT PLANTED THEM
  3. the student policies: one that must pass and two that must be stopped, with
     the line number of the move that would have ruined the part or the machine

  4. NOT a cutting-force verdict. The gate has no force model. Printing a number
     for "this feed breaks the tool" would be inventing the physics this package
     says it does not have, and the omission is more useful printed than hidden.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cnc import verify as vf
from cnc import student as sd
from cnc import machine as mc
from cnc import gate_cnc as gn


def main():
    out = vf.run(verbose=True)

    print('\nstudent policies: one that must pass, two that must be stopped')
    m = mc.Machine()
    gate = gn.CncGate(m)
    rows = []
    for name, fn in sd.POLICIES:
        text = fn(m)
        v = gate.check(text)
        rows.append({'policy': name, 'admitted': v.ok, 'line': v.line,
                     'reasons': v.reasons})
        print('  %-13s %-9s %s'
              % (name, 'admitted' if v.ok else 'REFUSED',
                 ('' if v.ok else 'line %s: %s' % (v.line, ','.join(v.reasons)))))

    payload = {
        'legal': out['legal'],
        'faults': out['faults'],
        'faults_caught': out['faults_caught'],
        'faults_total': out['faults_total'],
        'repeatability': out['agreement'],
        'student_policies': rows,
    }
    dest = os.path.join(HERE, 'results_cnc.json')
    with open(dest, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=1)
    print('\nwrote', dest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
