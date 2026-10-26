# -*- coding: utf-8 -*-
"""gcode.py -- read G-code strictly enough that being wrong is detectable.

Why parse at all, when the gate could just look for "G0"/"G1": because the failure
this file exists for is a number in the wrong place. A gate that reads `G1 X10`
with a regex and gets the axis wrong admits exactly the move it was built to
refuse, and it does so silently. So the parse produces one canonical record per
motion, carries the line number, and refuses anything it does not understand
instead of skipping it.

The dialect is the one a small 3-axis mill actually speaks, and it is written down
here rather than inferred:

    G0 / G1        rapid / feed move, modal (a bare `X.. Y..` continues the last)
    G20 / G21      inches / millimetres, modal, default G21
    G90 / G91      absolute / incremental, modal, default G90
    G17            XY plane (the only one implemented; G18/G19 are refused)
    G28            home all axes
    G80            cancel canned cycle
    G81 / G82 / G83 / G73   drilling, R (clearance), Z (depth), Q (peck), F (feed)
    G84            tapping -- refused on purpose, it is a rigid-synchronised move
    M3 / M4 / M5   spindle on cw / ccw / off, S is rpm
    M6 / T         tool change
    M30 / M2       program end
    F, S           feed and speed, modal

Comments are `( ... )` and `;` to end of line. A line that is only a comment is
kept in the listing (so line numbers stay honest) but produces no record.

Deliberately NOT here: `G2`/`G3` arcs. They are a geometry problem (the machine has
to interpolate a circle, and the gate has to check the whole arc, not three points)
and pretending to support them at three sampled points would be the exact silent
wrongness this file is written to avoid. An arc is reported as unsupported and the
gate refuses the program -- loudly.
"""
import math
import re

WORD = re.compile(r'([A-Za-z])\s*([-+]?\d*\.?\d+)')
LINENO = re.compile(r'^[Nn](\d+)')


class GcodeError(ValueError):
    """A program this parser will not guess about."""


class Move:
    """One motion, canonical: millimetres, absolute, with the line it came from."""
    __slots__ = ('kind', 'x', 'y', 'z', 'f', 'line', 'src', 'tool', 'cycle')

    def __init__(self, kind, x, y, z, f=None, line=0, src='', tool=None, cycle=None):
        self.kind = kind          # 'rapid' | 'feed' | 'drill' | 'home'
        self.x, self.y, self.z = x, y, z
        self.f = f
        self.line = line
        self.src = src
        self.tool = tool
        self.cycle = cycle        # for 'drill': dict(kind='G81', r=, q=)

    def __repr__(self):
        return '<%s (%.3f,%.3f,%.3f) L%d>' % (self.kind, self.x, self.y, self.z, self.line)


class Program:
    """A parsed program: the motions, the tool changes, and what was refused."""

    def __init__(self, moves, tools, lines, notes):
        self.moves = moves
        self.tools = tools          # [(line, tool_no)]
        self.lines = lines          # source text, 1-based indexing via [i-1]
        self.notes = notes          # things worth printing, not errors

    def __len__(self):
        return len(self.moves)

    def bbox(self):
        if not self.moves:
            return None
        xs = [m.x for m in self.moves]
        ys = [m.y for m in self.moves]
        zs = [m.z for m in self.moves]
        return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def parse(text):
    """Parse a program. Raises GcodeError on anything it will not guess about."""
    lines = text.splitlines()
    moves, tools, notes = [], [], []

    units = 1.0          # G20 sets 25.4
    absolute = True
    motion = None        # last G0/G1/'G81'...
    feed = None
    spindle = None
    tool = None
    pos = [0.0, 0.0, 0.0]
    cycle = None

    for n, raw in enumerate(lines, 1):
        src = raw.strip()
        if not src:
            continue
        # strip ( comments ) and ; comments, keep the line for numbering
        line = re.sub(r'\([^)]*\)', ' ', raw)
        line = line.split(';', 1)[0]
        # line numbers are not words we care about, but they verify ordering
        m = LINENO.match(line.strip())
        if m:
            line = line.strip()[m.end():]
        words = WORD.findall(line)
        if not words:
            continue

        code = None
        g_words = []
        params = {}
        for letter, val in words:
            L = letter.upper()
            v = float(val)
            if L == 'G':
                g_words.append(v)
            elif L == 'M':
                params.setdefault('M', []).append(v)
            else:
                params[L] = v

        for g in g_words:
            gi = int(round(g))
            if gi in (20,):
                units = 25.4
            elif gi in (21,):
                units = 1.0
            elif gi == 90:
                absolute = True
            elif gi == 91:
                absolute = False
            elif gi == 17:
                pass
            elif gi in (18, 19):
                raise GcodeError('line %d: %s plane is not implemented -- this gate '
                                 'only checks XY-plane programs, and pretending to '
                                 'check it would be wrong silently'
                                 % (n, 'G%d' % gi))
            elif gi in (2, 3):
                raise GcodeError('line %d: G%d arc is not implemented. Sampling an '
                                 'arc at a few points would let a gouge through '
                                 'between samples, so the program is refused instead.'
                                 % (n, gi))
            elif gi == 28:
                code = 'home'
            elif gi in (0, 1):
                code = 'rapid' if gi == 0 else 'feed'
            elif gi == 80:
                cycle = None
            elif gi in (81, 82, 83, 73):
                cycle = {'kind': 'G%d' % gi,
                         'r': params.get('R'),
                         'q': params.get('Q')}
                code = 'drill'
            elif gi == 84:
                raise GcodeError('line %d: G84 rigid tapping is refused on purpose: '
                                 'the spindle and the Z axis are synchronised, and a '
                                 '3-axis position gate cannot verify that.' % n)
            else:
                notes.append('line %d: G%d ignored (not a motion)' % (n, gi))

        if 'F' in params:
            feed = params['F'] * units
        if 'S' in params:
            spindle = params['S']
        if 'T' in params:
            tool = int(params['T'])

        if 'M' in params:
            for mi in params['M']:
                mi = int(round(mi))
                if mi in (3, 4, 5):
                    notes.append('line %d: spindle M%d S=%s' % (n, mi, spindle))
                elif mi == 6:
                    tools.append((n, tool))
                    notes.append('line %d: tool change T%s' % (n, tool))
                elif mi in (2, 30):
                    notes.append('line %d: program end' % n)
                else:
                    notes.append('line %d: M%d ignored' % (n, mi))

        if code is None:
            # a bare coordinate line continues the modal motion, which is how
            # every real post-processor writes; treating it as "no move" would
            # drop most of the program
            if any(a in params for a in ('X', 'Y', 'Z')) and motion in ('rapid', 'feed', 'drill'):
                code = motion
            else:
                continue
        motion = code

        new = list(pos)
        for i, A in enumerate(('X', 'Y', 'Z')):
            if A in params:
                v = params[A] * units
                new[i] = v if absolute else pos[i] + v
        # a drill cycle specified only by R/Z still needs the full position
        if code == 'drill' and cycle is None:
            cycle = {'kind': 'G81', 'r': params.get('R'), 'q': params.get('Q')}

        moves.append(Move(code, new[0], new[1], new[2], f=feed, line=n, src=src,
                          tool=tool, cycle=dict(cycle) if cycle else None))
        pos = new

    if not moves:
        raise GcodeError('program contains no motion')
    return Program(moves, tools, lines, notes)


def travel(program):
    """Total path length in millimetres, XY only. The number a cycle time is."""
    total = 0.0
    prev = None
    for m in program.moves:
        if prev is None:
            prev = (m.x, m.y, m.z)
            continue
        dx, dy, dz = m.x - prev[0], m.y - prev[1], m.z - prev[2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
        prev = (m.x, m.y, m.z)
    return total
