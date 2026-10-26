# -*- coding: utf-8 -*-
"""student.py -- the layer under test: a program generater that is allowed to be wrong.

The claim spine-agent makes about its gate is not "the gate is correct", it is "a
WRONG proposer is survivable". In the minecraft half the wrong proposer is a 121k
policy trained by imitation on 2-second CPU; here it is the thing an AI asked to
write G-code actually does, and the interesting question is which mistakes it makes.

So this file is not a CAM system and does not pretend to be. It has three named
policies, and the names are the failures:

    careful     every rapid goes to safe Z first, feeds stay under the material
                rate, tool is out of its holder far enough. The control: a program
                the gate must admit.
    greedy      rapids straight from where it is to where it is going. This is what
                a language model writes when it is asked to "move the cutter to the
                next hole" and nobody told it that a rapid is not a move.
    shortsighted  clears the pocket at depth without retracting between passes, and
                ignores stickout when it reaches the bottom of a deep pocket. This
                is the error a human makes at 5 pm and an AI makes every time,
                because the number it needs (how far the tool sticks out of the
                holder) is not in the prompt.

Both of these are DETERMINISTIC and written out, so the failures are the ones
listed here rather than whatever a sampled model happened to do. Numbers about a
model's error rate belong in a bench that samples it; numbers about a GATE's
ability to stop a class of error belong here, and they should not move when the
model is reseeded.
"""
from . import machine as mc


def careful(m, depth=-2.0, feed=400.0):
    st, t = m.stock, m.tool
    r = t.radius
    sz = m.safe_z() + 2.0
    lines = ['G21 G90 G17', 'G0 Z%.2f' % sz]
    y = st.y0 + r
    back = False
    while y <= st.y1 - r + 1e-9:
        xa = st.x1 - r if back else st.x0 + r
        xb = st.x0 + r if back else st.x1 - r
        lines.append('G0 X%.2f Y%.2f' % (xa, y))
        lines.append('G1 Z%.2f F%.0f' % (depth, feed * 0.35))
        lines.append('G1 X%.2f F%.0f' % (xb, feed))
        lines.append('G0 Z%.2f' % sz)
        y += 2 * r * 0.55
        back = not back
    lines.append('G0 Z%.2f' % sz)
    return '\n'.join(lines) + '\n'


def greedy(m, depth=-2.0, feed=400.0):
    """Same cut, but every reposition is a rapid AT DEPTH.

    A rapid at depth between two passes of the same pocket is not always a crash --
    if the material between them is already gone it is legal, and a gate that
    refused it would refuse real CAM output. What makes this a crash is that the
    straight line between the two passes crosses uncut material, which is exactly
    what the gate's path sample is for.
    """
    st, t = m.stock, m.tool
    r = t.radius
    sz = m.safe_z() + 2.0
    lines = ['G21 G90 G17', 'G0 Z%.2f' % sz,
             'G0 X%.2f Y%.2f' % (st.x0 + r, st.y0 + r),
             'G1 Z%.2f F%.0f' % (depth, feed * 0.35),
             'G1 X%.2f F%.0f' % (st.x1 - r, feed)]
    # ... and now straight back across at depth, through the ridge it just left
    lines.append('G0 X%.2f Y%.2f' % (st.x0 + r, st.y0 + r + 3 * r))
    lines.append('G1 X%.2f F%.0f' % (st.x1 - r, feed))
    lines.append('G0 Z%.2f' % sz)
    return '\n'.join(lines) + '\n'


def shortsighted(m, depth=-2.0, feed=400.0, pocket_depth=None):
    """Ends with a deep plunge that runs the flutes out."""
    st, t = m.stock, m.tool
    r = t.radius
    sz = m.safe_z() + 2.0
    if pocket_depth is None:
        # deeper than the tool sticks out minus the stock it is already inside
        pocket_depth = st.top_z - (t.stickout + 1.5)
    lines = ['G21 G90 G17', 'G0 Z%.2f' % sz,
             'G0 X%.2f Y%.2f' % (st.x0 + 10, st.y0 + 10),
             'G1 Z%.2f F%.0f' % (depth, feed * 0.35),
             'G1 Z%.2f F%.0f' % (pocket_depth, feed * 0.15),
             'G0 Z%.2f' % sz]
    return '\n'.join(lines) + '\n'


POLICIES = (('careful', careful), ('greedy', greedy), ('shortsighted', shortsighted))
