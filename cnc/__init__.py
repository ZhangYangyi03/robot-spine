"""cnc -- the same gate, pointed at a machine tool instead of a walking body.

spine-agent runs three layers: a proposer, a physics check that can refuse, and a
plant that moves. The minecraft body and this package are two plants under one
gate; what they share is the only thing that matters for the claim -- the proposer
is allowed to be wrong, and the check is arithmetic that does not care what the
proposer wanted.

    gcode.py       a parser strict enough to be wrong about, and its dialect
    machine.py     3-axis kinematics, stock removal, and the collision scene
    gate_cnc.py    the gate: refuses a move before the machine makes it
    verify.py      planting faults, to say what the gate caught and by which rule

Nothing here is a CAM system. It does not decide what to cut; it decides whether
what was decided can be cut without hitting anything, which is the part where the
gates in this repo already have a record.
"""
__all__ = ["gcode", "machine", "gate_cnc", "verify"]
