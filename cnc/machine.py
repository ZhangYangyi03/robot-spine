# -*- coding: utf-8 -*-
"""machine.py -- a 3-axis mill, its stock, its fixture, and what a move removes.

The plant for the CNC half of spine-agent, in the same spirit as embodied.py: the
geometry is written out rather than imported, so the gate and the plant cannot
disagree because a library disagreed with itself.

Coordinates, stated once because every number below depends on it:

    Z = 0 is the TOP OF THE STOCK, and cutting goes negative. This is what every
    mill program in the world assumes, and it is the reason a crash is usually a
    positive Z that looked small.

    The stock occupies [x0,x1] x [y0,y1] x [-stock_z, 0].
    The table's top face is at z = -stock_z: the stock sits on it, so cutting
    through the stock means cutting the table.
    Clamps stand on the table OUTSIDE the stock footprint and their tops are
    ABOVE z=0. That is not decoration -- it is the reason a rapid at Z=0 across
    the part is a crash, and a gate that ignored the clamps would pass exactly the
    move that ruins the vice.

What is modelled and what is not, since the gate's honesty depends on the list:

    modelled      flat endmill as a swept cylinder; removal = everything above the
                  tool tip within the cylinder; holder and spindle nose as a
                  cylinder at a fixed height above the tip; clamps and table as
                  boxes; tool length = stickout
    NOT modelled   deflection, chatter, tool wear, coolant, thermal growth, the
                  difference between a flat and a ball endmill's floor, and the
                  machine's own acceleration limits. None of those can be seen by
                  a position gate, and claiming them would make this file a
                  simulator it is not.

The removal model is a HEIGHTMAP, one Z per (x, y) cell. For a 3-axis mill that is
exact rather than an approximation: the tool axis is vertical, so a column of
material can only be shortened from the top, and no move can create an undercut.
A 5-axis machine would need a voxel field and this file would be wrong; the file
says so instead of being quietly wrong later.
"""
import math


class Stock:
    """A rectangular block and the fixture around it."""

    def __init__(self, x0=-30.0, y0=-20.0, x1=30.0, y1=20.0, top_z=0.0,
                 stock_z=10.0, clamps=None, cell=0.5):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.top_z = top_z
        self.stock_z = stock_z
        self.table_z = top_z - stock_z
        self.cell = cell
        # clamps: (x0, y0, x1, y1, top_z). Default is four blocks at the corners,
        # each reaching 12 mm above the part -- a height a tool that is 8 mm out
        # of its holder will hit.
        self.clamps = list(clamps) if clamps is not None else [
            (x0 - 12.0, y0 - 12.0, x0 - 2.0, y0 + 4.0, top_z + 12.0),
            (x0 - 12.0, y1 - 4.0, x0 - 2.0, y1 + 12.0, top_z + 12.0),
            (x1 + 2.0, y0 - 12.0, x1 + 12.0, y0 + 4.0, top_z + 12.0),
            (x1 + 2.0, y1 - 4.0, x1 + 12.0, y1 + 12.0, top_z + 12.0),
        ]
        self.nx = max(1, int(round((x1 - x0) / cell)))
        self.ny = max(1, int(round((y1 - y0) / cell)))
        self.dx = (x1 - x0) / self.nx
        self.dy = (y1 - y0) / self.ny
        self.h = [[top_z] * self.ny for _ in range(self.nx)]
        self.removed = 0.0          # mm^3, so "did this program cut anything" is a number
        self.cut_cells = 0

    # -- grid access -------------------------------------------------------
    def cell_centre(self, i, j):
        return (self.x0 + (i + 0.5) * self.dx, self.y0 + (j + 0.5) * self.dy)

    def height_at(self, x, y):
        i = int((x - self.x0) / self.dx)
        j = int((y - self.y0) / self.dy)
        if i < 0 or j < 0 or i >= self.nx or j >= self.ny:
            return None
        return self.h[i][j]

    def inside(self, x, y, margin=0.0):
        return (self.x0 - margin <= x <= self.x1 + margin and
                self.y0 - margin <= y <= self.y1 + margin)

    def current_top(self):
        return max(max(col) for col in self.h)

    def lowest_cut(self):
        return min(min(col) for col in self.h)

    # -- the plant ---------------------------------------------------------
    def sweep(self, a, b, radius, floor_z=None):
        """Cut a straight move with a flat endmill of `radius`.

        Returns the swept volume in mm^3 and the deepest Z actually removed. The
        tool is treated as a vertical cylinder moved along the segment, which is
        the correct sweep for a 3-axis machine and is why this needs no sampling
        in Z -- only in XY, over the cells the cylinder covers.

        `floor_z` is the machine's Z limit: cells are never cut below it, because
        a real spindle stops at the end of its travel rather than continuing.
        """
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(math.ceil(L / (0.5 * min(self.dx, self.dy)))))
        vol = 0.0
        deepest = None
        r_cells = radius + 0.5 * math.hypot(self.dx, self.dy)
        for k in range(n + 1):
            t = k / n
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            z = a[2] + (b[2] - a[2]) * t
            if floor_z is not None and z < floor_z:
                z = floor_z
            i0 = max(0, int((x - r_cells - self.x0) / self.dx))
            i1 = min(self.nx - 1, int((x + r_cells - self.x0) / self.dx))
            j0 = max(0, int((y - r_cells - self.y0) / self.dy))
            j1 = min(self.ny - 1, int((y + r_cells - self.y0) / self.dy))
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    cx, cy = self.cell_centre(i, j)
                    if (cx - x) ** 2 + (cy - y) ** 2 > radius * radius:
                        continue
                    cur = self.h[i][j]
                    if z < cur - 1e-9:
                        removed = cur - z
                        self.h[i][j] = z
                        vol += removed * self.dx * self.dy
                        self.cut_cells += 1
                        deepest = z if deepest is None else min(deepest, z)
        self.removed += vol
        return vol, deepest

    def material_under(self, x, y, z):
        """Is there material at or above `z` in the column at (x, y)?"""
        h = self.height_at(x, y)
        return h is not None and h > z + 1e-9


class Tool:
    """The cutting tool and the metal above it, which is what actually crashes."""

    def __init__(self, diameter=6.0, flute=15.0, stickout=20.0,
                 holder_diameter=30.0, holder_len=40.0):
        self.radius = diameter / 2.0
        self.flute = flute
        self.stickout = stickout
        self.holder_radius = holder_diameter / 2.0
        self.holder_len = holder_len

    def holder_bottom(self, tip_z):
        """Z of the lowest point of the holder, given the tool tip's Z.

        This single number is the most common crash in a real shop: a tool held
        too short so the holder reaches the part before the flutes finish. It is
        also the number a CAM system reports as "stickout" and an AI writing G-code
        never thinks about at all.
        """
        return tip_z + self.stickout

    def __repr__(self):
        return '<Tool D%.1f stickout %.1f>' % (2 * self.radius, self.stickout)


class Machine:
    """The kinematic limits a 3-axis mill has, and nothing more."""

    def __init__(self, stock=None, tool=None, x_min=-100.0, x_max=100.0,
                 y_min=-100.0, y_max=100.0, z_min=-60.0, z_max=100.0,
                 max_feed=3000.0, rapid_safe_margin=2.0):
        self.stock = stock or Stock()
        self.tool = tool or Tool()
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.z_min, self.z_max = z_min, z_max
        self.max_feed = max_feed
        self.rapid_safe_margin = rapid_safe_margin

    # The safe Z a rapid must be at before it travels over the part. It is the
    # taller of the part's current top and the clamp tops, plus a margin -- not a
    # constant, because a constant is how "G0 Z5" becomes a crash after the clamps
    # are changed.
    def safe_z(self):
        top = max(self.stock.current_top(),
                  max([c[4] for c in self.stock.clamps] or [self.stock.current_top()]))
        return top + self.rapid_safe_margin

    def clone_fresh(self):
        """A machine with an uncut stock, for re-running a program twice."""
        s = self.stock
        fresh = Stock(s.x0, s.y0, s.x1, s.y1, s.top_z, s.stock_z,
                      clamps=s.clamps, cell=s.cell)
        return Machine(fresh, Tool(2 * self.tool.radius, self.tool.flute,
                                   self.tool.stickout, 2 * self.tool.holder_radius,
                                   self.tool.holder_len),
                       self.x_min, self.x_max, self.y_min, self.y_max,
                       self.z_min, self.z_max, self.max_feed,
                       self.rapid_safe_margin)

    def __repr__(self):
        return '<Machine stock %.0fx%.0fx%.0f tool %r safe_z %.1f>' % (
            self.stock.x1 - self.stock.x0, self.stock.y1 - self.stock.y0,
            self.stock.stock_z, self.tool, self.safe_z())
