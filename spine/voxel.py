# -*- coding: utf-8 -*-
"""voxel.py -- a world made of blocks, and the distance from a capsule to it.

This is the seam between the two halves of spine-agent. robot-spine knows how to
compute where a link is and how far it is from another link; a voxel world knows
what is solid and where. Neither knows the other, and the merge is the one
function at the bottom of this file: capsule-to-block distance with a sign.

Why distance and not overlap: the interesting question for a controller walking a
body through a Minecraft world is not "did it intersect" (by then the tick is
already wrong) but "how much room is left", so that a gate can refuse the tick
that spends the last of it. That makes the whole check arithmetic, with no
sampling and no randomness, exactly like capsule_capsule in collision.py.

Coordinates. Minecraft is y-up; robot-spine's URDFs are z-up. The two frames are
related by to_spine_frame(), a proper rotation (det = +1, checked by a test), so a
world position and a URDF target are the same point expressed twice rather than
two conventions that drift apart.
"""
import math

# a block that does not stop a body (Minecraft's own list, trimmed to what the
# scenes here use). Everything else is solid.
PASSABLE = frozenset((
    'air', 'cave_air', 'void_air', 'water', 'lava', 'fire', 'torch', 'grass',
    'fern', 'tall_grass', 'snow_layer', 'vine', 'ladder', 'sign', 'rail',
))

# what hurts to stand in, and how much per tick (half-hearts; a player has 20)
HAZARD = {
    'lava': 4.0,
    'fire': 1.0,
    'magma_block': 1.0,
    'cactus': 1.0,
    'campfire': 1.0,
}


def is_solid(name):
    """A block stops a body if it is not in PASSABLE. None (empty) is not solid."""
    return bool(name) and name not in PASSABLE


def hazard_of(name):
    return HAZARD.get(name, 0.0)


class VoxelWorld:
    """Sparse block storage: {(x, y, z): name}. Absent means air.

    Sparse on purpose -- a flat 256^2 * 128 array is 8M cells and this repo's
    scenes are a few thousand blocks. The trade is that a column query is a dict
    lookup per cell instead of an index, which the bench measures.
    """

    def __init__(self, blocks=None):
        self.blocks = dict(blocks or {})

    # -- construction ----------------------------------------------------
    @classmethod
    def flat(cls, width=16, depth=16, top=64, name='stone', bottom=0):
        """A slab of solid blocks with the top layer at y=top, feet at top+1."""
        w = cls()
        w.add_box((0, bottom, 0), (width - 1, top, depth - 1), name)
        return w

    def copy(self):
        return VoxelWorld(self.blocks)

    def add_box(self, lo, hi, name='stone'):
        """Inclusive cell range."""
        for x in range(lo[0], hi[0] + 1):
            for y in range(lo[1], hi[1] + 1):
                for z in range(lo[2], hi[2] + 1):
                    self.blocks[(x, y, z)] = name
        return self

    def carve_box(self, lo, hi):
        for x in range(lo[0], hi[0] + 1):
            for y in range(lo[1], hi[1] + 1):
                for z in range(lo[2], hi[2] + 1):
                    self.blocks.pop((x, y, z), None)
        return self

    def set(self, cell, name='stone'):
        if name is None or name == 'air':
            self.blocks.pop(tuple(cell), None)
        else:
            self.blocks[tuple(cell)] = name
        return self

    def get(self, cell):
        return self.blocks.get(tuple(cell))

    def solid(self, cell):
        return is_solid(self.blocks.get(tuple(cell)))

    def hazard(self, cell):
        return hazard_of(self.blocks.get(tuple(cell), 'air'))

    # -- queries used by the planner and the gate -------------------------
    def top_at(self, x, z, ceiling=320):
        """Highest solid y in column (x, z), or None for an empty column."""
        for y in range(int(ceiling), -320, -1):
            if self.solid((x, y, z)):
                return y
        return None

    def column(self, x, z):
        return sorted(c[1] for c in self.blocks if c[0] == x and c[2] == z
                      and is_solid(self.blocks[c]))

    def is_hazard_column(self, x, z, y_lo, y_hi):
        """Any hazard block in the column between two heights, inclusive."""
        for y in range(int(math.floor(y_lo)), int(math.ceil(y_hi)) + 1):
            if self.hazard((x, y, z)) > 0.0:
                return self.blocks[(x, y, z)]
        return None

    def __len__(self):
        return len(self.blocks)

    def __repr__(self):
        return '<VoxelWorld %d blocks %s>' % (len(self.blocks),
                                              sorted(set(self.blocks.values()))[:4])


# -- distance, which is the whole point --------------------------------------

def dist_point_box(p, cell):
    """Euclidean distance from a point to an axis-aligned unit cube. Closed form."""
    x, y, z = cell
    dx = max(x - p[0], 0.0, p[0] - (x + 1.0))
    dy = max(y - p[1], 0.0, p[1] - (y + 1.0))
    dz = max(z - p[2], 0.0, p[2] - (z + 1.0))
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def dist_seg_box(p, q, cell, iters=48):
    """Distance from the segment pq to a unit cube.

    d(t) = dist(p + t (q - p), box) is convex in t (distance to a convex set is
    convex, and the parametrisation is affine), so a ternary search converges to
    the minimum and cannot get the answer wrong -- it can only be imprecise in t.
    48 iterations shrink the bracket by (2/3)^48 ~ 4e-9, and the bench reports the
    worst error against a dense brute-force sample instead of assuming that.
    """
    best_hi = 1.0
    lo, hi = 0.0, 1.0
    dx, dy, dz = q[0] - p[0], q[1] - p[1], q[2] - p[2]

    def f(t):
        return dist_point_box((p[0] + t * dx, p[1] + t * dy, p[2] + t * dz), cell)

    for _ in range(iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if f(m1) <= f(m2):
            hi = m2
        else:
            lo = m1
        if hi - lo < 1e-12:
            break
    return min(f(0.5 * (lo + hi)), f(0.0), f(1.0))


def dist_seg_box_sampled(p, q, cell, n=4000):
    """Brute force, for checking dist_seg_box. Not used by the gate."""
    dx, dy, dz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
    best = dist_point_box(p, cell)
    for i in range(1, n):
        t = i / float(n)
        d = dist_point_box((p[0] + t * dx, p[1] + t * dy, p[2] + t * dz), cell)
        if d < best:
            best = d
    return best


def cells_near(p, q, r, margin=0.0):
    """Integer cells whose unit cube can come within r + margin of the segment."""
    pad = r + margin
    lo = [min(p[i], q[i]) - pad for i in range(3)]
    hi = [max(p[i], q[i]) + pad for i in range(3)]
    for x in range(int(math.floor(lo[0])), int(math.floor(hi[0])) + 1):
        for y in range(int(math.floor(lo[1])), int(math.floor(hi[1])) + 1):
            for z in range(int(math.floor(lo[2])), int(math.floor(hi[2])) + 1):
                yield (x, y, z)


def capsule_world_clearance(world, p, q, r, margin=0.0, return_first=False):
    """Signed clearance from a capsule (segment pq, radius r) to the solid world.

    Negative means the capsule is inside a block. Returns (distance, cell, name);
    distance is +inf when nothing solid is within r + margin, which is the
    honest answer -- "no block close enough to matter", not "clear".
    """
    best, best_cell, best_name = float('inf'), None, None
    for cell in cells_near(p, q, r, margin):
        name = world.blocks.get(cell)
        if name is None or not is_solid(name):
            continue
        d = dist_seg_box(p, q, cell) - r
        if d < best:
            best, best_cell, best_name = d, cell, name
            if return_first and d < 0.0:
                return best, best_cell, best_name
    return best, best_cell, best_name


def aabb_clearance(world, lo, hi, margin=0.0):
    """Signed clearance from an axis-aligned box to the solid world (body hitbox)."""
    p = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0]
    best, best_cell, best_name = float('inf'), None, None
    for x in range(int(math.floor(lo[0] - margin)), int(math.floor(hi[0] + margin)) + 1):
        for y in range(int(math.floor(lo[1] - margin)), int(math.floor(hi[1] + margin)) + 1):
            for z in range(int(math.floor(lo[2] - margin)), int(math.floor(hi[2] + margin)) + 1):
                name = world.blocks.get((x, y, z))
                if name is None or not is_solid(name):
                    continue
                # box-vs-box: the largest per-axis overlap, negated, is the
                # penetration if they overlap at all on every axis
                dx = min(hi[0], x + 1.0) - max(lo[0], x)
                dy = min(hi[1], y + 1.0) - max(lo[1], y)
                dz = min(hi[2], z + 1.0) - max(lo[2], z)
                if dx > 0 and dy > 0 and dz > 0:
                    d = -min(dx, dy, dz)
                else:
                    d = math.sqrt(max(0.0, -dx) ** 2 + max(0.0, -dy) ** 2
                                  + max(0.0, -dz) ** 2)
                if d < best:
                    best, best_cell, best_name = d, (x, y, z), name
    return best, best_cell, best_name


# -- the one transform between the two halves --------------------------------

def to_spine_frame(p):
    """Minecraft (x, y up, z) -> robot-spine (x, y, z up). A -90 deg turn about x."""
    return [p[0], -p[2], p[1]]


def from_spine_frame(s):
    return [s[0], s[2], -s[1]]
