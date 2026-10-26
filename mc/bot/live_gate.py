# -*- coding: utf-8 -*-
"""live_gate.py -- run EmbodiedGate against a RUNNING Minecraft server.

The gate does not get a mirror written by this file. It gets a socket and asks the
body for one column of the world at a time, answered out of the chunk data the
server itself sent. So the two numbers this script prints are produced by two
different code paths on two different machines-of-thought:

    refused / admitted     -- my gate's arithmetic, on world data from the game
    health / final position -- the game's own physics, on the same ticks

usage: python live_gate.py '<json config>'
config keys: port, goal_z, max_drop, gated, out, ticks
"""
import json
import math
import socket
import sys

sys.path.insert(0, "D:\\Users\\china\\Desktop\\项目_开发\\robot-spine")

from spine import embodied as emb
from spine import live, voxel


class LiveBlocks:
    """What embodied.py calls: world.blocks.get((x, y, z)). Answered by the game."""

    def __init__(self, world):
        self.world = world

    def get(self, cell, default=None):
        got = self.world.name_at(cell)
        return default if got is None else got


def main():
    cfg = json.loads(sys.argv[1])
    port = cfg["port"]
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    print("LISTEN", port, flush=True)
    conn, _ = srv.accept()
    print("ACCEPTED", flush=True)
    link = live.LineLink(conn)

    # The world is a socket. y_top 200 / y_bottom -64 is the whole playable column
    # for these scenes, fetched once per (x, z) and cached.
    rworld = live.RemoteWorld(link, y_top=200, y_bottom=-64)

    class Facade:
        """VoxelWorld's shape, backed by the game."""
        blocks = None
        def solid(self, cell):
            return voxel.is_solid(self.blocks.get(cell))
        def hazard(self, cell):
            return voxel.hazard_of(self.blocks.get(cell, "air"))

    world = Facade()
    world.blocks = LiveBlocks(rworld)

    body = emb.bodymod.Body()
    gate = emb.EmbodiedGate(world, body=body, max_drop=cfg.get("max_drop", 3.0),
                            rules=None if cfg.get("gated", True) else ())
    gated = cfg.get("gated", True)

    cum = {"ticks": 0, "admitted": 0, "refused": 0, "reasons": {},
           "first_refusal_tick": None, "start": None, "end": None,
           "min_health": 20.0, "columns_asked": 0}
    trace = []
    goal_z = cfg["goal_z"]
    last = None
    try:
        while True:
            line = link.readline()
            if line is None:
                break
            if not line:
                continue
            if line[0] != "S":
                continue
            s = json.loads(line[1:])
            cum["ticks"] += 1
            p, v = s["pos"], s["vel"]
            cum["start"] = cum["start"] or p
            cum["end"] = p
            cum["min_health"] = min(cum["min_health"], s.get("health", 20))
            last = s
            st = emb.bodymod.State(p[0], p[1], p[2], s["yaw"], v[0], v[1], v[2],
                                   s["on_ground"], False, s["tick"])
            body.stand(st)
            # the plan: walk along z toward goal_z. yaw 0 in this world faces -z,
            # so the sign of the remaining distance also picks the facing.
            remaining = goal_z - p[2]
            if abs(remaining) < 0.5:
                intent, want, yaw_cmd = "stop", [0.0, 0.0, 0.0], 0.0
            else:
                sgn = 1.0 if remaining > 0 else -1.0
                intent = "forward"
                want = [0.0, 0.0, emb.WALK * sgn]
                yaw_cmd = 0.0 if sgn < 0 else 3.14159
            act = {"intent": intent, "keys": {"forward": want[2] != 0.0},
                   "dv": want, "yaw": yaw_cmd}
            reasons = []
            if gated:
                vd = gate.check(st, act)
                ok, reasons = vd.ok, vd.reasons
                # a refusal on a real column means the column just changed under
                # us (a block placed), so drop the cache for the cells ahead
                if not ok:
                    # the column ahead is what the refusal was about; drop it so the
                    # next verdict is computed on the world as it is now
                    rworld.invalidate(p[0] + want[0], p[2] + want[2])
            else:
                ok = True
            if ok:
                cum["admitted"] += 1
            else:
                cum["refused"] += 1
                if cum["first_refusal_tick"] is None:
                    cum["first_refusal_tick"] = s["tick"]
                for r in reasons:
                    cum["reasons"][r] = cum["reasons"].get(r, 0) + 1
                act = {"intent": "STOP:" + ",".join(reasons), "keys": {},
                       "dv": [0.0, 0.0, 0.0], "yaw": s["yaw"]}
            trace.append({"t": s["tick"], "pos": p, "ok": ok, "why": reasons,
                          "intent": act["intent"]})
            cum["columns_asked"] = rworld.queries
            link.send({"act": act})
    except (ConnectionResetError, ConnectionError) as exc:
        print("LINK-DROP", exc, flush=True)

    cum["final_health"] = last.get("health") if last else None
    json.dump({"cum": cum, "trace": trace}, open(cfg["out"], "w"), indent=1)
    print("SUMMARY", json.dumps(cum), flush=True)


if __name__ == "__main__":
    main()
