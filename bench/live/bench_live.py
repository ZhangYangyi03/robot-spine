# -*- coding: utf-8 -*-
"""bench_live.py -- the same gate, against a running Minecraft server.

Every other bench in this repo compares the gate to a plant this repo also
wrote. That checks the gate against my own arithmetic and nothing more. This one
removes the author from the loop:

    the world the gate reads   -- the server's chunks, over the body's socket
    the physics the gate meets -- the server's own 20 Hz integrator
    the damage the gate is judged by -- the server's health, which the gate never
                                          sees and cannot influence

So the two columns below are independent measurements of the same episode. A
refusal count with health 20 is a gate that was right; a refusal count with a
dead body is a gate that was wrong about the world.

What it runs, on a floating platform 160 blocks up (so "off the edge" is lethal
rather than a two-block step onto grass):

    gated     the intent walks toward the void; the gate is consulted each tick
    ungated   identical intent, gate switched to admit everything -- the control

The control is not decoration. Without it, "the body survived" proves only that
the platform was big enough. With it, survival is attributable to the gate.

usage:
    python bench/live/bench_live.py                    # both arms, writes results
    python bench/live/bench_live.py --gated            # one arm
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BOT_DIR = os.path.join(ROOT, "mc", "bot")
RCON_PORT = 25575
RCON_PASS = "spine"


def rcon():
    sys.path.insert(0, ROOT)
    from spine.live import Rcon
    return Rcon(port=RCON_PORT, password=RCON_PASS)


def play(rc, name, port, goal_z, gated, ticks, place):
    """One episode: body in the game, gate in Python, verdicts and outcome side by side."""
    out = os.path.join(BOT_DIR, name + ".json")
    cfg = {"port": port, "goal_z": goal_z, "max_drop": 3.0, "gated": gated,
           "out": out, "ticks": ticks}
    gate = subprocess.Popen([sys.executable, "-u", "live_gate.py", json.dumps(cfg)],
                            cwd=BOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    time.sleep(1.0)
    args = ["node", "spine-body.js", name + ".jsonl", str(port), str(ticks)]
    if place:
        args.append(",".join(str(v) for v in place))
    body = subprocess.Popen(args, cwd=BOT_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    try:
        body_out, _ = body.communicate(timeout=ticks * 0.12 + 120)
    except subprocess.TimeoutExpired:
        body.kill()
        body_out = "TIMEOUT"
    try:
        gate_out, _ = gate.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        gate.kill()
        gate_out = ""

    summary = next((json.loads(l[8:]) for l in (gate_out or "").splitlines()
                    if l.startswith("SUMMARY")), None)
    if summary is None:
        return {"arm": name, "error": (gate_out or "")[-400:]}

    rows = []
    path = os.path.join(BOT_DIR, name + ".jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh if '"s"' in l]
    deaths = [l for l in (body_out or "").splitlines() if l.startswith("DEATH")]

    summary.update({
        "arm": name,
        "gated": gated,
        "real_min_health": round(min(r["s"]["health"] for r in rows), 2) if rows else None,
        "real_final": rows[-1]["s"]["pos"] if rows else None,
        "fell": bool(deaths),
        "death_log": deaths,
        "goal_z": goal_z,
    })
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=260)
    ap.add_argument("--goal-z", type=float, default=-60.0)
    ap.add_argument("--platform", default="0,101,0", help="where the bench places the body")
    ap.add_argument("--gated", action="store_true")
    ap.add_argument("--ungated", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results_live.json"))
    args = ap.parse_args()
    if not (args.gated or args.ungated):
        args.gated = args.ungated = True

    rc = rcon()
    place = tuple(float(v) for v in args.platform.split(","))
    results = []
    port = 7900
    for gated, name in ((True, "arm_gated"), (False, "arm_ungated")):
        if gated and not args.gated:
            continue
        if not gated and not args.ungated:
            continue
        rc.cmd("kill @e[type=player]")
        time.sleep(0.6)
        for f in glob.glob(os.path.join(ROOT, "mc", "flatworld", "playerdata", "*.dat*")):
            try:
                os.remove(f)
            except OSError:
                pass
        got = play(rc, name, port, args.goal_z, gated, args.ticks, place)
        port += 1
        results.append(got)
        print(name, json.dumps({k: got.get(k) for k in
                                ("admitted", "refused", "reasons", "real_min_health",
                                 "real_final", "fell", "columns_asked")}, ensure_ascii=False))

    payload = {"host": "minecraft 1.21.1 server on 127.0.0.1:25565",
               "ticks": args.ticks, "goal_z": args.goal_z,
               "platform": list(place), "arms": results}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
