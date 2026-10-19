# -*- coding: utf-8 -*-
"""mujoco_backend.py -- the same controller, on a real physics engine.

Why this file exists: everything else in this repo runs against a plant whose
equations are four lines of Python, which is convenient and proves nothing about
whether those equations are the arm. MuJoCo resolves the same arm with a
generalised-coordinate solver, collision detection and an RK4 integrator it did
not get from me. If my plant and MuJoCo disagree about gravity, one of us is
wrong and it is worth knowing which.

Two measured claims, both in bench/results.json:

  gravity    my Lagrangian G(q) vs MuJoCo's qfrc_bias on a grid of poses. If the
             arm in demo/two_link.xml is the arm in control.py, these are the same
             number, and the bench prints the difference.
  loop       the same PD controller driven through MuJoCo instead of the analytic
             plant: same setpoint, same gains. Settling and steady-state error are
             reported side by side, and the difference is the physics that the
             analytic plant does not model (it has no friction, no contact, no
             actuator delay, and a different mass distribution inside each link).

Environment notes, measured on this host rather than assumed:

  * MuJoCo's C++ XML loader cannot open a path with non-ASCII characters. This
    repo lives under ".../项目_开发/...", so the model is copied to an ASCII temp
    path before loading. The first attempt at this file failed with
    "Error opening file ... No such file or directory" on a file that existed.
  * numpy's bundled OpenBLAS dies with "Memory allocation still failed after 10
    retries" on this host with 2.4 GB of 14 GB free, unless the BLAS thread count
    is pinned to 1. So the subprocess sets OPENBLAS_NUM_THREADS=1 -- the backend
    runs out of process and sets it for itself.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

SPINE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SPINE_DIR)
XML = os.path.join(REPO, 'demo', 'two_link.xml')

_RUNNER = r"""
import json, sys
import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path(sys.argv[1])
data = mujoco.MjData(model)
cfg = json.loads(sys.argv[2])
out = {}

# ---- 1. gravity agreement on a grid -------------------------------------
pts = cfg['grid']
rows = []
for q in pts:
    data.qpos[:] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    rows.append([float(x) for x in data.qfrc_bias])
out['mujoco_bias'] = rows

# ---- 2. closed loop with a PD, torques supplied from outside --------------
def run_loop(gains, q_des, q0, t_end, dt, tau_cap):
    data.qpos[:] = q0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    kp, kd = gains
    n = int(round(t_end / dt))
    trace = []
    for i in range(n):
        q = [float(data.qpos[0]), float(data.qpos[1])]
        qd = [float(data.qvel[0]), float(data.qvel[1])]
        tau = [kp[j] * (q_des[j] - q[j]) - kd[j] * qd[j] for j in range(2)]
        tau = [max(-tau_cap, min(tau_cap, t)) for t in tau]
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        trace.append([round((i + 1) * dt, 6), q, qd, tau])
    # final resting state, after the loop
    return {'trace': trace,
            'final_q': [float(data.qpos[0]), float(data.qpos[1])],
            'final_qd': [float(data.qvel[0]), float(data.qvel[1])]}

out['loop'] = run_loop(cfg['gains'], cfg['q_des'], cfg['q0'], cfg['t_end'],
                       cfg['dt'], cfg['tau_cap'])

# ---- 3. gravity compensation through MuJoCo: PD + the engine's own bias ---
def run_comp(q_des, q0, t_end, dt, tau_cap):
    data.qpos[:] = q0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    kp, kd = cfg['gains']
    n = int(round(t_end / dt))
    for i in range(n):
        q = [float(data.qpos[0]), float(data.qpos[1])]
        qd = [float(data.qvel[0]), float(data.qvel[1])]
        tau = [kp[j] * (q_des[j] - q[j]) - kd[j] * qd[j] + float(data.qfrc_bias[j])
               for j in range(2)]
        tau = [max(-tau_cap, min(tau_cap, t)) for t in tau]
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
    return {'final_q': [float(data.qpos[0]), float(data.qpos[1])],
            'final_qd': [float(data.qvel[0]), float(data.qvel[1])]}

out['loop_with_bias_comp'] = run_comp(cfg['q_des'], cfg['q0'], cfg['t_end'],
                                      cfg['dt'], cfg['tau_cap'])
print(json.dumps(out))
"""


def _ascii_copy(path):
    """MuJoCo cannot open a non-ASCII path on Windows; give it one it can."""
    tmp = os.path.join(tempfile.gettempdir(), 'robot_spine_mj')
    os.makedirs(tmp, exist_ok=True)
    dst = os.path.join(tmp, os.path.basename(path))
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(path):
        shutil.copy(path, dst)
    return dst


def available():
    """Why this returns False matters more than that it does."""
    env = dict(os.environ)
    env['OPENBLAS_NUM_THREADS'] = '1'
    try:
        r = subprocess.run([sys.executable, '-c', 'import mujoco'],
                           capture_output=True, text=True, timeout=120, env=env)
        return r.returncode == 0, (r.stderr or '').strip()[-300:]
    except Exception as exc:                      # noqa: BLE001
        return False, repr(exc)


def run(grid=None, gains=None, q_des=None, q0=None, t_end=4.0, dt=0.002,
        tau_cap=100.0, timeout=600):
    """Ask MuJoCo the two questions. Returns the raw numbers, no verdicts."""
    ok, why = available()
    if not ok:
        return {'skipped': 'mujoco not importable', 'detail': why}
    if grid is None:
        grid = [[-1.2 + 0.4 * i, -1.2 + 0.4 * j] for i in range(7) for j in range(7)]
    cfg = {'grid': grid,
           'gains': gains or [[18.0, 8.0], [2.2, 1.1]],
           'q_des': q_des or [0.0, 0.0],
           'q0': q0 or [0.6, 0.9],
           't_end': t_end, 'dt': dt, 'tau_cap': tau_cap}
    env = dict(os.environ)
    env.update({'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1', 'PYTHONIOENCODING': 'utf-8'})
    r = subprocess.run([sys.executable, '-c', _RUNNER, _ascii_copy(XML),
                        json.dumps(cfg)],
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace', timeout=timeout, env=env)
    if r.returncode != 0:
        return {'skipped': 'mujoco subprocess failed',
                'detail': (r.stderr or '')[-800:]}
    return json.loads(r.stdout)
