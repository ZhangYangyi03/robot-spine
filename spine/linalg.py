# -*- coding: utf-8 -*-
"""linalg.py -- the 3D rigid-body maths, stdlib only.

Pure Python on purpose: this repo is meant to run on the same kind of machine
the tools it plugs into run on (craftagent ships no dependencies for the same
reason), and on this host a numpy import has been seen to die with MemoryError
while 1.5 GB of 14 GB was free. 4x4 matrices are lists of lists.

Conventions:
  * transforms are 4x4, row-major, right-handed, translations in the last column
  * rotations are rotation matrices; the 6D representation is the first two
    COLUMNS of R, row-major (the same one unifolm-wla-toolkit uses, spec 3.1)
  * axis-angle uses the Rodrigues formula; rpy is fixed-axis xyz (URDF / spec 5.1)
"""
import math


def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def mat_mul(A, B):
    n, k, m = len(A), len(B), len(B[0])
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Oi = out[i]
        for t in range(k):
            a = Ai[t]
            if a == 0.0:
                continue
            Bt = B[t]
            for j in range(m):
                Oi[j] += a * Bt[j]
    return out


def mat_vec(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def transpose(A):
    return [list(r) for r in zip(*A)]


def inv_T(T):
    """Inverse of a rigid transform, without a general inverse."""
    R = [[T[i][j] for j in range(3)] for i in range(3)]
    p = [T[i][3] for i in range(3)]
    Rt = transpose(R)
    pi = mat_vec(Rt, p)
    out = eye(4)
    for i in range(3):
        for j in range(3):
            out[i][j] = Rt[i][j]
        out[i][3] = -pi[i]
    return out


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def scale(a, s):
    return [x * s for x in a]


def rpy_to_R(rpy):
    r, p, y = [float(v) for v in rpy]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]
    Ry = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
    Rz = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]
    return mat_mul(mat_mul(Rz, Ry), Rx)


def axis_angle_R(axis, angle):
    n = norm(axis)
    if n < 1e-12:
        return eye(3)
    u = [a / n for a in axis]
    c, s = math.cos(angle), math.sin(angle)
    K = [[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]
    K2 = mat_mul(K, K)
    return [[eye(3)[i][j] + s * K[i][j] + (1 - c) * K2[i][j] for j in range(3)]
            for i in range(3)]


def so3_log(R):
    """Rotation matrix -> rotation vector (axis * angle).

    The naive `theta / (2 sin theta)` blows up near 0 and near pi, so it is split
    three ways. Written this carefully because the first version of this file
    returned ZERO for a 1e-7 rad rotation, which silently made every
    finite-difference Jacobian check in the bench wrong by 1.0 rad/s.
    """
    tr = R[0][0] + R[1][1] + R[2][2]
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    v = [0.5 * (R[2][1] - R[1][2]), 0.5 * (R[0][2] - R[2][0]), 0.5 * (R[1][0] - R[0][1])]
    s = norm(v)                                   # s = sin(theta)
    if s > 1e-12:
        th = math.atan2(s, c)
        return scale(v, th / s)
    if c > 0.0:                                   # theta ~ 0: second-order term
        return [0.0, 0.0, 0.0]
    m = [[(R[i][j] + R[j][i]) / 2.0 for j in range(3)] for i in range(3)]
    i = max(range(3), key=lambda k: m[k][k])
    w = [0.0, 0.0, 0.0]
    w[i] = math.sqrt(max(0.0, (m[i][i] + 1.0) / 2.0))
    for k in range(3):
        if k != i:
            w[k] = m[i][k] / (2.0 * w[i]) if w[i] > 1e-12 else 0.0
    return scale(w, math.pi)


def rot_err(R_cur, R_des):
    """Orientation error as a 3-vector: log(R_cur^T R_des), i.e. world-frame."""
    return so3_log(mat_mul(transpose(R_cur), R_des))


def rot_to_r6(R):
    """rotation-6D: first two columns of R, row-major (unifolm-wla spec 3.1)."""
    return [R[i][0] for i in range(3)] + [R[i][1] for i in range(3)]


def r6_to_rot(r6):
    """Inverse of rot_to_r6 with Gram-Schmidt (the part the resume calls out)."""
    a = [r6[0], r6[1], r6[2]]
    b = [r6[3], r6[4], r6[5]]
    na = norm(a)
    if na < 1e-12:
        return eye(3)
    a = [x / na for x in a]
    d = dot(a, b)
    b = [b[i] - d * a[i] for i in range(3)]
    nb = norm(b)
    b = [x / nb for x in b] if nb > 1e-12 else [0.0, 0.0, 0.0]
    c = cross(a, b)
    return [[a[i], b[i], c[i]] for i in range(3)]


def chol_solve(A, b):
    """Solve A x = b for symmetric positive (semi-definite) A, n small. None if not PD."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 1e-15:
                    return None
                L[i][j] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def dls(J, e, lam):
    """Damped least squares: dq = (J^T J + lam^2 I)^-1 J^T e, with lam raised if needed."""
    n = len(J[0])
    Jt = transpose(J)
    JtJ = mat_mul(Jt, J)
    Jte = mat_vec(Jt, e)
    l = lam
    for _ in range(8):
        A = [[JtJ[i][j] + (l * l if i == j else 0.0) for j in range(n)] for i in range(n)]
        x = chol_solve(A, Jte)
        if x is not None:
            return x
        l *= 4.0
    return [0.0] * n
