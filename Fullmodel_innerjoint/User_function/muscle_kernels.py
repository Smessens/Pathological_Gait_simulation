# -*- coding: utf-8 -*-
"""numba kernels for the hot paths of gait_controller.GaitController.

joint_torques(): muscle torques, joint limits and inner-thigh forces at every
Runge-Kutta stage; muscle_step(): activation and contractile-element update once per
integration step. They perform the same operations, in the same order, as the
pure-Python methods of GaitController (powers are written as products in both, since
numba and CPython evaluate x**2 differently), so both give identical results; the
controller uses these when numba is installed.

Constant tables (built by GaitController):
    J       joint ids: ankleL, kneeL, hipL, ankleR, kneeR, hipR, innerthighL, innerthighR
    mus     per muscle (14 rows, left then right): l_opt, l_slack, F_max, v_max
    base    l_opt + l_slack of the 7 muscles of a leg
    arc     rho*r0, sin(phi_ref - phi_max), phi_max for VAS@knee, SOL@ankle, GAS@ankle,
            GAS@knee, TA@ankle, HAM@knee
    hipc    rho*r0, phi_ref for HAM, GLU, HFL at the hip
    lever   r0, phi_max for TA@ankle, GAS@ankle, SOL@ankle, GAS@knee, VAS@knee, HAM@knee
    hlev    r0 of HAM, GLU, HFL at the hip
    gen     epsilon_ref, w, c, K, N, eps_pe of the muscle model (N and eps_pe, the strain at
            which the parallel elasticity reaches F_max, include the aging ratios)
"""
import math

import numpy as np
from numba import njit

import Muscle_actuation_layer as muscle

joint_limits = njit(cache=True)(muscle.joint_limits)


@njit(cache=True)
def leg_lmtu(ankle, knee, hip, base, arc, hipc, out, o):
    out[o + 0] = base[0] + arc[0, 0] * (arc[0, 1] - math.sin(knee - arc[0, 2]))
    out[o + 1] = base[1] + arc[1, 0] * (arc[1, 1] - math.sin(ankle - arc[1, 2]))
    out[o + 2] = base[2] + (arc[2, 0] * (arc[2, 1] - math.sin(ankle - arc[2, 2]))
                            - arc[3, 0] * (arc[3, 1] - math.sin(knee - arc[3, 2])))
    out[o + 3] = base[3] + -arc[4, 0] * (arc[4, 1] - math.sin(ankle - arc[4, 2]))
    out[o + 4] = base[4] + (-arc[5, 0] * (arc[5, 1] - math.sin(knee - arc[5, 2])) - hipc[0, 0] * (hip - hipc[0, 1]))
    out[o + 5] = base[5] + -hipc[1, 0] * (hip - hipc[1, 1])
    out[o + 6] = base[6] + hipc[2, 0] * (hip - hipc[2, 1])


@njit(cache=True)
def angles(q, J, out):
    out[0] = q[J[0]]
    out[1] = -q[J[1]]
    out[2] = q[J[2]]
    out[3] = -q[J[3]]
    out[4] = q[J[4]]
    out[5] = -q[J[5]]


@njit(cache=True)
def force(i, lmtu, lce, mus, gen):
    l_se_norm = (lmtu - lce) / mus[i, 1]
    if l_se_norm > 1:
        x = (l_se_norm - 1) / gen[0]
        return x * x * mus[i, 2]
    return 0.0


@njit(cache=True)
def vce(i, lce, lmtu, act, mus, gen):
    eps, w, c, K, N, eps_pe = gen[0], gen[1], gen[2], gen[3], gen[4], gen[5]
    l_se_norm = (lmtu - lce) / mus[i, 1]
    x = (l_se_norm - 1) / eps
    f_se = x * x if l_se_norm > 1 else 0.0
    l_ce_norm = lce / mus[i, 0]
    x = 2 * (l_ce_norm - 1 + w) / w
    f_be = x * x if l_ce_norm - 1 + w < 0 else 0.0
    x = (l_ce_norm - 1) / eps_pe
    f_pe = x * x if l_ce_norm > 1 else 0.0
    x = abs(l_ce_norm - 1) / w
    f_ce = math.exp(c * (x * x * x))
    f_v = (f_se + f_be) / (f_pe + f_ce * act)
    if f_v <= 1:
        v_norm = (f_v - 1) / (f_v * K + 1)
    elif f_v <= N:
        v_norm = ((f_v - N) / (N - 1) + 1) / (1 - 7.56 * K * (f_v - N) / (N - 1))
    else:
        v_norm = 0.01 * (f_v - N) + 1
    return v_norm * mus[i, 3] * mus[i, 0]


@njit(cache=True)
def leg_torques(ankle, knee, hip, d_ankle, d_knee, d_hip, F, o, lever, hlev, parts):
    T_TA = lever[0, 0] * math.cos(ankle - lever[0, 1]) * F[o + 3]
    T_GAS_a = lever[1, 0] * math.cos(ankle - lever[1, 1]) * F[o + 2]
    T_SOL = lever[2, 0] * math.cos(ankle - lever[2, 1]) * F[o + 1]
    T_GAS_k = lever[3, 0] * math.cos(knee - lever[3, 1]) * F[o + 2]
    T_VAS = lever[4, 0] * math.cos(knee - lever[4, 1]) * F[o + 0]
    T_HAM_k = lever[5, 0] * math.cos(knee - lever[5, 1]) * F[o + 4]
    T_HAM_h = hlev[0] * F[o + 4]
    T_GLU = hlev[1] * F[o + 5]
    T_HFL = hlev[2] * F[o + 6]
    parts[0], parts[1], parts[2], parts[3], parts[4] = T_TA, T_GAS_a, T_GAS_k, T_SOL, T_VAS
    parts[5], parts[6], parts[7], parts[8] = T_HAM_k, T_HAM_h, T_GLU, T_HFL
    ankle_t = joint_limits(0, ankle, d_ankle) + T_GAS_a + T_SOL - T_TA
    knee_t = joint_limits(1, knee, d_knee) + T_VAS - T_GAS_k - T_HAM_k
    hip_t = joint_limits(2, hip, d_hip) + T_GLU + T_HAM_h - T_HFL
    return ankle_t, knee_t, hip_t


@njit(cache=True)
def pressure_sheet(p, v):
    u1 = 104967 * p
    u2 = v / 0.5
    return -u1 * (1 + math.copysign(1, u1) * u2)


@njit(cache=True)
def joint_torques(q, qd, lce, J, mus, base, arc, hipc, lever, hlev, gen, Qq):
    a = np.empty(6)
    d = np.empty(6)
    angles(q, J, a)
    angles(qd, J, d)
    lmtu = np.empty(14)
    leg_lmtu(a[0], a[1], a[2], base, arc, hipc, lmtu, 0)
    leg_lmtu(a[3], a[4], a[5], base, arc, hipc, lmtu, 7)
    F = np.empty(14)
    for i in range(14):
        F[i] = force(i, lmtu[i], lce[i], mus, gen)
    parts = np.empty(9)
    ankle_L, knee_L, hip_L = leg_torques(a[0], a[1], a[2], d[0], d[1], d[2], F, 0, lever, hlev, parts)
    ankle_R, knee_R, hip_R = leg_torques(a[3], a[4], a[5], d[3], d[4], d[5], F, 7, lever, hlev, parts)
    Qq[J[6]] = pressure_sheet(q[J[6]], qd[J[6]])
    Qq[J[7]] = - pressure_sheet(-q[J[7]], -qd[J[7]])
    Qq[J[0]] = ankle_L
    Qq[J[1]] = - knee_L
    Qq[J[2]] = hip_L
    Qq[J[3]] = - ankle_R
    Qq[J[4]] = knee_R
    Qq[J[5]] = - hip_R


@njit(cache=True)
def muscle_step(q, stim, dt, tau_act, tau_deact, first, lce_prev, lmtu_prev, act_prev, J, mus, base, arc, hipc,
                gen, act, lmtu, lce, Fm):
    """Activation, muscle-tendon length, contractile length and force after one step."""
    a = np.empty(6)
    angles(q, J, a)
    leg_lmtu(a[0], a[1], a[2], base, arc, hipc, lmtu, 0)
    leg_lmtu(a[3], a[4], a[5], base, arc, hipc, lmtu, 7)
    if first:
        for i in range(14):
            act[i] = stim[i]
            lce[i] = lmtu[i] - mus[i, 1]
    else:
        for i in range(14):
            tau = tau_act if stim[i] >= act_prev[i] else tau_deact
            f = dt / tau
            frac = 1 / (1 + f)
            act[i] = f * frac * stim[i] + frac * act_prev[i]
        for i in range(14):
            v0 = vce(i, lce_prev[i], lmtu_prev[i], act_prev[i], mus, gen)
            v1 = vce(i, lce_prev[i], lmtu[i], act[i], mus, gen)
            lce[i] = lce_prev[i] + 0.5 * (v0 + v1) * dt
    for i in range(14):
        Fm[i] = force(i, lmtu[i], lce[i], mus, gen)
