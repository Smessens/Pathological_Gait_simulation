#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:29:41 2022

@author: matthieuxaussems

Reflex control laws of Geyer & Herr (2010): muscle stimulations computed from
delayed sensory signals. The delays themselves (5, 10 and 20 ms) are applied by
gait_controller.py, which calls these functions once per integration step.
"""

VAS = 0
SOL = 1
GAS = 2
TA = 3
HAM = 4
GLU = 5
HFL = 6


class ReflexParameters:
    """Reflex gains, length offsets and pre-stimulations (defaults: Geyer & Herr 2010)."""

    def __init__(self, parameters):
        get = parameters.get
        self.G_VAS = get("G_VAS", 2e-4)
        self.G_SOL = get("G_SOL", 1.2 / 4000)
        self.G_GAS = get("G_GAS", 1.1 / 1500)
        self.G_TA = get("G_TA", 1.1)
        self.G_SOL_TA = get("G_SOL_TA", 0.0001)
        self.G_HAM = get("G_HAM", 2.166666666666667e-04)
        self.G_GLU = get("G_GLU", 1 / 3000.)
        self.G_HFL = get("G_HFL", 0.5)
        self.G_HAM_HFL = get("G_HAM_HFL", 4)
        self.G_delta_theta = get("G_delta_theta", 1.145915590261647)

        self.loff_TA = get("loff_TA", 0.72)
        self.lopt_TA = get("lopt_TA", 0.06)
        self.loff_HAM = get("loff_HAM", 0.85)
        self.lopt_HAM = get("lopt_HAM", 0.10)
        self.loff_HFL = get("loff_HFL", 0.65)
        self.lopt_HFL = get("lopt_HFL", 0.11)

        self.k_swing = get("k_swing", 0.25)
        self.k_p = get("k_p", 1.909859317102744)
        self.k_d = get("k_d", 0.2)
        self.phi_k_off = get("phi_k_off", 2.967059728390360)
        self.theta_ref = get("theta_ref", 0.104719755119660)

        self.So = get("So", 0.01)
        self.So_VAS = get("So_VAS", 0.08)
        self.So_BAL = get("So_BAL", 0.05)


def lead(stance_L, stance_R, previous, counts, first_step):
    """Which leg leads in double support, from the 10 ms delayed stance signals.

    counts holds, per leg, the number of consecutive steps spent in stance and is
    updated here (once per integration step). previous is the delayed stance of the
    previous step. Returns (RonL, LonR): RonL = 1 when the right leg is the leading
    leg (it landed last), so the left leg should prepare its swing; LonR likewise.
    """
    if first_step:
        return 0, 0

    counts[0] = counts[0] + 1 if stance_L else 0
    counts[1] = counts[1] + 1 if stance_R else 0

    if not (stance_L and stance_R):
        return 0, 0
    countL, countR = counts
    if countL > countR:
        return (0, 0) if countR == 1 else (1, 0)
    if countL < countR:
        return (0, 0) if countL == 1 else (0, 1)
    # both legs landed on the same step: look at the step before
    if previous[0] > previous[1]:
        return 1, 0
    if previous[0] < previous[1]:
        return 0, 1
    return 0, 0


def stimulations(stance, leader, k, n_s, n_m, n_l, r, swing,
                 theta_s, dtheta_s, ipsi_dx_s, contra_dx_s, F_HAM_s, F_GLU_s, lce_HAM_s, lce_HFL_s,
                 F_VAS_m, knee_m,
                 F_SOL_l, F_GAS_l, lce_TA_l):
    """Stimulations of the 7 muscles of one leg at integration step k.

    stance        10 ms delayed stance of this leg (False while t <= 10 ms)
    leader        1 when the other leg leads in double support (this leg should lift off)
    n_s, n_m, n_l short (5 ms), medium (10 ms) and long (20 ms) delays, in steps; the
                  delayed signals below are only read once k exceeds their delay
    r             ReflexParameters
    swing         per-leg dict {"count", "last_PTO"}, updated during swing
    *_s, *_m, *_l sensory signals delayed by n_s, n_m and n_l steps:
                  trunk pitch and its rate, ipsi- and contralateral inner-thigh load
                  displacement, muscle forces [N], contractile lengths [m] and the
                  knee hyperextension state
    """
    Stim = [0.0] * 7

    if stance:
        swing["count"] = 0

        # short delay: hip muscles balance the trunk, scaled by the leg load
        if k <= n_s:
            Stim[HAM] = r.So_BAL
            Stim[GLU] = r.So_BAL
            Stim[HFL] = r.So_BAL
        else:
            delta_theta = theta_s - r.theta_ref
            u1 = r.k_p * delta_theta + r.k_d * dtheta_s
            u1_pos = max(0, u1)
            u1_neg = min(u1, 0)
            u2 = max(0, ipsi_dx_s) * 200

            Stim[HAM] = max(0, min((r.So_BAL + u1_pos) * u2, 1))
            Stim[HFL] = max(0, min((r.So_BAL - u1_neg) * u2, 1)) + leader * r.k_swing
            Stim[GLU] = 0.7 * Stim[HAM] - leader * 0.7 * r.k_swing

        # medium delay: VAS force feedback, knee hyperextension and contralateral load
        if k <= n_m:
            Stim[VAS] = 0
        else:
            Stim[VAS] = r.So_VAS + F_VAS_m * r.G_VAS - 2 * knee_m - leader * 200 * max(0, contra_dx_s)

        # long delay: ankle muscles
        if k <= n_l:
            Stim[GAS] = r.So
            Stim[SOL] = r.So
            Stim[TA] = r.So
        else:
            Stim[GAS] = r.So + F_GAS_l * r.G_GAS
            Stim[SOL] = r.So + F_SOL_l * r.G_SOL
            Stim[TA] = r.So - F_SOL_l * r.G_SOL_TA + max(0, (lce_TA_l / r.lopt_TA - r.loff_TA)) * r.G_TA

    else:
        swing["count"] += 1

        # short delay
        if k <= n_s:
            Stim[HAM] = r.So
            Stim[GLU] = r.So
            Stim[HFL] = r.So
        else:
            Stim[HAM] = r.So + r.G_HAM * F_HAM_s
            Stim[GLU] = r.So + r.G_GLU * F_GLU_s
            if swing["count"] == 1:  # trunk lean at take-off
                swing["last_PTO"] = theta_s - r.theta_ref
            Stim[HFL] = (r.So + r.G_delta_theta * swing["last_PTO"]
                         + r.G_HFL * max(0, (lce_HFL_s / r.lopt_HFL - r.loff_HFL))
                         - r.G_HAM_HFL * max(0, (lce_HAM_s / r.lopt_HAM - r.loff_HAM)))

        Stim[VAS] = 0
        Stim[GAS] = r.So
        Stim[SOL] = r.So

        # long delay
        if k <= n_l:
            Stim[TA] = r.So
        else:
            Stim[TA] = r.So + max(0, (lce_TA_l / r.lopt_TA - r.loff_TA)) * r.G_TA

    for m in range(7):
        Stim[m] = max(0.01, min(Stim[m], 1))
    return Stim


import sys
import os
# Get the directory where your script is located
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(2,  os.path.join(parent_dir, "workR"))
import TestworkR


if __name__ == "__main__":
    TestworkR.runtest(1000e-7,10,c=False)
