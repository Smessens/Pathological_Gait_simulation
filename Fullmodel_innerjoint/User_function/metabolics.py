# -*- coding: utf-8 -*-
"""Metabolic power of the muscles of Geyer's model.

Umberger et al. (2003) muscle energy model as implemented by OpenSim's
Umberger2010MuscleMetabolicsProbe with its default options (aerobic factor 1.5,
Bhargava recruitment of slow- and fast-twitch fibres, negative mechanical work
included, total power of a muscle never negative, heat rate of a muscle at least
1 W/kg). Muscle mass = F_max / specific tension x density x l_opt. The basal rate
(1.2 W per kg of body mass) is added by the caller.

The contractile element's velocity and active force come from the state of Geyer's
muscle model (Muscle_actuation_layer.vce_compute): F_ce = F_max a f_l f_v, with the
force-velocity factor f_v = (f_se + f_be) / (f_pe + f_l a); the velocity is < 0 when
shortening, as in OpenSim.
"""
import numpy as np

SPECIFIC_TENSION = 0.25e6   # [N/m^2]
DENSITY = 1059.7            # [kg/m^3]
AEROBIC_FACTOR = 1.5
BASAL_RATE = 1.2            # [W per kg of body mass]
# Fraction of slow-twitch fibres, VAS SOL GAS TA HAM GLU HFL (Johnson et al. 1973)
SLOW_TWITCH = (0.47, 0.80, 0.50, 0.73, 0.55, 0.52, 0.50)


class MuscleMetabolics:
    """Metabolic power [W] of the 14 muscles (left VAS..HFL, then right) for a given state."""

    def __init__(self, Fmax, lopt, lslack, vmax, eps_pe, N, w, c, K, eps_ref):
        self.Fmax, self.lopt, self.lslack = (np.asarray(x, dtype=float) for x in (Fmax, lopt, lslack))
        self.vmax = np.asarray(vmax, dtype=float)          # [l_opt/s]
        self.eps_pe, self.N, self.w, self.c, self.K, self.eps_ref = eps_pe, N, w, c, K, eps_ref
        self.mass = self.Fmax / SPECIFIC_TENSION * DENSITY * self.lopt
        self.ratio = np.tile(SLOW_TWITCH, len(self.Fmax) // len(SLOW_TWITCH))
        self.alpha_fast = 153.0 / self.vmax                 # fast-twitch V_max = muscle V_max
        self.alpha_slow = 100.0 / (self.vmax / 2.5)         # slow-twitch V_max = V_max / 2.5

    def state(self, act, lce, lmtu):
        """Normalized length, force-length factor, velocity [l_opt/s] and active force [N] of the CE."""
        l_se = (np.asarray(lmtu) - lce) / self.lslack
        f_se = np.where(l_se > 1, ((l_se - 1) / self.eps_ref) ** 2, 0.0)
        l_ce = np.asarray(lce) / self.lopt
        f_be = np.where(l_ce - 1 + self.w < 0, (2 * (l_ce - 1 + self.w) / self.w) ** 2, 0.0)
        f_pe = np.where(l_ce > 1, ((l_ce - 1) / self.eps_pe) ** 2, 0.0)
        f_l = np.exp(self.c * (np.abs(l_ce - 1) / self.w) ** 3)
        f_v = (f_se + f_be) / (f_pe + f_l * act)
        K, N = self.K, self.N
        v = np.where(f_v <= 1, (f_v - 1) / (f_v * K + 1),
                     np.where(f_v <= N, ((f_v - N) / (N - 1) + 1) / (1 - 7.56 * K * (f_v - N) / (N - 1)),
                              0.01 * (f_v - N) + 1)) * self.vmax
        return l_ce, f_l, v, self.Fmax * act * f_l * f_v

    def power(self, u, act, lce, lmtu):
        """Total metabolic power of the muscles [W] (u: excitations, act: activations)."""
        u, act = np.asarray(u, dtype=float), np.asarray(act, dtype=float)
        l_ce, f_l, v, F_ce = self.state(act, lce, lmtu)
        A = np.where(u > act, u, 0.5 * (u + act))
        # Bhargava et al. (2004): slow-twitch fibres are recruited first
        u_slow = self.ratio * np.sin(0.5 * np.pi * u)
        u_fast = (1 - self.ratio) * (1 - np.cos(0.5 * np.pi * u))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(u == 0, 1.0, u_slow / (u_slow + u_fast))
        S = AEROBIC_FACTOR
        # activation and maintenance heat [W/kg]
        h_am = 128 * (1 - ratio) + 25
        am = S * A ** 0.6 * np.where(l_ce <= 1, h_am, 0.4 * h_am + 0.6 * h_am * f_l)
        # shortening (v <= 0) and lengthening heat [W/kg]
        slow = np.minimum(-self.alpha_slow * v, 100.0)
        fast = self.alpha_fast * v * (1 - ratio)
        sl = np.where(v <= 0, S * A ** 2 * (slow * ratio - fast), S * A * 4.0 * self.alpha_slow * v)
        sl = np.where(l_ce > 1, sl * f_l, sl)
        # mechanical work rate of the contractile element [W/kg], negative work included
        work = -F_ce * v * self.lopt / self.mass
        sl = np.where(am + sl + work < 0, sl - (am + sl + work), sl)   # total power never negative
        heat = np.maximum(am + sl, 1.0)                                 # at least 1 W/kg of heat
        return float(np.sum((heat + work) * self.mass))
