# -*- coding: utf-8 -*-
"""Neuromuscular state of the walking model and its time stepping.

The state (low-pass filters, neural delay lines, muscle activations and contractile
element lengths) advances once per accepted integration step, in step(), called from
user_dirdyn_loop with dt equal to the integration step. joint_torques(), called from
user_JointForces at every Runge-Kutta stage, only turns that state and the joint
angles of the stage into joint torques; it never advances the state.

One controller is created per simulation (user_dirdyn_init), so nothing leaks from
one run to the next.
"""
import math
from datetime import datetime

import numpy as np

import Muscle_actuation_layer as muscle
import Neural_control_layer as neural
import gait_graph

try:
    import muscle_kernels as kernels  # numba versions of joint_torques and of the muscle update
except ImportError:
    kernels = None

VAS, SOL, GAS, TA, HAM, GLU, HFL = range(7)
ANKLE, KNEE, HIP = range(3)

TAU_ACTIVATION = 0.01        # excitation-contraction coupling [s]
TAU_THIGH_LOAD = 0.02        # low-pass filter of the inner-thigh displacement [s]
NEURAL_DELAYS = (0.005, 0.01, 0.02)  # short, medium, long [s]
MEASURE_PERIOD = 0.1         # fitness bookkeeping period [s]
TARGET_SPEED = 1.3           # [m/s]


def get(mbs_data):
    """Controller of this simulation (created on first use if user_dirdyn_init did not)."""
    controller = getattr(mbs_data, "gait_controller", None)
    if controller is None:
        controller = start(mbs_data, mbs_data.user_model.get("dt", 0))
    return controller


def start(mbs_data, dt):
    """Create a fresh controller for a new simulation."""
    mbs_data.gait_controller = GaitController(mbs_data, dt)
    return mbs_data.gait_controller


class DelayLine:
    """Values of a signal for the last `length` integration steps."""

    def __init__(self, length):
        self.length = length
        self.data = [None] * length

    def push(self, k, value):
        self.data[k % self.length] = value

    def at(self, k):
        """Value pushed at integration step k (k must be within the last `length` steps)."""
        return self.data[k % self.length]


def trunk_angle(P_hip, P_trunk, V_hip, V_trunk):
    """Trunk pitch w.r.t. the vertical and its rate, as useful_functions.trunk_angle."""
    dx = P_trunk[1] - P_hip[1]
    dz = P_trunk[3] - P_hip[3]
    dvx = V_trunk[1] - V_hip[1]
    dvz = V_trunk[3] - V_hip[3]
    return math.atan2(dx, -dz), (-dz * dvx + dx * dvz) / (0.8) ** 2


def pressure_sheet(p, v):
    """Inner-thigh contact force, as useful_functions.pressure_sheet."""
    u1 = 104967 * p
    u2 = v / 0.5
    return -u1 * (1 + math.copysign(1, u1) * u2)


def low_filter(x, tau, dt, previous):
    """First-order low-pass filter (backward Euler), as useful_functions.low_filter."""
    f = dt / tau
    frac = 1 / (1 + f)
    return f * frac * x + frac * previous


class GaitController:

    def __init__(self, mbs_data, dt):
        p = mbs_data.user_model
        self.p = p
        self.dt = dt
        self.tf = p.get("tf", 0)
        self.flag_graph = p.get("flag_graph", 0)
        self.flag_fitness = p.get("flag_fitness", False)
        self.speed_window = p.get("speed_window", 0.3)  # allowed distance to the 1.3 m/s target [m]
        self.id = p.get("id", 0)
        self.reflex = neural.ReflexParameters(p)

        muscle.set_parameters(p)
        self._muscle_constants()

        self.n_s, self.n_m, self.n_l = (int(round(d / dt)) for d in NEURAL_DELAYS)
        self.n_measure = int(round(MEASURE_PERIOD / dt))
        size = self.n_l + 2
        self.hist_thigh = DelayLine(size)   # filtered inner-thigh displacement (L, R)
        self.hist_trunk = DelayLine(size)   # trunk pitch and pitch rate
        self.hist_stance = DelayLine(size)  # contact-based stance (L, R)
        self.hist_knee = DelayLine(size)    # knee hyperextension state (L, R)
        self.hist_muscle = DelayLine(size)  # muscle forces and contractile lengths (14 each)

        jid = mbs_data.joint_id
        self.jA_L, self.jK_L, self.jH_L = jid["ankleL"], jid["kneeL"], jid["hipL"]
        self.jA_R, self.jK_R, self.jH_R = jid["ankleR"], jid["kneeR"], jid["hipR"]
        self.jT_L, self.jT_R = jid["innerthighL"], jid["innerthighR"]
        sid = mbs_data.sensor_id
        self.s_trunk, self.s_hip = sid["Sensor_trunk"], sid["Sensor_hip"]
        self.s_feet = (sid["Sensor_BallL"], sid["Sensor_HeelL"], sid["Sensor_BallR"], sid["Sensor_HeelR"])

        # Muscle state, both legs: indices 0-6 left, 7-13 right (VAS, SOL, GAS, TA, HAM, GLU, HFL)
        q = mbs_data.q.tolist()
        self.lmtu = self._lmtu_legs(q)
        self.lce = [l - ls for l, ls in zip(self.lmtu, self.lslack)]  # series elastic element at slack length
        self.act = [0.01] * 14
        self.stim = [0.01] * 14
        self.Fm = [0.0] * 14

        self.use_kernels = kernels is not None and p.get("flag_numba", True)
        if self.use_kernels:
            self._kernel_tables()

        self.k = -1              # index of the last accepted integration step
        self.t_last = None
        self.Ldx = 0.0
        self.Rdx = 0.0
        self.lead_counts = [0, 0]
        self.previous_stance = (0, 0)
        self.swing = ({"count": 0, "last_PTO": 0}, {"count": 0, "last_PTO": 0})
        self.total_fm = 0.0
        self.stop_reason = None
        self.prev_datetime = datetime.now()

    # ------------------------------------------------------------------ muscle model

    def _muscle_constants(self):
        """Per-muscle constants of Muscle_actuation_layer as Python floats (both legs)."""
        M = muscle
        self.lopt = [float(x) for x in M.l_opt_muscle] * 2
        self.lslack = [float(x) for x in M.l_slack_muscle] * 2
        self.Fmax = [float(x) for x in M.F_max_muscle] * 2
        self.vmax = [float(x) for x in M.v_max_muscle] * 2
        self.base = [float(lo + ls) for lo, ls in zip(M.l_opt_muscle, M.l_slack_muscle)]

        def arc(joint, m):  # rho*r0, sin(phi_ref - phi_max), phi_max of a muscle crossing ankle or knee
            return (float(M.rho[joint, m] * M.r_0[joint, m]),
                    math.sin(M.phi_ref[joint, m] - M.phi_max[joint, m]), float(M.phi_max[joint, m]))
        self.c_VAS_k = arc(KNEE, VAS)
        self.c_SOL_a = arc(ANKLE, SOL)
        self.c_GAS_a = arc(ANKLE, GAS)
        self.c_GAS_k = arc(KNEE, GAS)
        self.c_TA_a = arc(ANKLE, TA)
        self.c_HAM_k = arc(KNEE, HAM)
        self.c_HAM_h = (float(M.rho[HIP, HAM] * M.r_0[HIP, HAM]), float(M.phi_ref[HIP, HAM]))
        self.c_GLU_h = (float(M.rho[HIP, GLU] * M.r_0[HIP, GLU]), float(M.phi_ref[HIP, GLU]))
        self.c_HFL_h = (float(M.rho[HIP, HFL] * M.r_0[HIP, HFL]), float(M.phi_ref[HIP, HFL]))

        def lever(joint, m):  # r0, phi_max
            return float(M.r_0[joint, m]), float(M.phi_max[joint, m])
        self.r_TA_a, self.r_GAS_a, self.r_SOL_a = lever(ANKLE, TA), lever(ANKLE, GAS), lever(ANKLE, SOL)
        self.r_GAS_k, self.r_VAS_k, self.r_HAM_k = lever(KNEE, GAS), lever(KNEE, VAS), lever(KNEE, HAM)
        self.r_HAM_h, self.r_GLU_h, self.r_HFL_h = (float(M.r_0[HIP, HAM]), float(M.r_0[HIP, GLU]),
                                                     float(M.r_0[HIP, HFL]))

    def _kernel_tables(self):
        """Constant tables and state arrays for muscle_kernels (see its docstring)."""
        self.J = np.array([self.jA_L, self.jK_L, self.jH_L, self.jA_R, self.jK_R, self.jH_R, self.jT_L, self.jT_R],
                          dtype=np.int64)
        self.k_mus = np.array([self.lopt, self.lslack, self.Fmax, self.vmax]).T.copy()
        self.k_base = np.array(self.base)
        self.k_arc = np.array([self.c_VAS_k, self.c_SOL_a, self.c_GAS_a, self.c_GAS_k, self.c_TA_a, self.c_HAM_k])
        self.k_hipc = np.array([self.c_HAM_h, self.c_GLU_h, self.c_HFL_h])
        self.k_lever = np.array([self.r_TA_a, self.r_GAS_a, self.r_SOL_a, self.r_GAS_k, self.r_VAS_k, self.r_HAM_k])
        self.k_hlev = np.array([self.r_HAM_h, self.r_GLU_h, self.r_HFL_h])
        self.k_gen = np.array([muscle.epsilon_ref, muscle.w_muscle, muscle.c, muscle.K_muscle, muscle.N_muscle], dtype=float)
        self.lce_a, self.lmtu_a, self.act_a = np.array(self.lce), np.array(self.lmtu), np.array(self.act)

    def _lmtu(self, ankle, knee, hip):
        """Muscle-tendon lengths of one leg, as the lmtu_update* functions."""
        b = self.base
        c, s, pm = self.c_VAS_k
        VAS_ = b[VAS] + c * (s - math.sin(knee - pm))
        c, s, pm = self.c_SOL_a
        SOL_ = b[SOL] + c * (s - math.sin(ankle - pm))
        ca, sa, pma = self.c_GAS_a
        ck, sk, pmk = self.c_GAS_k
        GAS_ = b[GAS] + (ca * (sa - math.sin(ankle - pma)) - ck * (sk - math.sin(knee - pmk)))
        c, s, pm = self.c_TA_a
        TA_ = b[TA] + -c * (s - math.sin(ankle - pm))
        ck, sk, pmk = self.c_HAM_k
        ch, prh = self.c_HAM_h
        HAM_ = b[HAM] + (-ck * (sk - math.sin(knee - pmk)) - ch * (hip - prh))
        ch, prh = self.c_GLU_h
        GLU_ = b[GLU] + -ch * (hip - prh)
        ch, prh = self.c_HFL_h
        HFL_ = b[HFL] + ch * (hip - prh)
        return [VAS_, SOL_, GAS_, TA_, HAM_, GLU_, HFL_]

    def _angles(self, q):
        """Joint angles in the conventions of the muscle model."""
        return (q[self.jA_L], -q[self.jK_L], q[self.jH_L], -q[self.jA_R], q[self.jK_R], -q[self.jH_R])

    def _lmtu_legs(self, q):
        aL, kL, hL, aR, kR, hR = self._angles(q)
        return self._lmtu(aL, kL, hL) + self._lmtu(aR, kR, hR)

    def _force(self, i, lmtu, lce):
        """Muscle force F_m = F_max f_se(l_se), as vce_compute(...)[1]."""
        l_se_norm = (lmtu - lce) / self.lslack[i]
        if l_se_norm > 1:
            x = (l_se_norm - 1) / muscle.epsilon_ref
            return x * x * self.Fmax[i]
        return 0.0

    def _vce(self, i, lce, lmtu, act):
        """Contractile-element velocity, as vce_compute(...)[0]."""
        w = muscle.w_muscle
        l_se_norm = (lmtu - lce) / self.lslack[i]
        x = (l_se_norm - 1) / muscle.epsilon_ref
        f_se = x * x if l_se_norm > 1 else 0
        l_ce_norm = lce / self.lopt[i]
        x = 2 * (l_ce_norm - 1 + w) / w
        f_be = x * x if l_ce_norm - 1 + w < 0 else 0
        x = (l_ce_norm - 1) / w
        f_pe = x * x if l_ce_norm > 1 else 0
        x = abs(l_ce_norm - 1) / w
        f_ce = math.exp(muscle.c * (x * x * x))
        f_v = (f_se + f_be) / (f_pe + f_ce * act)
        K, N = muscle.K_muscle, muscle.N_muscle
        if f_v <= 1:
            v_norm = (f_v - 1) / (f_v * K + 1)
        elif f_v <= N:
            v_norm = ((f_v - N) / (N - 1) + 1) / (1 - 7.56 * K * (f_v - N) / (N - 1))
        else:
            v_norm = 0.01 * (f_v - N) + 1
        return v_norm * self.vmax[i] * self.lopt[i]

    def _leg_torques(self, ankle, knee, hip, d_ankle, d_knee, d_hip, F, o):
        """Ankle, knee and hip torques of one leg (muscles + joint limits); F[o:o+7] are its forces."""
        r, pm = self.r_TA_a
        T_TA = r * math.cos(ankle - pm) * F[o + TA]
        r, pm = self.r_GAS_a
        T_GAS_a = r * math.cos(ankle - pm) * F[o + GAS]
        r, pm = self.r_SOL_a
        T_SOL = r * math.cos(ankle - pm) * F[o + SOL]
        r, pm = self.r_GAS_k
        T_GAS_k = r * math.cos(knee - pm) * F[o + GAS]
        r, pm = self.r_VAS_k
        T_VAS = r * math.cos(knee - pm) * F[o + VAS]
        r, pm = self.r_HAM_k
        T_HAM_k = r * math.cos(knee - pm) * F[o + HAM]
        T_HAM_h = self.r_HAM_h * F[o + HAM]
        T_GLU = self.r_GLU_h * F[o + GLU]
        T_HFL = self.r_HFL_h * F[o + HFL]

        ankle_t = muscle.joint_limits(ANKLE, ankle, d_ankle) + T_GAS_a + T_SOL - T_TA
        knee_t = muscle.joint_limits(KNEE, knee, d_knee) + T_VAS - T_GAS_k - T_HAM_k
        hip_t = muscle.joint_limits(HIP, hip, d_hip) + T_GLU + T_HAM_h - T_HFL
        parts = [T_TA, T_GAS_a, T_GAS_k, T_SOL, T_VAS, T_HAM_k, T_HAM_h, T_GLU, T_HFL]
        return ankle_t, knee_t, hip_t, parts

    # ------------------------------------------------------------- every RK stage

    def joint_torques(self, mbs_data):
        """Fill Qq from the current neuromuscular state and the joint angles of this stage."""
        if self.use_kernels:
            kernels.joint_torques(mbs_data.q, mbs_data.qd, self.lce_a, self.J, self.k_mus, self.k_base, self.k_arc,
                                  self.k_hipc, self.k_lever, self.k_hlev, self.k_gen, mbs_data.Qq)
            return
        q = mbs_data.q.tolist()
        qd = mbs_data.qd.tolist()
        aL, kL, hL, aR, kR, hR = self._angles(q)
        daL, dkL, dhL, daR, dkR, dhR = self._angles(qd)

        lmtu = self._lmtu(aL, kL, hL) + self._lmtu(aR, kR, hR)
        lce = self.lce
        F = [self._force(i, lmtu[i], lce[i]) for i in range(14)]
        ankle_L, knee_L, hip_L, _ = self._leg_torques(aL, kL, hL, daL, dkL, dhL, F, 0)
        ankle_R, knee_R, hip_R, _ = self._leg_torques(aR, kR, hR, daR, dkR, dhR, F, 7)

        Qq = mbs_data.Qq
        Qq[self.jT_L] = pressure_sheet(q[self.jT_L], qd[self.jT_L])
        Qq[self.jT_R] = - pressure_sheet(-q[self.jT_R], -qd[self.jT_R])
        Qq[self.jA_L] = ankle_L
        Qq[self.jK_L] = - knee_L
        Qq[self.jH_L] = hip_L
        Qq[self.jA_R] = - ankle_R
        Qq[self.jK_R] = knee_R
        Qq[self.jH_R] = - hip_R

    # ----------------------------------------------------- once per accepted step

    def step(self, mbs_data, tsim):
        """Advance the neuromuscular state to the accepted state at time tsim."""
        if self.t_last is not None:
            if tsim <= self.t_last:      # user_dirdyn_loop runs twice at t0
                return
            if abs(tsim - self.t_last - self.dt) > 1e-9:
                raise RuntimeError("gait_controller needs a fixed-step integrator: step %g s instead of %g s"
                                   % (tsim - self.t_last, self.dt))
        k = self.k + 1
        self.k, self.t_last = k, tsim
        dt = self.dt
        r = self.reflex
        q = mbs_data.q.tolist()
        qd = mbs_data.qd.tolist()
        sensors = mbs_data.sensors

        # --- sensory signals
        P_hip, V_hip = sensors[self.s_hip].P, sensors[self.s_hip].V
        theta, dtheta = trunk_angle(P_hip, sensors[self.s_trunk].P, V_hip, sensors[self.s_trunk].V)
        ballL, heelL, ballR, heelR = (sensors[i].P[3] >= 0 for i in self.s_feet)  # z points down
        StanceL = 1 if (ballL or heelL) else 0
        StanceR = 1 if (ballR or heelR) else 0
        if tsim < 0.002:
            StanceL = 1

        self.Ldx = low_filter(q[self.jT_L], TAU_THIGH_LOAD, dt, self.Ldx)
        self.Rdx = low_filter(-q[self.jT_R], TAU_THIGH_LOAD, dt, self.Rdx)

        self.hist_thigh.push(k, (self.Ldx, self.Rdx))
        self.hist_trunk.push(k, (theta, dtheta))
        self.hist_stance.push(k, (StanceL, StanceR))

        # stance as seen by the controller: contact delayed by 10 ms
        if k >= self.n_m:
            dSL, dSR = self.hist_stance.at(k - self.n_m)
        else:
            dSL, dSR = 0, 0
        RonL, LonR = neural.lead(dSL, dSR, self.previous_stance, self.lead_counts, k == 0)
        self.previous_stance = (dSL, dSR)

        # knee hyperextension state, delayed by 10 ms
        kneeL, kneeR = -q[self.jK_L], q[self.jK_R]
        dkneeL, dkneeR = -qd[self.jK_L], qd[self.jK_R]
        if k >= self.n_m:
            ksL = kneeL - r.phi_k_off if (kneeL - r.phi_k_off) > 0 and dkneeL > 0 else 0
            ksR = kneeR - r.phi_k_off if (kneeR - r.phi_k_off) > 0 and dkneeR > 0 else 0
            self.hist_knee.push(k, (ksL, ksR))
            knee_mL, knee_mR = self.hist_knee.at(k - self.n_m)
        else:
            self.hist_knee.push(k, (0, 0))
            knee_mL = knee_mR = 0

        # --- delayed signals
        theta_s = dtheta_s = Ldx_s = Rdx_s = None
        F_s = lce_s = F_m = F_l = lce_l = [None] * 14
        if k > self.n_s:
            theta_s, dtheta_s = self.hist_trunk.at(k - self.n_s)
            Ldx_s, Rdx_s = self.hist_thigh.at(k - self.n_s)
            F_s, lce_s = self.hist_muscle.at(k - self.n_s)
        if k > self.n_m:
            F_m = self.hist_muscle.at(k - self.n_m)[0]
        if k > self.n_l:
            F_l, lce_l = self.hist_muscle.at(k - self.n_l)

        stance_L = dSL if k > self.n_m else 0
        stance_R = dSR if k > self.n_m else 0
        n_s, n_m, n_l = self.n_s, self.n_m, self.n_l
        StimL = neural.stimulations(stance_L, RonL, k, n_s, n_m, n_l, r, self.swing[0],
                                    theta_s, dtheta_s, Ldx_s, Rdx_s,
                                    F_s[HAM], F_s[GLU], lce_s[HAM], lce_s[HFL],
                                    F_m[VAS], knee_mL, F_l[SOL], F_l[GAS], lce_l[TA])
        StimR = neural.stimulations(stance_R, LonR, k, n_s, n_m, n_l, r, self.swing[1],
                                    theta_s, dtheta_s, Rdx_s, Ldx_s,
                                    F_s[7 + HAM], F_s[7 + GLU], lce_s[7 + HAM], lce_s[7 + HFL],
                                    F_m[7 + VAS], knee_mR, F_l[7 + SOL], F_l[7 + GAS], lce_l[7 + TA])
        stim = StimL + StimR

        # --- muscle dynamics: activation (first order, 10 ms) and contractile element
        if self.use_kernels:
            act_a, lmtu_a, lce_a, Fm_a = np.empty(14), np.empty(14), np.empty(14), np.empty(14)
            kernels.muscle_step(mbs_data.q, np.array(stim), dt, TAU_ACTIVATION, k == 0, self.lce_a, self.lmtu_a,
                                self.act_a, self.J, self.k_mus, self.k_base, self.k_arc, self.k_hipc, self.k_gen,
                                act_a, lmtu_a, lce_a, Fm_a)
            self.lce_a, self.lmtu_a, self.act_a = lce_a, lmtu_a, act_a
            act, lmtu, lce, Fm = act_a.tolist(), lmtu_a.tolist(), lce_a.tolist(), Fm_a.tolist()
        else:
            act, lmtu, lce, Fm = self._muscle_step(q, stim, k == 0)
        self.hist_muscle.push(k, (Fm, lce))
        self.lmtu, self.lce, self.act, self.stim, self.Fm = lmtu, lce, act, stim, Fm

        # effort as in the thesis: biarticular GAS and HAM counted at both joints
        Fm18 = [Fm[TA], Fm[GAS], Fm[GAS], Fm[SOL], Fm[VAS], Fm[HAM], Fm[HAM], Fm[GLU], Fm[HFL],
                Fm[7 + TA], Fm[7 + GAS], Fm[7 + GAS], Fm[7 + SOL], Fm[7 + VAS], Fm[7 + HAM], Fm[7 + HAM],
                Fm[7 + GLU], Fm[7 + HFL]]
        self.total_fm += dt * np.sum(Fm18) / 21000

        if self.flag_graph:
            self._collect_graph(q, qd, Fm18, (StanceL, StanceR), tsim)

        if k > 0 and k % self.n_measure == 0:
            self._measure(mbs_data, tsim, theta, Fm18)

    def _muscle_step(self, q, stim, first):
        """Activation, muscle-tendon length, contractile length and force after one step."""
        dt = self.dt
        lmtu = self._lmtu_legs(q)
        if first:
            act = list(stim)
            lce = [l - ls for l, ls in zip(lmtu, self.lslack)]
        else:
            act = [low_filter(s, TAU_ACTIVATION, dt, a) for s, a in zip(stim, self.act)]
            lce = []
            for i in range(14):
                v0 = self._vce(i, self.lce[i], self.lmtu[i], self.act[i])
                v1 = self._vce(i, self.lce[i], lmtu[i], act[i])
                lce.append(self.lce[i] + 0.5 * (v0 + v1) * dt)
        Fm = [self._force(i, lmtu[i], lce[i]) for i in range(14)]
        return act, lmtu, lce, Fm

    def _collect_graph(self, q, qd, Fm18, stance, tsim):
        aL, kL, hL, aR, kR, hR = self._angles(q)
        daL, dkL, dhL, daR, dkR, dhR = self._angles(qd)
        partsL = self._leg_torques(aL, kL, hL, daL, dkL, dhL, self.Fm, 0)[3]
        partsR = self._leg_torques(aR, kR, hR, daR, dkR, dhR, self.Fm, 7)[3]
        a = self.act
        act = [a[TA], a[GAS], a[SOL], a[VAS], a[HAM], a[GLU], a[HFL],
               a[7 + TA], a[7 + GAS], a[7 + SOL], a[7 + VAS], a[7 + HAM], a[7 + GLU], a[7 + HFL]]
        gait_graph.collect_muscle(partsL + partsR, Fm18, act, self.stim, list(stance), tsim, self.dt, self.tf)

    def finish(self, mbs_data):
        if self.flag_graph and self.t_last is not None:
            gait_graph.show_ext(self.t_last, self.dt)

    # ------------------------------------------------------- fitness, every 0.1 s

    def _measure(self, mbs_data, tsim, theta, Fm18):
        model = mbs_data.user_model
        P_hip = mbs_data.sensors[self.s_hip].P
        now = datetime.now()
        print("\n", round(tsim, 2), " fitness ", round(model["fitness"]), " ct:", now.strftime("%H:%M:%S"),
              "I", int((now - self.prev_datetime).total_seconds()), "s")
        print("speed", round(abs(P_hip[1] / tsim), 3))
        self.prev_datetime = now
        if not self.flag_fitness:
            return

        model["fitness"] -= 1                                           # survived time
        model["fitness"] += (self.total_fm / tsim) / 4                  # effort
        model["fitness"] += abs(P_hip[1] - tsim * TARGET_SPEED) / 4     # distance to the target speed

        index = round(tsim / MEASURE_PERIOD)
        model["fitness_memory"][index] = model["fitness"]
        model["fm_memory"][index] = np.sum(Fm18) / 10000
        print("memory fitness ", round(model["best_fitness_memory"][index], 3), round(model["fitness_memory"][index], 3))
        np.save("fitness_id" + str(self.id), model["fitness"])
        np.save("fitness_memory" + str(self.id), np.append(model["fitness_memory"], [0], axis=0))  # last digit flags an early stop

        if model["best_fitness_memory"][index] + 2 < model["fitness_memory"][index]:
            print("DISQUALIFIED: fitness too high compared to baseline.  Baseline : ",
                  model["best_fitness_memory"][index], model["fitness_memory"][index])
            np.save("fitness_memory" + str(self.id), np.append(model["fitness_memory"], [1], axis=0))
            self._stop(mbs_data, "baseline")
        if abs(P_hip[1] - tsim * TARGET_SPEED) > self.speed_window:
            print("DISQUALIFIED: Outside allowed area", flush=True)
            self._stop(mbs_data, "area")
        if P_hip[3] > -0.75:
            print("DISQUALIFIED: Hip too low", P_hip[3], flush=True)
            self._stop(mbs_data, "hip")
        if theta < 0 or theta > 0.5:
            print("DISQUALIFIED: trunk angle outside allowed range", flush=True)
            self._stop(mbs_data, "trunk")

    def _stop(self, mbs_data, reason):
        if self.stop_reason is None:
            self.stop_reason = reason
        mbs_data.flag_stop = 1  # ends the simulation after this step
