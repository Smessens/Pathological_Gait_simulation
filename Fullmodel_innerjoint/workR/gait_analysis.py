#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gait metrics of recorded runs and the young/aged comparison of the thesis (4.2.4).

Input: .npz records written by run_gait.py --record (state every 1 ms). Like the
thesis, strides run from heel strike to heel strike, the first 3 strides of each leg
are dropped, each stride is resampled on 0-100 % and the metrics are taken from the
mean stride (the mean of the left and right legs' mean strides).

Conventions (clinical): hip flexion angle between trunk and thigh (0 = aligned,
+ = flexion), knee flexion angle (0 = straight), ankle dorsiflexion angle (0 = foot
perpendicular to the shank, + = dorsiflexion). Angles come from the joint coordinates,
calibrated against the segment geometry of the model. Moments are the joint torques
applied by muscles and joint limits (internal moments); power = torque x joint
angular velocity (+ = generation).

    python gait_analysis.py young.npz aged10.npz aged13.npz --labels Young "Old 1.0 m/s" "Old 1.3 m/s" \
        --out aging/                       # tables (Markdown, JSON) and figures (SVG)
"""
import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np

WORKR = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(WORKR)
DROP_STRIDES = 3


# ---------------------------------------------------------------- model geometry

def load_model():
    sys.path[:0] = [os.path.join(PROJECT, "User_function"), os.path.join(PROJECT, "userfctR")]
    cwd = os.getcwd()
    os.chdir(WORKR)
    with contextlib.redirect_stdout(io.StringIO()):
        import MBsysPy as Robotran
        mbs = Robotran.MbsData("../dataR/Fullmodel_innerjoint.mbs")
        mbs.__load_symbolic_fct__(os.path.dirname(Robotran.mbsyspy.mbs_data.__file__), ["sensor", "gensensor"],
                                  mbs.symbolic_path)
    os.chdir(cwd)
    return mbs, Robotran.MbsSensor(mbs)


def points(mbs, sensor, q):
    """Planar positions (x forward, y up) of the landmarks for joint coordinates q (1..n)."""
    mbs.q[1:] = q
    mbs.qd[1:] = 0.0
    out = {}
    for key, name in (("trunk_top", "Sensor_trunk"), ("heel_L", "Sensor_HeelL"), ("ball_L", "Sensor_BallL"),
                      ("heel_R", "Sensor_HeelR"), ("ball_R", "Sensor_BallR")):
        sensor.comp_s_sensor(mbs.sensor_id[name])
        out[key] = np.array([sensor.P[1], -sensor.P[3]])
    # body-frame origins: the tree starts at the left foot (Fig 2.3 of the thesis)
    for key, joint in (("hip", "hipR"), ("knee_L", "kneeL"), ("ankle_L", "ankleL"),
                       ("knee_R", "kneeR"), ("ankle_R", "ankleR")):
        sensor.comp_gen_sensor(mbs.joint_id[joint])
        out[key] = np.array([sensor.P[1], -sensor.P[3]])
    return out


def signed_angle(a, b):
    """Counter-clockwise angle from a to b [rad]."""
    return np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])


def geometric_angles(p, side):
    """Clinical hip flexion, knee flexion and ankle dorsiflexion of one leg [deg]."""
    trunk_down = p["hip"] - p["trunk_top"]
    thigh = p["knee_" + side] - p["hip"]
    shank = p["ankle_" + side] - p["knee_" + side]
    foot = p["ball_" + side] - p["heel_" + side]
    hip = signed_angle(trunk_down, thigh)
    knee = -signed_angle(thigh, shank)
    ankle = np.pi / 2 - signed_angle(foot, -shank)
    return np.degrees([hip, knee, ankle])


# ---------------------------------------------------------------- one record

class Record:
    def __init__(self, path, model, span=None):
        r = np.load(path)
        self.path = path
        self.data = r["data"]
        if span is not None:  # analyse only samples with span[0] <= t < span[1]
            self.data = self.data[(self.data[:, 0] >= span[0]) & (self.data[:, 0] < span[1])]
        self.columns = [str(c) for c in r["columns"]]
        self.joints = json.loads(str(r["joints"]))
        self.parameters = json.loads(str(r["parameters"]))
        self.stop_reason = str(r["stop_reason"])
        self.t = self.col("t")
        nq = sum(c.startswith("q") and not c.startswith("qd") for c in self.columns)
        self.q = np.column_stack([self.col("q%d" % j) for j in range(1, nq + 1)])
        self.qd = np.column_stack([self.col("qd%d" % j) for j in range(1, nq + 1)])
        self.Qq = np.column_stack([self.col("Qq%d" % j) for j in range(1, nq + 1)])
        self._calibrate(*model)
        self._events()

    def col(self, name):
        return self.data[:, self.columns.index(name)]

    def _calibrate(self, mbs, sensor):
        """Clinical angle = sign * q_joint + offset, fitted on the model geometry."""
        sample = np.linspace(0, len(self.t) - 1, 400).astype(int)
        geo = {"L": [], "R": []}
        for i in sample:
            p = points(mbs, sensor, self.q[i])
            for side in "LR":
                geo[side].append(geometric_angles(p, side))
        self.angle, self.moment, self.power, self.fit_error = {}, {}, {}, {}
        for side in "LR":
            g = np.array(geo[side])
            for k, joint in enumerate(("hip", "knee", "ankle")):
                j = self.joints[joint + side] - 1
                qj = np.degrees(self.q[sample, j])
                sign = 1.0 if np.corrcoef(qj, g[:, k])[0, 1] > 0 else -1.0
                offset = np.mean(g[:, k] - sign * qj)
                self.fit_error[joint + side] = float(np.max(np.abs(g[:, k] - (sign * qj + offset))))
                self.angle[joint + side] = sign * np.degrees(self.q[:, j]) + offset
                self.moment[joint + side] = sign * self.Qq[:, j]  # + in the + angle direction
                self.power[joint + side] = self.Qq[:, j] * self.qd[:, j]

    def _events(self, min_gap=0.05):
        """Initial contact (heel or ball, normally the heel) and toe-off of each foot.

        Contact or air phases shorter than min_gap [s] are ignored (bounces)."""
        self.strikes, self.toe_offs, self.contact = {}, {}, {}
        n_gap = int(round(min_gap / np.median(np.diff(self.t))))
        for side in "LR":
            contact = (self.col("heel%s_z" % side) >= 0) | (self.col("ball%s_z" % side) >= 0)  # z points down
            for value in (False, True):  # fill short air gaps, then drop short contacts
                edges = np.flatnonzero(np.diff(np.r_[False, contact == value, False].astype(int)))
                for a, b in zip(edges[::2], edges[1::2]):
                    if b - a < n_gap and a > 0 and b < len(contact):
                        contact[a:b] = not value
            self.contact[side] = contact
            change = np.diff(contact.astype(int))
            self.strikes[side] = list(np.flatnonzero(change == 1) + 1)
            self.toe_offs[side] = list(np.flatnonzero(change == -1) + 1)

    def strides(self, side):
        hs = self.strikes[side][DROP_STRIDES:]
        return list(zip(hs[:-1], hs[1:]))

    def mean_stride(self, signal, side):
        rows = []
        for a, b in self.strides(side):
            x = np.linspace(0, 100, b - a)
            rows.append(np.interp(np.arange(101), x, signal[a:b]))
        return np.mean(rows, axis=0)

    def mean_both(self, quantity, joint):
        """Mean stride of a joint quantity: average of the left and right legs' mean strides."""
        q = getattr(self, quantity)
        return 0.5 * (self.mean_stride(q[joint + "L"], "L") + self.mean_stride(q[joint + "R"], "R"))

    LIMIT_SIGN = {("ankle", "L"): 1.0, ("knee", "L"): -1.0, ("hip", "L"): 1.0,   # muscle-model angle = sign * q
                  ("ankle", "R"): -1.0, ("knee", "R"): 1.0, ("hip", "R"): -1.0}

    def limit_power(self):
        """Mean |power| of the joint-limit torques per joint, both legs [W] (as limit_work_weight),
        and its parts while the foot is on the ground ("<joint>_stance") and in the air ("<joint>_swing")."""
        import Muscle_actuation_layer as muscle
        out = {}
        for k, joint in enumerate(("ankle", "knee", "hip")):
            stance = swing = 0.0
            for side in "LR":
                sign, j = self.LIMIT_SIGN[(joint, side)], self.joints[joint + side] - 1
                phi, dphi = sign * self.q[:, j], sign * self.qd[:, j]
                power = np.abs([muscle.joint_limits(k, a, d) * d for a, d in zip(phi, dphi)])
                stance += np.mean(power * self.contact[side])
                swing += np.mean(power * ~self.contact[side])
            out[joint], out[joint + "_stance"], out[joint + "_swing"] = float(stance + swing), float(stance), float(swing)
        return out

    def knee_limit(self):
        """Knee absorption peak [W], its % of stride, and the part of it due to the knee's joint limit.

        The limit is Geyer's hyperextension torque (Muscle_actuation_layer.joint_limits), a
        function of the knee angle in the muscle model's convention (-q left, +q right)."""
        import Muscle_actuation_layer as muscle
        total, limit = [], []
        for side, sign in (("L", -1.0), ("R", 1.0)):
            j = self.joints["knee" + side] - 1
            torque = np.array([muscle.joint_limits(1, a, d) for a, d in zip(sign * self.q[:, j], sign * self.qd[:, j])])
            total.append(self.mean_stride(self.Qq[:, j] * self.qd[:, j], side))
            limit.append(self.mean_stride(sign * torque * self.qd[:, j], side))
        total, limit = 0.5 * (total[0] + total[1]), 0.5 * (limit[0] + limit[1])
        i = int(np.argmin(total))
        return float(total[i]), i, float(limit[i])

    def toe_off_percent(self):
        values = []
        for side in "LR":
            to = np.array(self.toe_offs[side])
            for a, b in self.strides(side):
                inside = to[(to > a) & (to < b)]
                if len(inside):
                    values.append(100.0 * (inside[0] - a) / (b - a))
        return float(np.mean(values))

    def clearances(self):
        """Clearance of every swing (first 3 strides dropped) [m], as gait_controller scores it:
        lowest point of the foot (heel or ball) over the middle half of the air phase, contacts
        shorter than 50 ms inside it (scuffs) counting as zero."""
        out = []
        for side in "LR":
            low = np.maximum(0.0, -np.maximum(self.col("heel%s_z" % side), self.col("ball%s_z" % side)))
            air = ~self.contact[side]
            edges = np.flatnonzero(np.diff(np.r_[False, air, False].astype(int)))
            first = self.strikes[side][DROP_STRIDES] if len(self.strikes[side]) > DROP_STRIDES else len(air)
            for a, b in zip(edges[::2], edges[1::2]):
                if a < first or self.t[b - 1] - self.t[a] < 0.15:
                    continue
                n = b - a
                out.append(float(low[a + n // 4: a + 3 * n // 4].min()))
        return np.array(out)

    def basic(self):
        dur, length = [], []
        x = self.col("hip_x")
        for side in "LR":
            for a, b in self.strides(side):
                dur.append(self.t[b] - self.t[a])
                length.append(x[b] - x[a])
        dur, length = np.mean(dur), np.mean(length)
        clearance = self.clearances()
        return {"speed": length / dur, "stride_frequency": 1 / dur, "stride_length": length,
                "clearance": float(clearance.mean()), "clearance_sd": float(clearance.std()),
                "clearance_min": float(clearance.min()),
                "cadence": 120 / dur, "step_length": length / 2,
                "strides": sum(len(self.strides(s)) for s in "LR"), "duration": float(self.t[-1])}

    def metrics(self):
        to = self.toe_off_percent()
        i_to = int(round(to))
        m = {"toe_off_percent": to}
        hip, knee, ankle = (self.mean_both("angle", j) for j in ("hip", "knee", "ankle"))
        m.update({"hip_heel_strike_angle": hip[0], "hip_peak_flexion": hip.max(), "hip_peak_extension": hip.min(),
                  "hip_rom": hip.max() - hip.min()})
        m.update({"knee_heel_strike_angle": knee[0], "knee_peak_flexion_swing": knee[i_to:].max(),
                  "knee_rom": knee.max() - knee.min()})
        m.update({"ankle_heel_strike_angle": ankle[0], "ankle_toe_off_angle": ankle[i_to],
                  "ankle_peak_plantarflexion": ankle.min(), "ankle_rom": ankle.max() - ankle.min()})
        for joint, plus, minus in (("hip", "flexion", "extension"), ("knee", "flexion", "extension"),
                                   ("ankle", "dorsiflexion", "plantarflexion")):
            mom = self.mean_both("moment", joint)
            pw = self.mean_both("power", joint)
            m["%s_%s_moment" % (joint, plus)] = mom.max()
            m["%s_%s_moment" % (joint, minus)] = -mom.min()
            m["%s_power_generation" % joint] = pw.max()
            m["%s_power_absorption" % joint] = pw.min()
        return {k: float(v) for k, v in m.items()}


# ---------------------------------------------------------------- young/old trends

# Trends of Boyer et al. (2017) as listed in the thesis' Tables 4.2-4.4 ("young ... than
# older"), as the sign of (old - young) that passes, in the conventions above (absorption
# is negative, so a larger absorption in the young is old - young > 0). The knee ROM row
# reads "young ROM less than older" but is scored in the thesis as young > old (both old
# versions, with a smaller ROM, are green); that scoring is kept.
TRENDS = [
    ("hip_heel_strike_angle", "Hip heel-strike angle", "young more extended", +1),
    ("hip_peak_flexion", "Hip peak flexion", "young more extended", +1),
    ("hip_peak_extension", "Hip peak extension", "young more extended", +1),
    ("hip_rom", "Hip range of motion", "young smaller", +1),
    ("hip_flexion_moment", "Hip flexion moment", "young smaller", +1),
    ("hip_extension_moment", "Hip extension moment", "young smaller", +1),
    ("hip_power_generation", "Hip power generation", "young smaller", +1),
    ("hip_power_absorption", "Hip power absorption", "young greater", +1),
    ("knee_heel_strike_angle", "Knee heel-strike angle", "young more extended", +1),
    ("knee_peak_flexion_swing", "Knee peak flexion (swing)", "young more flexed", -1),
    ("knee_rom", "Knee range of motion", "young greater (as scored)", -1),
    ("knee_flexion_moment", "Knee flexion moment", "young greater", -1),
    ("knee_extension_moment", "Knee extension moment", "young greater", -1),
    ("knee_power_generation", "Knee power generation", "young greater", -1),
    ("knee_power_absorption", "Knee power absorption", "young greater", +1),
    ("ankle_heel_strike_angle", "Ankle heel-strike angle", "young more dorsiflexed", -1),
    ("ankle_toe_off_angle", "Ankle toe-off angle", "young more plantarflexed", +1),
    ("ankle_peak_plantarflexion", "Ankle peak plantarflexion", "young more plantarflexed", +1),
    ("ankle_rom", "Ankle range of motion", "young greater", -1),
    ("ankle_dorsiflexion_moment", "Ankle dorsiflexion moment", "young greater", -1),
    ("ankle_plantarflexion_moment", "Ankle plantarflexion moment", "young smaller", +1),
    ("ankle_power_generation", "Ankle power generation", "young smaller", +1),
    ("ankle_power_absorption", "Ankle power absorption", "young greater", +1),
]
# DeVita & Hortobagyi (2000): at the same speed older adults use the ankle plantar flexors
# less (-23 % angular impulse, -29 % work), the opposite of the thesis' two ankle rows.
DEVITA_FLIPS = {"ankle_plantarflexion_moment": ("young greater", -1),
                "ankle_power_generation": ("young greater", -1)}
# Differences within these bands count as no difference: the two halves (0-30 s, 30-60 s) of
# the 60 s young run differ by up to 1.0 deg in angles and 12 % in peak powers.
ANGLE_TOLERANCE = 1.5                  # deg
RELATIVE_TOLERANCE = 0.15              # of the young value, for moments and powers
FLOORS = {"Nm": 2.0, "W": 5.0}


def unit(key):
    if key.endswith("moment"):
        return "Nm"
    if "power" in key:
        return "W"
    return "deg"


def tolerance(key, young):
    u = unit(key)
    return ANGLE_TOLERANCE if u == "deg" else max(RELATIVE_TOLERANCE * abs(young), FLOORS[u])


def trend_table(young, old, flips=None):
    """Rows (label, trend, young, old, result, unit) with result 'yes', 'no' or '~' (within tolerance)."""
    rows, counts = [], {"yes": 0, "no": 0, "~": 0}
    for key, label, text, direction in TRENDS:
        if flips and key in flips:
            text, direction = flips[key]
        diff = old[key] - young[key]
        result = "~" if abs(diff) <= tolerance(key, young[key]) else ("yes" if diff * direction > 0 else "no")
        counts[result] += 1
        rows.append((label, text, young[key], old[key], result, unit(key)))
    return rows, counts


# ---------------------------------------------------------------- output

def figure(records, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]  # validated categorical slots
    styles = ["-", "--", ":", "-."]
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.5), sharex=True)
    titles = {"angle": ("Angle", "deg"), "moment": ("Moment", "Nm"), "power": ("Power", "W")}
    signs = {"hip": "flexion +", "knee": "flexion +", "ankle": "dorsiflexion +"}
    x = np.arange(101)
    for r, joint in enumerate(("hip", "knee", "ankle")):
        for c, quantity in enumerate(("angle", "moment", "power")):
            ax = axes[r, c]
            for rec, label, color, style in zip(records, labels, colors, styles):
                ax.plot(x, rec.mean_both(quantity, joint), style, color=color, lw=2, label=label)
                ax.axvline(rec.toe_off_percent(), color=color, lw=1, ls=style, alpha=0.6)
            ax.axhline(0, color="#9a9893", lw=0.8)
            name, u = titles[quantity]
            ax.set_title("%s %s (%s)" % (joint.capitalize(), name.lower(),
                                         signs[joint] if quantity != "power" else "generation +"), fontsize=10)
            ax.set_ylabel(u)
            ax.grid(color="#e4e3de", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("% of stride, from heel strike (vertical lines: toe-off); mean of both legs", fontsize=10)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=len(names), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("records", nargs="+", help=".npz records; the first is the young reference")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--out", default=None, help="folder for results.md, metrics.json and gait_curves.svg")
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(p) for p in args.records]

    model = load_model()
    records = [Record(p, model) for p in args.records]
    basics = [r.basic() for r in records]
    metrics = [r.metrics() for r in records]

    lines = ["| | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for key, name, fmt in (("speed", "Speed (m/s)", "%.3f"), ("stride_frequency", "Stride frequency (strides/s)", "%.3f"),
                           ("stride_length", "Stride length (m)", "%.3f"), ("cadence", "Cadence (steps/min)", "%.1f"),
                           ("step_length", "Step length (m)", "%.3f"), ("strides", "Strides analysed", "%d"),
                           ("duration", "Simulated time (s)", "%.1f")):
        lines.append("| %s | %s |" % (name, " | ".join(fmt % b[key] for b in basics)))
    lines.append("| Toe-off (%% of stride) | %s |" % " | ".join("%.1f" % m["toe_off_percent"] for m in metrics))
    lines.append("| Mid-swing foot clearance, mean (sd; lowest) (mm) | %s |" % " | ".join(
        "%.1f (%.1f; %.1f)" % (1000 * b["clearance"], 1000 * b["clearance_sd"], 1000 * b["clearance_min"]) for b in basics))
    out = ["## Basic metrics (thesis Table 4.1)", ""] + lines + [""]

    out += ["## Joint metrics and trends (thesis Tables 4.2-4.4)", "",
            "Pass (yes) = the old version differs from the young one in the direction of the trend by more than "
            "the tolerance (1.5 deg for angles, 15 % of the young value with floors of 2 Nm and 5 W for moments "
            "and powers, just above the differences between the two halves of the young run); ~ = within the "
            "tolerance.", ""]
    for rec, label, m in list(zip(records, labels, metrics))[1:]:
        for title, flips in (("trends as listed in the thesis", None),
                             ("ankle plantar-flexor rows as in DeVita & Hortobagyi (2000)", DEVITA_FLIPS)):
            rows, counts = trend_table(metrics[0], m, flips)
            out += ["### %s vs %s: %d/23 pass, %d within tolerance (%s)" % (label, labels[0], counts["yes"], counts["~"], title),
                    "", "| Metric | Trend | %s | %s | Pass |" % (labels[0], label), "|---|---|---|---|---|"]
            out += ["| %s | %s | %.1f %s | %.1f %s | %s |" % (n, t, y, u, o, u, res) for n, t, y, o, res, u in rows]
            out.append("")
    limits = [r.limit_power() for r in records]
    out += ["## Work of the joint limits (mean |torque x joint speed|, both legs)", "",
            "| | Hip | Knee | Knee, foot on the ground | Knee, foot in the air | Ankle |", "|---|---|---|---|---|---|"]
    for lp, label in zip(limits, labels):
        out.append("| %s | %.1f W | %.1f W | %.1f W | %.1f W | %.1f W |"
                   % (label, lp["hip"], lp["knee"], lp["knee_stance"], lp["knee_swing"], lp["ankle"]))
    out.append("")
    out += ["## Knee absorption and the knee's joint limit", "",
            "| | Absorption peak | % of stride | From the hyperextension limit |", "|---|---|---|---|"]
    for rec, label in zip(records, labels):
        peak, at, limit = rec.knee_limit()
        out.append("| %s | %.0f W | %d | %.0f W (%.0f %%) |" % (label, peak, at, limit, 100 * limit / peak))
    out.append("")
    out += ["Calibration of joint angles against the model geometry (max error, deg): "
            + ", ".join("%s %s" % (label, max(r.fit_error.values()).__format__(".3f")) for r, label in zip(records, labels)),
            ""]
    text = "\n".join(out)
    print(text)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "results.md"), "w") as f:
            f.write(text)
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump({label: {"basic": b, "joint": m, "limit_power": lp, "record": os.path.basename(r.path),
                               "stop": r.stop_reason}
                       for label, b, m, lp, r in zip(labels, basics, metrics, limits, records)}, f, indent=2)
        figure(records, labels, os.path.join(args.out, "gait_curves.svg"))


if __name__ == "__main__":
    main()
