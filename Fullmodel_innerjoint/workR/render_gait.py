#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render a Robotran .anim file as a stick-figure GIF (no MBsysPad needed).

Body positions come from Robotran's own sensor routines (symbolicR), so the
drawing uses exactly the model's kinematics.

    python render_gait.py                                   # ../animationR/dirdyn_q.anim -> ../animationR/gait.gif
    python render_gait.py --anim ../animationR/optimised.anim --out optimised.gif --t1 5
"""
import argparse
import contextlib
import io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, "User_function"))
sys.path.insert(1, os.path.join(parent_dir, "userfctR"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--anim", default="../animationR/dirdyn_q.anim", help=".anim (or dirdyn_q.res) file to render")
ap.add_argument("--out", default="../animationR/gait.gif", help="output GIF")
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--t0", type=float, default=0.0, help="start time [s]")
ap.add_argument("--t1", type=float, default=None, help="end time [s] (default: end of file)")
ap.add_argument("--title", default="")
args = ap.parse_args()

with contextlib.redirect_stdout(io.StringIO()):  # the user modules print a lot on import
    import MBsysPy as Robotran
    mbs_data = Robotran.MbsData("../dataR/Fullmodel_innerjoint.mbs")
    mbs_data.__load_symbolic_fct__(os.path.dirname(Robotran.mbsyspy.mbs_data.__file__),
                                   ["sensor", "gensensor"], mbs_data.symbolic_path)

# Named sensors of the model, plus the origin of each body frame (= joint location)
named = {"trunk_top": "Sensor_trunk", "hip": "Sensor_hip", "ballL": "Sensor_BallL", "heelL": "Sensor_HeelL",
         "ballR": "Sensor_BallR", "heelR": "Sensor_HeelR"}
joints = {"ankleL": "ankleL", "kneeL": "kneeL", "thighL": "innerthighL", "hipR": "hipR",
          "thighR": "innerthighR", "kneeR": "kneeR", "ankleR": "ankleR"}

q = np.loadtxt(args.anim)
t = q[:, 0]
t1 = t[-1] if args.t1 is None else args.t1
pick = np.searchsorted(t, np.arange(args.t0, t1 + 1e-9, 1.0 / args.fps))
pick = pick[pick < len(t)]

sensor = Robotran.MbsSensor(mbs_data)
P = {k: np.zeros((len(pick), 2)) for k in list(named) + list(joints)}
for n, i in enumerate(pick):
    mbs_data.q[1:] = q[i, 1:]
    mbs_data.qd[1:] = 0.0
    for key, name in named.items():
        sensor.comp_s_sensor(mbs_data.sensor_id[name])
        P[key][n] = (sensor.P[1], -sensor.P[3])  # gravity is +z in this model: height = -z
    for key, name in joints.items():
        sensor.comp_gen_sensor(mbs_data.joint_id[name])
        P[key][n] = (sensor.P[1], -sensor.P[3])
tt = t[pick]

chains = [(["heelL", "ballL", "ankleL", "heelL", "ankleL", "kneeL", "thighL", "hipR"], "#d1495b", 3, "left leg"),
          (["hipR", "thighR", "kneeR", "ankleR", "heelR", "ballR", "ankleR"], "#00798c", 3, "right leg"),
          (["hipR", "trunk_top"], "#30343f", 5, "trunk")]

fig, ax = plt.subplots(figsize=(8, 4.2), dpi=100)
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.9)
lines = [ax.plot([], [], "-o", color=c, lw=w, ms=3, label=label, solid_capstyle="round")[0] for _, c, w, label in chains]
trail, = ax.plot([], [], ":", color="#888888", lw=1)
info = ax.text(0.01, 0.95, "", transform=ax.transAxes, va="top", family="monospace", fontsize=9)
ax.axhline(0, color="#5c5c5c", lw=1.5)
ax.set_aspect("equal")
ax.set_ylim(-0.1, 1.9)
ax.set_xlabel("x [m]")
ax.set_ylabel("height [m]")
ax.legend(loc="upper right", fontsize=8, frameon=False)
if args.title:
    ax.set_title(args.title, fontsize=10)


def draw(n):
    for line, (chain, _, _, _) in zip(lines, chains):
        line.set_data([P[k][n, 0] for k in chain], [P[k][n, 1] for k in chain])
    trail.set_data(P["hipR"][:n + 1, 0], P["hipR"][:n + 1, 1])
    x = P["hipR"][n, 0]
    ax.set_xlim(x - 1.6, x + 1.6)
    ax.set_xticks(np.arange(np.floor(x - 1.6), np.ceil(x + 1.6) + 0.01, 0.5))
    speed = x / tt[n] if tt[n] > 0 else 0.0
    info.set_text("t = %5.2f s   x_hip = %5.2f m   mean speed = %4.2f m/s" % (tt[n], x, speed))
    return lines + [trail, info]


FuncAnimation(fig, draw, frames=len(tt)).save(args.out, writer=PillowWriter(fps=args.fps))
print("Wrote %s (%d frames, %.1f s)" % (args.out, len(tt), tt[-1] - tt[0]))
