#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay one optimized gait from a fitness_data log.

Builds the same parameter set as fitness_calculator() in reflex-CMAES.py and runs
the Robotran direct dynamics. Results are written as usual to resultsR/*.res and
animationR/dirdyn_q.anim (view it in MBsysPad, or turn it into a GIF with
render_gait.py).

    python run_gait.py                        # best gait of fitness_data/retuned_tf10, 10 s
    python run_gait.py --tf 30                # same gait, 30 s
    python run_gait.py --fitness              # with the optimizer's fitness and disqualification checks
    python run_gait.py --log aged13_tf10 --tf 60 --record aged13.npz --no-files   # 60 s, sampled for gait_analysis.py

A log's model settings (aged muscles, target speed) are read from
fitness_data/<log>settings.json when optimize_gains.py wrote one.

The 2024 logs (compact_tf10, ...) were tuned before the neuromuscular timing was fixed
(see gait_controller.py); their gains no longer walk, but they can still be replayed
with --log and --row.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, "User_function"))
sys.path.insert(1, os.path.join(parent_dir, "userfctR"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import MBsysPy as Robotran

# Order of the suggestion vectors (specific_parameters in reflex-CMAES.py, KEYS in optimize_gains.py)
PARAMETER_KEYS = ['G_VAS', 'G_SOL', 'G_GAS', 'G_TA', 'G_SOL_TA', 'G_HAM', 'G_GLU', 'G_HFL', 'G_HAM_HFL',
                  'G_delta_theta', 'theta_ref']

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--log", default="retuned_tf10", help="fitness_data log name (default: retuned_tf10)")
ap.add_argument("--row", default="best", help="row of the log, or 'best' (lowest fitness) (default: best)")
ap.add_argument("--tf", type=float, default=10, help="simulated time [s] (default: 10)")
ap.add_argument("--dt", type=float, default=1000e-7, help="integration step [s] (default: 1e-4, as optimized)")
ap.add_argument("--fitness", action="store_true", help="enable fitness bookkeeping and early disqualification")
ap.add_argument("--graph", action="store_true", help="collect gait_graph data (saved to numpy_archive/)")
ap.add_argument("--aged", action="store_true", help="aged muscles (gait_controller.AGED), for logs without settings")
ap.add_argument("--target-speed", type=float, default=None, help="speed of the fitness [m/s] (default: the log's)")
ap.add_argument("--record", default=None, help="save the state every 1 ms to this .npz (for gait_analysis.py)")
ap.add_argument("--no-files", action="store_true", help="do not write resultsR/*.res and the .anim")
args = ap.parse_args()

suggestions = np.load("fitness_data/" + args.log + "memory_suggestion.npy", allow_pickle=True)
fitnesses = np.load("fitness_data/" + args.log + "memory_fitness.npy", allow_pickle=True).astype(float)
row = int(np.argmin(fitnesses)) if args.row == "best" else int(args.row)
if len(suggestions[row]) != len(PARAMETER_KEYS):
    sys.exit("Log '%s' stores %d parameters per row; only %d-parameter logs (like retuned_tf10) are supported."
             % (args.log, len(suggestions[row]), len(PARAMETER_KEYS)))
suggestion = dict(zip(PARAMETER_KEYS, suggestions[row]))
print("Replaying %s row %d (fitness %.3f when optimized) for %g s" % (args.log, row, fitnesses[row], args.tf), flush=True)

model = {}
settings_file = "fitness_data/" + args.log + "settings.json"
if os.path.exists(settings_file):
    with open(settings_file) as f:
        model = json.load(f)["model"]
if args.aged:
    import gait_controller
    model.update(gait_controller.AGED)
if args.target_speed is not None:
    model["target_speed"] = args.target_speed
if model:
    print("Model settings:", model)

n_memory = max(200, int(round(args.tf / 0.1)) + 2)  # one fitness entry every 0.1 s
parameters = {
    "dt": args.dt,
    "tf": args.tf,
    "flag_graph": args.graph,
    "id": 0,

    "flag_fitness": args.fitness,
    "best_fitness_memory": np.ones(n_memory) * 10 * args.tf,
    "fitness_memory": np.ones(n_memory) * 10 * args.tf,
    "fm_memory": np.zeros(n_memory),
    "fitness": 10 * args.tf,

    "k_swing": 0.25,
    "k_p": 1.909859317102744,
    "k_d": 0.2,
    "phi_k_off": 2.967059728390360,
    "loff_TA": 0.72,
    "loff_HAM": 0.85,
    "loff_HFL": 0.65,
}
parameters.update(model)
parameters.update(suggestion)
if args.record:
    parameters["record_file"] = os.path.abspath(args.record)
if args.no_files:
    parameters["flag_outputs"] = False

mbs_data = Robotran.MbsData('../dataR/Fullmodel_innerjoint.mbs')
mbs_data.process = 1
mbs_part = Robotran.MbsPart(mbs_data)
mbs_part.set_options(rowperm=1, verbose=1)
mbs_part.run()

mbs_data.process = 3
mbs_dirdyn = Robotran.MbsDirdyn(mbs_data)
mbs_data.user_model = parameters
mbs_dirdyn.set_options(dt0=args.dt, tf=args.tf, save2file=0 if args.no_files else 1)
if args.no_files:
    mbs_dirdyn.store_results = False

start = time.time()
try:
    mbs_dirdyn.run()
except Exception as e:  # numerical failure, e.g. a fallen model simulated without --fitness
    print("Simulation stopped early:", e)
print("Wall time: %.1f min" % ((time.time() - start) / 60))
controller = mbs_data.gait_controller
if controller.stop_reason:  # area: left the 1.3 m/s window, hip: fell, trunk: trunk angle, baseline: fitness
    print("Disqualified (%s) at t = %.1f s" % (controller.stop_reason, controller.t_last))
if args.fitness:
    print("Fitness:", float(np.load("fitness_id0.npy")))
