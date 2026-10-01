#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-tune the reflex gains of the walking model with CMA-ES.

Every candidate is simulated for --tf seconds with the thesis fitness (time alive,
distance to the 1.3 m/s target, muscle effort) and its disqualification rules, in
parallel worker processes. All evaluations are appended to
fitness_data/<name>memory_{fitness,suggestion,fitness_breakdown}.npy (the format of
reflex-CMAES.py, so run_gait.py --log <name> replays them) and the best gains so far
are written to fitness_data/<name>best.json.

    python optimize_gains.py --name fixedtiming_tf10 --workers 4
    python optimize_gains.py --name fixedtiming_tf10 --resume        # continue a run

The search starts from the gains of Geyer & Herr (2010) and explores each gain
between 1/4 and 4 times its default (log scale).
"""
import argparse
import contextlib
import io
import json
import math
import multiprocessing
import os
import pickle
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

WORKR = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(WORKR)
MBS_FILE = os.path.join(PROJECT, "dataR", "Fullmodel_innerjoint.mbs")

# name, Geyer & Herr (2010) value, lower, upper, scale ("log" or "lin")
SEARCH_SPACE = [
    ("G_VAS", 2e-4, 0.5e-4, 8e-4, "log"),
    ("G_SOL", 1.2 / 4000, 1.2 / 16000, 1.2 / 1000, "log"),
    ("G_GAS", 1.1 / 1500, 1.1 / 6000, 1.1 / 375, "log"),
    ("G_TA", 1.1, 0.275, 4.4, "log"),
    ("G_SOL_TA", 1e-4, 0.25e-4, 4e-4, "log"),
    ("G_HAM", 2.166666666666667e-04, 2.166666666666667e-04 / 4, 2.166666666666667e-04 * 4, "log"),
    ("G_GLU", 1 / 3000., 1 / 12000., 4 / 3000., "log"),
    ("G_HFL", 0.5, 0.125, 2.0, "log"),
    ("G_HAM_HFL", 4.0, 1.0, 16.0, "log"),
    ("G_delta_theta", 1.145915590261647, 1.145915590261647 / 4, 1.145915590261647 * 4, "log"),
    ("theta_ref", 0.104719755119660, 0.0, 0.3, "lin"),
]
KEYS = [s[0] for s in SEARCH_SPACE]


def encode(values):
    """Parameter values -> normalized coordinates in [0, 1]."""
    x = []
    for (name, _, lo, hi, scale), v in zip(SEARCH_SPACE, values):
        x.append((math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) if scale == "log" else (v - lo) / (hi - lo))
    return np.array(x)


def decode(x):
    """Normalized coordinates -> parameter values."""
    values = []
    for (name, _, lo, hi, scale), u in zip(SEARCH_SPACE, np.clip(x, 0, 1)):
        values.append(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if scale == "log" else lo + u * (hi - lo))
    return values


# ------------------------------------------------------------------ worker side

_robotran = None


def _init_worker():
    global _robotran
    sys.path[:0] = [os.path.join(PROJECT, "User_function"), os.path.join(PROJECT, "userfctR"), WORKR]
    with contextlib.redirect_stdout(io.StringIO()):
        import MBsysPy
    _robotran = MBsysPy


def simulate(values, tf, dt=1000e-7):
    """Thesis fitness of one gain set (lower is better), plus its trace and why it stopped."""
    tf_memory = max(200, int(round(tf / 0.1)) + 2)
    parameters = {
        "dt": dt, "tf": tf, "flag_graph": False, "id": 0, "flag_outputs": False,
        "flag_fitness": True,
        "best_fitness_memory": np.ones(tf_memory) * 10 * tf,
        "fitness_memory": np.ones(tf_memory) * 10 * tf,
        "fm_memory": np.zeros(tf_memory),
        "fitness": 10 * tf,
    }
    parameters.update(zip(KEYS, values))
    with contextlib.redirect_stdout(io.StringIO()):
        mbs_data = _robotran.MbsData(MBS_FILE)
        mbs_data.process = 1
        part = _robotran.MbsPart(mbs_data)
        part.set_options(rowperm=1, verbose=0)
        part.run()
        mbs_data.process = 3
        dirdyn = _robotran.MbsDirdyn(mbs_data)
        mbs_data.user_model = parameters
        dirdyn.store_results = False
        dirdyn.set_options(dt0=dt, tf=tf, save2file=0)
        with tempfile.TemporaryDirectory() as tmp:  # the controller saves fitness_id*.npy in the cwd
            os.chdir(tmp)
            try:
                dirdyn.run()
            except RuntimeError:
                pass  # numerical failure: the fitness reached so far stands
            os.chdir(WORKR)
    controller = mbs_data.gait_controller
    trace = np.append(parameters["fitness_memory"][:200], [0])
    return float(parameters["fitness"]), float(controller.t_last or 0.0), controller.stop_reason, trace


def _evaluate(args):
    values, tf = args
    start = time.time()
    fitness, alive, reason, trace = simulate(values, tf)
    return fitness, alive, reason, trace, time.time() - start


# ------------------------------------------------------------------ main

def main():
    import cma

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="fixedtiming_tf10", help="log name in fitness_data/")
    ap.add_argument("--tf", type=float, default=10, help="simulated time per evaluation [s]")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--sigma0", type=float, default=0.15, help="initial step size (normalized coordinates)")
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="continue from fitness_data/<name>cma.pkl")
    args = ap.parse_args()

    os.chdir(WORKR)
    base = os.path.join("fitness_data", args.name)
    state_file = base + "cma.pkl"
    if args.resume and os.path.exists(state_file):
        with open(state_file, "rb") as f:
            es = pickle.load(f)
        memory = {k: list(np.load(base + "memory_%s.npy" % k, allow_pickle=True))
                  for k in ("fitness", "suggestion", "fitness_breakdown")}
        print("Resuming %s after %d evaluations" % (args.name, len(memory["fitness"])))
    else:
        x0 = encode([s[1] for s in SEARCH_SPACE])
        es = cma.CMAEvolutionStrategy(x0, args.sigma0, {"bounds": [0, 1], "popsize": args.popsize,
                                                        "seed": args.seed, "verbose": -9})
        memory = {"fitness": [], "suggestion": [], "fitness_breakdown": []}

    # compile the numba kernels once before the workers start
    _init_worker()
    simulate([s[1] for s in SEARCH_SPACE], 0.002)

    context = multiprocessing.get_context("spawn")
    best = (min(memory["fitness"]) if memory["fitness"] else math.inf)
    with ProcessPoolExecutor(args.workers, mp_context=context, initializer=_init_worker) as pool:
        for generation in range(args.generations):
            if es.stop():
                print("CMA-ES stopped:", es.stop())
                break
            start = time.time()
            X = es.ask()
            candidates = [decode(x) for x in X]
            results = list(pool.map(_evaluate, [(c, args.tf) for c in candidates]))
            es.tell(X, [r[0] for r in results])

            for values, (fitness, alive, reason, trace, seconds) in zip(candidates, results):
                memory["fitness"].append(fitness)
                memory["suggestion"].append(values)
                memory["fitness_breakdown"].append(trace)
                if fitness < best:
                    best = fitness
                    with open(base + "best.json", "w") as f:
                        json.dump({"fitness": fitness, "survived_s": alive, "evaluation": len(memory["fitness"]) - 1,
                                   "parameters": dict(zip(KEYS, values))}, f, indent=2)
            for k, v in memory.items():
                np.save(base + "memory_%s.npy" % k, np.array(v))
            with open(state_file, "wb") as f:
                pickle.dump(es, f)

            fits = [r[0] for r in results]
            alive = [r[1] for r in results]
            print("gen %3d | evals %5d | best %7.2f | this gen: min %7.2f median %7.2f | alive max %5.2f s median %5.2f s | %4.0f s"
                  % (generation, len(memory["fitness"]), best, min(fits), float(np.median(fits)), max(alive),
                     float(np.median(alive)), time.time() - start), flush=True)


if __name__ == "__main__":
    main()
