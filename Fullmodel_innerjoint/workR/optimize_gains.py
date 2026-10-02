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

The search starts from the gains of Geyer & Herr (2010), or from --start (a
<name>best.json, evaluated first), and explores each gain between 1/4 and 4 times its
default (log scale). The gains of fitness_data/retuned_tf10 were found in two stages.
The first removes the disqualification for leaving the 1.3 m/s window (+-0.3 m), so
that gaits that walk stably but too slowly still score better than falls; the second
uses the thesis rules with a small step, since gains a few percent away from a walking
gait often fall. These commands reproduce the two logs:

    python optimize_gains.py --name stageA_tf5 --tf 5 --window inf --generations 13
    python optimize_gains.py --name retuned_tf10 --start fitness_data/stageA_tf5best.json --sigma0 0.01 --generations 12

--aged uses the aged muscles of the thesis (gait_controller.AGED, Thelen 2003) and
--target-speed the walking speed of the fitness; both are stored in
fitness_data/<name>settings.json, which run_gait.py reads to replay the gaits.
"""
import argparse
import collections
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


def set_range(factor):
    """Search the gains between 1/factor and factor times Geyer & Herr's values (log scale)."""
    global SEARCH_SPACE
    SEARCH_SPACE = [(n, g, g / factor, g * factor, s) if s == "log" else (n, g, lo, hi, s)
                    for n, g, lo, hi, s in SEARCH_SPACE]


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


def simulate(values, tf, window=0.3, dt=1000e-7, model=None):
    """Thesis fitness of one gain set (lower is better), plus its trace and why it stopped.

    model: extra model parameters (aging ratios, target_speed)."""
    tf_memory = max(200, int(round(tf / 0.1)) + 2)
    parameters = {
        "dt": dt, "tf": tf, "flag_graph": False, "id": 0, "flag_outputs": False,
        "flag_fitness": True,
        "best_fitness_memory": np.ones(tf_memory) * 10 * tf,
        "fitness_memory": np.ones(tf_memory) * 10 * tf,
        "fm_memory": np.zeros(tf_memory),
        "fitness": 10 * tf,
        "speed_window": window,
    }
    parameters.update(model or {})
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
    values, tf, window, model = args
    start = time.time()
    fitness, alive, reason, trace = simulate(values, tf, window, model=model)
    return fitness, alive, reason, trace, time.time() - start


# ------------------------------------------------------------------ main

def _replace(path, mode, write):
    """Write a log file through a temporary file, so stopping a run never leaves it truncated."""
    with open(path + ".tmp", mode) as f:
        write(f)
    os.replace(path + ".tmp", path)


def main():
    import cma

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="fixedtiming_tf10", help="log name in fitness_data/")
    ap.add_argument("--tf", type=float, default=10, help="simulated time per evaluation [s]")
    ap.add_argument("--window", type=float, default=0.3,
                    help="allowed distance to the 1.3 m/s target before disqualification [m] (inf: none)")
    ap.add_argument("--start", default=None, help="start from the parameters of a <name>best.json")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--sigma0", type=float, default=0.15, help="initial step size (normalized coordinates)")
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="continue from fitness_data/<name>cma.pkl")
    ap.add_argument("--target-speed", type=float, default=1.3, help="walking speed of the fitness [m/s]")
    ap.add_argument("--aged", action="store_true", help="aged muscles (gait_controller.AGED)")
    ap.add_argument("--limit-weight", type=float, default=0,
                    help="weight of the joint-limit work in the fitness, per W of mean power (default 0)")
    ap.add_argument("--range", type=float, default=4,
                    help="search the gains between 1/range and range times Geyer & Herr's values (default 4)")
    args = ap.parse_args()

    _init_worker()
    import gait_controller
    os.chdir(WORKR)
    base = os.path.join("fitness_data", args.name)
    state_file = base + "cma.pkl"
    settings_file = base + "settings.json"
    model = {"target_speed": args.target_speed}
    if args.aged:
        model.update(gait_controller.AGED)
    if args.limit_weight:
        model["limit_work_weight"] = args.limit_weight
    if args.resume and os.path.exists(settings_file):
        with open(settings_file) as f:
            settings = json.load(f)
        args.tf, args.window, model = settings["tf"], settings["window"], settings["model"]
        args.range = settings.get("range", 4)
        print("Settings of %s: tf %g s, window %g m, range %g, model %s"
              % (args.name, args.tf, args.window, args.range, model))
    else:
        _replace(settings_file, "w", lambda f: json.dump(
            {"tf": args.tf, "window": args.window, "model": model, "start": args.start, "sigma0": args.sigma0,
             "popsize": args.popsize, "seed": args.seed, "range": args.range}, f, indent=2))
    if args.range != 4:
        set_range(args.range)
    exact_start = None
    if args.resume and os.path.exists(state_file):
        with open(state_file, "rb") as f:
            es = pickle.load(f)
        memory = {k: list(np.load(base + "memory_%s.npy" % k, allow_pickle=True))
                  for k in ("fitness", "suggestion", "fitness_breakdown")}
        print("Resuming %s after %d evaluations" % (args.name, len(memory["fitness"])))
    else:
        if args.start:
            with open(args.start) as f:
                start = json.load(f)["parameters"]
            x0 = encode([start[k] for k in KEYS])
        else:
            x0 = encode([s[1] for s in SEARCH_SPACE])
        es = cma.CMAEvolutionStrategy(x0, args.sigma0, {"bounds": [0, 1], "popsize": args.popsize,
                                                        "seed": args.seed, "verbose": -9})
        if args.start:  # evaluation 0 is the starting point itself (inject takes the internal coordinates)
            es.inject([es.mean], force=True)
            exact_start = [start[k] for k in KEYS]  # decode(encode(v)) may differ from v in the last bit
        memory = {"fitness": [], "suggestion": [], "fitness_breakdown": []}

    # compile the numba kernels once before the workers start
    simulate([s[1] for s in SEARCH_SPACE], 0.002, model=model)

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
            if exact_start is not None:
                candidates[0], exact_start = exact_start, None
            results = list(pool.map(_evaluate, [(c, args.tf, args.window, model) for c in candidates]))
            es.tell(X, [r[0] for r in results])

            for values, (fitness, alive, reason, trace, seconds) in zip(candidates, results):
                memory["fitness"].append(fitness)
                memory["suggestion"].append(values)
                memory["fitness_breakdown"].append(trace)
                if fitness < best:
                    best = fitness
                    _replace(base + "best.json", "w", lambda f: json.dump(
                        {"fitness": fitness, "survived_s": alive, "evaluation": len(memory["fitness"]) - 1,
                         "parameters": dict(zip(KEYS, values))}, f, indent=2))
            for k, v in memory.items():
                _replace(base + "memory_%s.npy" % k, "wb", lambda f: np.save(f, np.array(v)))
            _replace(state_file, "wb", lambda f: pickle.dump(es, f))

            fits = [r[0] for r in results]
            alive = [r[1] for r in results]
            stops = collections.Counter(r[2] for r in results if r[2])
            print("gen %3d | evals %5d | best %7.2f | this gen: min %7.2f median %7.2f | alive max %5.2f s median %5.2f s"
                  " | stopped: %s | %4.0f s"
                  % (generation, len(memory["fitness"]), best, min(fits), float(np.median(fits)), max(alive),
                     float(np.median(alive)), " ".join("%s %d" % kv for kv in sorted(stops.items())) or "none",
                     time.time() - start), flush=True)


if __name__ == "__main__":
    main()
