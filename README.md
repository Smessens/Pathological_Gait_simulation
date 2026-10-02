# Simulating-human-walking-to-virtually-devellop-and-test-new-methods-of-assistance
Repository used by Matthieu AUSSEMS and Nicolas DINEUR for the development of the Robotran model of the human walking based on the Geyer's model as part of their thesis

## Running it today (tested October 2026, Linux, Python 3.11)
```
pip install --index-url https://www.robotran.eu/dist/ MBsysPy==1.30.0   # robotran.eu can be flaky: retry
pip install -r requirements.txt
cd Fullmodel_innerjoint/workR
python run_gait.py           # replay the re-tuned gait (best of fitness_data/retuned_tf10), 10 s
python render_gait.py        # animationR/dirdyn_q.anim -> animationR/gait.gif
python optimize_gains.py     # re-tune the reflex gains (CMA-ES on all CPU cores; --resume continues a run)
```
`run_gait.py --log L --row N` replays another logged gain set, `--tf` sets the duration, `--fitness` adds the optimizer's fitness and disqualification checks. The first run compiles the numba kernels (about a minute); later runs reuse them.

### Changes since 2024
- **Neuromuscular timing.** Robotran calls `user_JointForces` five times per integration step (four Runge-Kutta stages and an output evaluation). The 2024 code advanced the muscle and reflex state at almost every call with `dt/2`, so it ran about 2.15 times too fast: activation time constant ~4.6 ms instead of 10 ms, neural delays ~9/5/2 ms instead of 20/10/5 ms, contractile velocities ~2.15 times too high. The state now lives in `User_function/gait_controller.py` and advances exactly once per accepted step (from `user_dirdyn_loop`, with the integration step); `user_JointForces` only turns it into torques. The reflex laws (`Neural_control_layer.py`) and the muscle model are unchanged (checked against the 2024 functions). Gains tuned on the 2024 timing (`compact_tf10`, ...) no longer walk (the best 2024 gains fall after 1.6 s), hence the re-tuning below.
- **Re-tuned gains** (`fitness_data/retuned_tf10`, replayed by `run_gait.py`). A search with the thesis fitness starting from Geyer & Herr's gains stalled (`fixedtiming_tf10`: after 336 simulations the best gait still fell after 4.2 s), since nearly every candidate falls within a few seconds. The gains were therefore re-tuned in two stages (commands in `optimize_gains.py`): 5 s simulations without the ±0.3 m speed window (`stageA_tf5`, 156 simulations), then the thesis fitness and rules over 10 s, starting from the best gait of the first stage (`retuned_tf10`, 144 simulations). The re-tuned model walks at 1.30 m/s, within ±0.11 m of the 1.3 m/s target, and also passes the thesis rules over 30 s (`run_gait.py --tf 30 --fitness`); its thesis fitness over 10 s is 6.32. The gait is sensitive to the gains: of 12 random changes of about 3% to every gain, 8 still walk the full 10 s. G_VAS ends at the top of its search range (4 times Geyer & Herr's value). With Geyer & Herr's gains the corrected model also walks, but at 0.84 m/s.
- **No state carried between simulations.** Each simulation gets a fresh controller, so results no longer depend on what ran before in the same Python process.
- **Speed.** About 6 s of computation per simulated second on one core instead of ~60 s in 2024, and 3.4 s with `"flag_outputs": False` (skips the `*.res` contact outputs; the optimizer does). Besides updating the state once per step, numba compiles Robotran's generated symbolic files (`User_function/fast_symbolic.py`) and the muscle model (`muscle_kernels.py`), with results identical to the pure-Python code; `"flag_numba": False` in the parameters runs pure Python.
- Disqualified runs stop through Robotran's `flag_stop` instead of an injected infinite force.

### Aging study (thesis 4.2), re-run on the corrected model
- **Aged muscles.** All five changes of Thelen (2003) listed in the thesis, as ratios of the young values (`gait_controller.AGED`): maximum isometric force ×0.7, maximum shortening velocity ×0.8, deactivation time constant ×1.2 (Geyer's model has one 10 ms constant: activation now falls with 12 ms and rises with 10 ms), strain at which the parallel elasticity reaches F_max ×0.5/0.6, eccentric force enhancement N ×1.8/1.4. The 2024 code had only the F_max and v_max scaling (F_max ×0.778 in its last aged setup).
- **Aged gaits.** Re-tuned from the young gait with the aged muscles and the thesis fitness at 1.0 and 1.3 m/s, in stages: 5 s and 10 s without the speed window, then the thesis rules over 10 s (`fitness_data/aged10_*`, `aged13_*`; the final logs are `aged10_tf10` and `aged13_tf10`). At 1.3 m/s the search reached the 4× bound of G_VAS and G_HFL, so its later stages search up to 8× Geyer & Herr's gains (`optimize_gains.py --range 8`). The two gaits give their logged fitness when replayed (7.48 and 7.20) and walk 60 s without falling.
- **Analysis** (`gait_analysis.py`; tables in `workR/aging/results.md`, curves in `workR/aging/gait_curves.svg`): 60 s of each gait, mean stride of both legs, first 3 strides dropped, as in the thesis; joint angles (calibrated on the model geometry), applied joint torques, power = torque × joint velocity. A trend of the thesis' Tables 4.2-4.4 passes when the old gait differs in its direction by more than 1.5° or 15 % (moments, powers), just above the difference between the two halves of the young run.

| | Young 1.3 m/s | Old 1.0 m/s | Old 1.3 m/s | Thesis (young / old 1.0 / old 1.3) |
|---|---|---|---|---|
| Speed (m/s) | 1.300 | 0.974 | 1.290 | 1.3 / 1.0 / 1.3 |
| Stride frequency (strides/s) | 0.868 | 0.692 | 0.817 | 0.92 / 0.77 / 0.89 |
| Stride length (m) | 1.497 | 1.408 | 1.579 | 1.42 / 1.30 / 1.46 |
| Thesis trends passed (of 23) | | 7 (6 within tolerance) | 5 (5 within tolerance) | 14 / 12 |

On the corrected model the aged gaits reproduce few of the expected trends, and at equal speed (old vs young at 1.3 m/s) several go against the trends of older adults: longer and slower strides, more ankle plantarflexion and ankle power at push-off, less hip power (older adults at the same speed take shorter steps and use the hip more and the ankle less, DeVita & Hortobágyi 2000). In both aged gaits the knee slams into its hyperextension limit at the end of swing: about 90 % of the knee's peak power absorption comes from the joint limit, not the muscles. The fitness (speed, survival, muscle effort) rewards reaching the speed with little muscle force and nothing in it favours a cautious gait (thesis 5.1); with weaker muscles the optimizer raises the plantar-flexor and hip-flexor gains instead (G_SOL 1.4× Geyer & Herr's value when young, 1.8× and 3.0× when old; G_HFL 2.0×, 4.0× and 3.3×; the 1.0 m/s search used the 4× range and its G_HFL ends on that bound). To regenerate:

```
python run_gait.py --log retuned_tf10 --tf 60 --fitness --window inf --no-files --record young.npz
python run_gait.py --log aged10_tf10 --tf 60 --fitness --window inf --no-files --record aged10.npz
python run_gait.py --log aged13_tf10 --tf 60 --fitness --window inf --no-files --record aged13.npz
python gait_analysis.py young.npz aged10.npz aged13.npz --labels "Young 1.3 m/s" "Old 1.0 m/s" "Old 1.3 m/s" --out aging
```

### Penalizing the work of the joint limits
- **Why.** In the gaits above the knee spends about half of the time in the range of its hyperextension limit (Geyer's soft stop from 175°, `Muscle_actuation_layer.joint_limits`): the aged knees stop their extension on it at the end of swing (about 90 % of their absorption peak) and the young knee leans on it in stance. The fitness rewards this, since the limit holds the knee without muscle force.
- **Penalty.** `optimize_gains.py --limit-weight W` adds W × (work of the joint limits / t) to the fitness at every 0.1 s check, the work being the integral of Σ|limit torque × joint speed| over the six joints (`limit_work_weight` of `gait_controller`, stored in the log's settings.json). Two weights: 1/300 per W, where the young gait's knee-limit term equals its effort term, and 1/50 per W; beyond about 1/25 per W, walking would score worse than falling for the gait with the most limit work.
- **Searches** (10 s, from the gaits above; logs `fitness_data/*_lim*_tf10`). Under the thesis rules (`*_lim_tf10`, 1/300) nearly all candidates near the start left the ±0.3 m window, so the gains were searched without it (`*_limA_tf10` at 1/300, `*_lim50_tf10` at 1/50); at 1/50 the old 1.3 m/s search drifted out of the window and continued under the thesis rules (`aged13_lim50B_tf10`). Each final gait is the best candidate that passes the thesis rules over 10 s and walks 60 s: the best 10 s gaits at 1/50 fell after 12-14 s, so the rows kept are 23 (young), 108 (old 1.0) and 26 (old 1.3). At 1/300 no young gait better than the unpenalized one was found: the fitness is chaotic (a 1e-16 change of the gains moves the 10 s fitness by about 2 %, the size of the differences between the best candidates at this weight).

| 60 s runs (`workR/aging_limit300`, `workR/aging_limit50`) | No penalty | 1/300 per W | 1/50 per W |
|---|---|---|---|
| Knee-limit power, young / old 1.0 / old 1.3 (W) | 15.6 / 9.2 / 22.7 | 15.6 / 6.3 / 18.6 | 10.4 / 1.8 / 15.7 |
| Old knees' absorption peak (W), part from the limit: 1.0 / 1.3 m/s | -783, 94 % / -1110, 91 % | -306, 86 % / -313, 5 % | -194, 0 % / -341, 0 % |
| Stride length, young / old 1.0 / old 1.3 (m) | 1.497 / 1.408 / 1.579 | 1.497 / 1.417 / 1.554 | 1.516 / 1.435 / 1.477 |
| Stride frequency, young / old 1.0 / old 1.3 (strides/s) | 0.868 / 0.692 / 0.817 | 0.868 / 0.691 / 0.825 | 0.851 / 0.667 / 0.908 |
| Thesis trends passed, old 1.0 / old 1.3 (of 23) | 7 / 5 | 9 / 7 | 6 / 7 |

The penalty does what it targets. At 1/50 the joint limit no longer makes the aged knees' absorption peak, which is now in early stance (5-6 % of the stride) like the young one's; the old 1.0 m/s knee is in the limit's range 13 % of the time instead of 48 %. At 1.3 m/s the old gait now takes shorter and more frequent strides than the young one, as older adults do at equal speed. The agreement with the 23 trends does not improve, though, and the main mismatches remain with all three fitnesses: at 1.3 m/s the old ankle plantarflexes much more at toe-off (-30 to -36° against -15 to -20° for the young gait), both old gaits land with a straighter knee (2-4° of flexion against 8-11°) and generate more knee power, and the old 1.0 m/s gait has lower hip moments and power than the young reference, which walks faster. Removing the joint-limit loophole is therefore not enough for the re-tuned reflexes to reproduce aged gait; as the thesis suggests (5.1), nothing in the fitness (speed, survival, effort) favours the cautious strategy of older adults. The penalty also costs stability: the best 10 s gaits fell within 15 s at 1/50, while those at 1/300 walk 60 s. The 2024 knee limit is disabled beyond 185° (a guard against unrealistic angles); no gait here exceeds 184.3°. To regenerate (young.npz as above):

```
python run_gait.py --log aged10_limA_tf10 --tf 60 --fitness --window inf --no-files --record aged10_limA.npz
python run_gait.py --log aged13_limA_tf10 --tf 60 --fitness --window inf --no-files --record aged13_limA.npz
python gait_analysis.py young.npz aged10_limA.npz aged13_limA.npz --labels "Young 1.3 m/s" "Old 1.0 m/s" "Old 1.3 m/s" --out aging_limit300
python run_gait.py --log young_lim50_tf10 --row 23 --tf 60 --fitness --window inf --no-files --record young_lim50.npz
python run_gait.py --log aged10_lim50_tf10 --row 108 --tf 60 --fitness --window inf --no-files --record aged10_lim50.npz
python run_gait.py --log aged13_lim50B_tf10 --row 26 --tf 60 --fitness --window inf --no-files --record aged13_lim50.npz
python gait_analysis.py young_lim50.npz aged10_lim50.npz aged13_lim50.npz --labels "Young 1.3 m/s" "Old 1.0 m/s" "Old 1.3 m/s" --out aging_limit50
```

### Requiring ground clearance
- **Why.** The thesis (5.1) attributes part of the mismatch to the model having no reason to lift its feet (no fear of tripping) and suggests requiring a minimal ground clearance. Measured as the lowest point of the swing foot (heel or ball) over the middle half of each swing, the clearance of the gaits above is 51 mm for the young gait, 28 mm for the old 1.3 m/s gait and 16 mm for the old 1.0 m/s one (typical human minimum toe clearance: 1-2 cm): the young model lifts its feet generously, and only the old 1.0 m/s gait is as low as people.
- **Term.** `optimize_gains.py --clearance-weight 10 --clearance-min 0.03` adds 10 × the mean shortfall of the swings' clearance below 3 cm [m] at every check (`clearance_weight` and `clearance_min` of `gait_controller`); a contact shorter than 50 ms within a swing (a scuff) counts as zero clearance. Clearing 3 cm (about twice the human minimum, a margin for a careful walker) scores nothing, so lifting the feet higher is not rewarded, which avoids the trade-off the thesis expected from a reward for clearance. The term comes on top of the 1/50 per W joint-limit penalty.
- **Searches** (`fitness_data/*_clear_tf10`, from the 1/50 gaits saved as `fitness_data/*row*.json`): 15 generations without the speed window (23 for the old 1.0 m/s gait, which kept improving); each final gait is again the best candidate that passes the thesis rules over 10 s and walks 60 s, here the best of each log. No young gait better than the start was found, since it already clears 3 cm: the young gait is the 1/50 one.

| 60 s runs (`workR/aging`, `aging_limit300`, `aging_limit50`, `aging_clearance`) | No penalty | 1/300 per W | 1/50 per W | 1/50 per W + clearance |
|---|---|---|---|---|
| Mid-swing clearance, young / old 1.0 / old 1.3 (mm) | 51 / 16 / 28 | 51 / 17 / 26 | 58 / 18 / 36 | 58 / 29 / 31 |
| Knee-limit power, young / old 1.0 / old 1.3 (W) | 15.6 / 9.2 / 22.7 | 15.6 / 6.3 / 18.6 | 10.4 / 1.8 / 15.7 | 10.4 / 2.1 / 9.5 |
| Stride length, young / old 1.0 / old 1.3 (m) | 1.497 / 1.408 / 1.579 | 1.497 / 1.417 / 1.554 | 1.516 / 1.435 / 1.477 | 1.516 / 1.520 / 1.529 |
| Thesis trends passed, old 1.0 / old 1.3 (of 23) | 7 / 5 | 9 / 7 | 6 / 7 | 5 / 8 |

The term works on the gait that lacked clearance: the old 1.0 m/s gait now clears 29 mm (lowest swing 25 mm, against 11 mm before) and lands with a flexed knee like the young gait (12° at heel strike, against 4°); the old 1.3 m/s gait stays near the minimum (31 mm) while the joint-limit penalty keeps lowering its knee-limit work. The agreement with the trends does not improve: across the four fitnesses 5 to 9 of the 23 trends pass. The old gaits meet the clearance with more knee extensor work (knee power generation 139 and 173 W against 6 W for the young gait) and at 1.3 m/s still plantarflex the ankle much more at push-off (-29° against -15°), although that gait now generates more hip power than the young one (522 against 387 W), as older adults do. Neither removing the joint-limit loophole nor requiring clearance makes the re-tuned aged reflexes reproduce aged gait: with weaker and slower muscles the optimizer keeps the target speed by raising the muscle gains, whereas older adults shift work from the ankle to the hip and take shorter steps, and nothing in a fitness built on a fixed speed and low effort favours that. As at 1/50, the best candidates sit close to falling: 5 of the 8 best old 1.0 m/s candidates fell within 30 s. To regenerate (young_lim50.npz as above):

```
python run_gait.py --log aged10_clear_tf10 --tf 60 --fitness --window inf --no-files --record aged10_clear.npz
python run_gait.py --log aged13_clear_tf10 --tf 60 --fitness --window inf --no-files --record aged13_clear.npz
python gait_analysis.py young_lim50.npz aged10_clear.npz aged13_clear.npz --labels "Young 1.3 m/s" "Old 1.0 m/s" "Old 1.3 m/s" --out aging_clearance
```

MBsysPy 1.30.0 reproduces the 2024 results: re-running the passive test of commit `85395b4` with the 2024 code gives its committed `*.res` files to ~1e-14.

## User_function
The following 3 codes contain the functions to be used in Joint_forces to obtain the torques to be applied to the joints.
 
### Muscle actuation layer : 
Functions use for the Muscle actuation layer. Functions to compute vce and lce (integration) adn joint limits and functions to update lmtu and torque.

### Neural control layer : 
Functions use for the neural control layer. Functions to compute the muscle stimulations from the delayed sensory signals.

### gait_controller :
State of the neuromuscular model for one simulation (delay lines, filters, activations, contractile lengths, fitness bookkeeping). It advances once per accepted integration step and gives the joint torques at every Runge-Kutta stage.

### fast_symbolic, muscle_kernels :
numba compilation of Robotran's symbolic files and of the muscle model (optional, identical results).

### Useful function : 
Functions to compute the trunk angle, to obtained the gait phase and to compute the force of actuation of the inner thigh prismatic joint. The low pass filter used to pass from the stimulation to the activation is implemented in this file.  

The following codes are used for validation of functions with exported signals from simulink in excel files:

### Muscle_test : 
Test of muscle layer functions

### Muscle_stimulation_test : 
Test of muscle layer functions and neural control layers functions

## AnimationR
In Robotran, the "AnimationR" folder contains files and resources related to the visualization and graphical animation of simulation results. This folder is generally used to create visual representations of the movement of simulated bodies or mechanical engineering systems.

## dataR
In Robotran, the "dataR" folder is generally used to store the data required to models the Multibody system in the MBsysPad platform.

## resultsR
Results obtained with simulations of our Robotran model.

## userfctR
The only files used for this projetc in userfctR are : 

### user_JointForces : 
The user_jointforces in Robotran is a function for specifying the forces applied to the joints of a mechanical engineering system.
It allows the user to define customized forces according to their specific needs. In the framework of this project, it returns the torques applied to joints (Ankle, knee and Hip) and to the inner-thigh joints, computed by gait_controller.

### user_ExtForces : 
The user_extforces file in Robotran allows the user to specify external forces applied to the mechanical engineering system.
It allows forces to be customized according to the user's specific needs.
In this way, realistic scenarios can be modeled, taking into account the external forces acting on the system.
In the framework of this project, this file is used to compute the external functions applied to the heel and ball points of the feet to model the ground contact.

### user_dirdyn : 
Dirdyn in Robotran is a function for the dynamic simulation of a mechanical engineering system. It uses the equations of motion to calculate the positions, velocities, accelerations and forces of each body in the system.
This File can be used by the user for sensor initialization. In the framework of this project sensor are used to have Heel and Ball positions to compute the phase of walking. Sensors are also applied on the trunk to obtain its pitch angle. After each accepted step, user_dirdyn_loop updates these sensors and advances gait_controller.

## workR
main : The role of the "main" function in Robotran is to act as a starting point for program execution, where you can define the program's initial behavior and organize the order in which instructions are executed.

run_gait.py replays a logged gain set, render_gait.py turns an animation into a GIF, optimize_gains.py re-tunes the reflex gains (CMA-ES). reflex-CMAES.py and reflex_tester.py are the 2024 optimization scripts (scikit-optimize).
