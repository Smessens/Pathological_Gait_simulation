# Simulating-human-walking-to-virtually-devellop-and-test-new-methods-of-assistance
Repository used by Matthieu AUSSEMS and Nicolas DINEUR for the development of the Robotran model of the human walking based on the Geyer's model as part of their thesis

## Running it today (tested October 2026, Linux, Python 3.11)
```
pip install --index-url https://www.robotran.eu/dist/ MBsysPy==1.30.0   # robotran.eu can be flaky: retry
pip install -r requirements.txt
cd Fullmodel_innerjoint/workR
python run_gait.py           # replay the re-tuned gait (best of fitness_data/fixedtiming_tf10), 10 s
python render_gait.py        # animationR/dirdyn_q.anim -> animationR/gait.gif
python optimize_gains.py     # re-tune the reflex gains (CMA-ES on all CPU cores; --resume continues a run)
```
`run_gait.py --log L --row N` replays another logged gain set, `--tf` sets the duration, `--fitness` adds the optimizer's fitness and disqualification checks. The first run compiles the numba kernels (about a minute); later runs reuse them.

### Changes since 2024
- **Neuromuscular timing.** Robotran calls `user_JointForces` five times per integration step (four Runge-Kutta stages and an output evaluation). The 2024 code advanced the muscle and reflex state at almost every call with `dt/2`, so it ran about 2.15 times too fast: activation time constant ~4.6 ms instead of 10 ms, neural delays ~9/5/2 ms instead of 20/10/5 ms, contractile velocities ~2.15 times too high. The state now lives in `User_function/gait_controller.py` and advances exactly once per accepted step (from `user_dirdyn_loop`, with the integration step); `user_JointForces` only turns it into torques. The reflex laws (`Neural_control_layer.py`) and the muscle model are unchanged (checked against the 2024 functions). Gains tuned on the 2024 timing (`compact_tf10`, ...) no longer walk; `fixedtiming_tf10` holds gains re-tuned on the corrected model.
- **No state carried between simulations.** Each simulation gets a fresh controller, so results no longer depend on what ran before in the same Python process.
- **Speed.** About 6 s of computation per simulated second on one core instead of ~60 s in 2024, and 3.4 s with `"flag_outputs": False` (skips the `*.res` contact outputs; the optimizer does). Besides updating the state once per step, numba compiles Robotran's generated symbolic files (`User_function/fast_symbolic.py`) and the muscle model (`muscle_kernels.py`), with results identical to the pure-Python code; `"flag_numba": False` in the parameters runs pure Python.
- Disqualified runs stop through Robotran's `flag_stop` instead of an injected infinite force.

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
