# Simulating-human-walking-to-virtually-devellop-and-test-new-methods-of-assistance
Repository used by Matthieu AUSSEMS and Nicolas DINEUR for the development of the Robotran model of the human walking based on the Geyer's model as part of their thesis

## Running it today (tested October 2026, Linux, Python 3.11)
```
pip install --index-url https://www.robotran.eu/dist/ MBsysPy==1.30.0   # robotran.eu can be flaky: retry
pip install -r requirements.txt
cd Fullmodel_innerjoint/workR
python run_gait.py           # replay compact_tf10 row 1026, the set in reflex_tester.py (~1 min per simulated second)
python render_gait.py        # animationR/dirdyn_q.anim -> animationR/gait.gif
```
`run_gait.py --row N` (or `--row best`) replays another row, `--tf` sets the duration, `--fitness` adds the optimizer's fitness and disqualification checks. `reflex-CMAES.py` runs the reflex-gain optimization (scikit-optimize, not CMA-ES).

With MBsysPy 1.30.0, re-running the passive test of commit `85395b4` reproduces its committed results: the same `user_JointForces` call sequence and the same states up to ~1e-14. Run each gait in a fresh Python process: `Neural_control_layer` keeps module-level state between simulations, so a gait evaluated after another one in the same process starts differently.

## User_function
The following 3 codes contain the functions to be used in Joint_forces to obtain the torques to be applied to the joints.
 
### Muscle actuation layer : 
Functions use for the Muscle actuation layer. Functions to compute vce and lce (integration) adn joint limits and functions to update lmtu and torque.

### Neural control layer : 
Functions use for the neural control layer. Functions to compute the muscle stimulations.

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
It allows the user to define customized forces according to their specific needs. In the framework of this project, user functions are assembled and used together in this file to obtain the torques applied to joints (Ankle, knee and Hip).

### user_ExtForces : 
The user_extforces file in Robotran allows the user to specify external forces applied to the mechanical engineering system.
It allows forces to be customized according to the user's specific needs.
In this way, realistic scenarios can be modeled, taking into account the external forces acting on the system.
In the framework of this project, this file is used to compute the external functions applied to the heel and ball points of the feet to model the ground contact.

### user_dirdyn : 
Dirdyn in Robotran is a function for the dynamic simulation of a mechanical engineering system. It uses the equations of motion to calculate the positions, velocities, accelerations and forces of each body in the system.
This File can be used by the user for sensor initialization. In the framework of this project sensor are used to have Heel and Ball positions to compute the phase of walking. Sensors are also applied on the trunk to obtain its pitch angle.

## workR
main : The role of the "main" function in Robotran is to act as a starting point for program execution, where you can define the program's initial behavior and organize the order in which instructions are executed.
