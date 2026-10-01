# -*- coding: utf-8 -*-
"""Module for the definition of functions related to Equilibrium analysis."""
# Author: Robotran Team
# (c) Universite catholique de Louvain, 2020

import os
import sys

from MBsysPy import MbsSensor
import MBsysPy as Robotran

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,  os.path.join(parent_dir, "User_function"))

import fast_symbolic
import gait_controller

SENSORS = ("Sensor_trunk", "Sensor_hip", "Sensor_BallL", "Sensor_HeelL", "Sensor_BallR", "Sensor_HeelR")

def user_dirdyn_init(mbs_data, mbs_dirdyn):
    """Run specific operation required by the user before running direct dynamic.

    Parameters
    ----------
    mbs_data : MBsysPy.MbsData
        The instance containing the multibody project.
    mbs_dirdyn : MBsysPy.MbsDirdyn
        The instance of the current direct dynamic process.

    Returns
    -------
    None.

    """
    Robotran.define_output_vector("external_force_X", 4)
    Robotran.define_output_vector("external_force_Z", 4)
    
    Robotran.define_output_vector("pos_Z", 4)
    
    # Robotran.define_output_vector("stance", 2)
    
    # Robotran.define_output_vector("stim_left", 7)
    # Robotran.define_output_vector("stim_right", 7)
    
    # Robotran.define_output_vector("Ldx_Rdx_no_filter", 2)
    # Robotran.define_output_vector("Ldx_Rdx_filter", 2)
    
    
    # Robotran.define_output_vector("Fm_VAS", 1)

    # Robotran.define_output_vector("knee_state", 6)
    
    # Robotran.define_output_vector("hip_position_Z", 1)
    Robotran.define_output_vector("velocity_Z", 4)
    Robotran.define_output_vector("velocity_X", 4)
    Robotran.define_output_vector("F_slide", 4)
    Robotran.define_output_vector("F_stick", 4)
    # Robotran.define_output_vector("knee_angle_velocity", 1)
    # Robotran.define_output_vector("knee_angle_delta", 1)

    for i in range (mbs_data.Nsensor):
    
        mbs_data.sensors.append(Robotran.MbsSensor(mbs_data))

    # compiled versions of the symbolic files (numba), unless disabled
    if mbs_data.user_model.get("flag_numba", True):
        fast_symbolic.accelerate(mbs_data)

    # fresh neuromuscular state for this simulation, stepped with the integration step
    gait_controller.start(mbs_data, mbs_dirdyn.get_options("dt0"))

    return


def user_dirdyn_loop(mbs_data, mbs_dirdyn):
    """Run specific operation required by the user at the end of each integrator step.

    In case of multistep integrator, this function is not called at intermediate
    steps.

    Parameters
    ----------
    mbs_data : MBsysPy.MbsData
        The instance containing the multibody project.
    mbs_dirdyn : MBsysPy.MbsDirdyn
        The instance of the current direct dynamic process.

    Returns
    -------
    None.
    """

    # Computing the sensors (trunk and hip for the trunk pitch, heels and balls for the contacts)
    for name in SENSORS:
        sensor_id = mbs_data.sensor_id[name]
        mbs_data.sensors[sensor_id].comp_s_sensor(sensor_id)

    # Advance the neuromuscular state once per accepted step
    gait_controller.get(mbs_data).step(mbs_data, mbs_data.tsim)

    return


def user_dirdyn_finish(mbs_data, mbs_dirdyn):
    """Run specific operations required by the user when direct dynamic analysis ends.

    Parameters
    ----------
    mbs_data : MBsysPy.MbsData
        The instance containing the multibody project.
    mbs_dirdyn : MBsysPy.MbsDirdyn
        The instance of the current direct dynamic process.

    Returns
    -------
    None.

    """

    gait_controller.get(mbs_data).finish(mbs_data)

    return
