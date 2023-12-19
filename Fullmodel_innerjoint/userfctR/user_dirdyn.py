# -*- coding: utf-8 -*-
"""Module for the definition of functions related to Equilibrium analysis."""
# Author: Robotran Team
# (c) Universite catholique de Louvain, 2020

from MBsysPy import MbsSensor
import MBsysPy as Robotran

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

    # Creating the sensors
    
    Sensor_trunk = MbsSensor(mbs_data)
    Sensor_hip = MbsSensor(mbs_data)
    
    Sensor_BallL = MbsSensor(mbs_data)
    Sensor_BallR = MbsSensor(mbs_data)
    Sensor_HeelL = MbsSensor(mbs_data)
    Sensor_HeelR = MbsSensor(mbs_data)
    
    # Defining the sensor ids as 1 and 4 (assuming your project has 4 sensors).
    
    id_trunk = mbs_data.sensor_id["Sensor_trunk"]
    id_hip = mbs_data.sensor_id["Sensor_hip"]
    
    id_BallL = mbs_data.sensor_id["Sensor_BallL"]
    id_BallR = mbs_data.sensor_id["Sensor_BallR"]
    id_HeelL = mbs_data.sensor_id["Sensor_HeelL"]
    id_HeelR = mbs_data.sensor_id["Sensor_HeelR"]
    
    
    # Computing the sensors
    
    Sensor_trunk.comp_s_sensor(id_trunk)
    Sensor_hip.comp_s_sensor(id_hip)
    
    Sensor_BallL.comp_s_sensor(id_BallL)
    Sensor_BallR.comp_s_sensor(id_BallR)
    Sensor_HeelL.comp_s_sensor(id_HeelL)
    Sensor_HeelR.comp_s_sensor(id_HeelR)

    # Add different outputs:
        
    mbs_data.sensors[id_trunk]=Sensor_trunk
    mbs_data.sensors[id_hip]=Sensor_hip
    
    mbs_data.sensors[id_BallL]=Sensor_BallL
    mbs_data.sensors[id_HeelL]=Sensor_HeelL
    mbs_data.sensors[id_BallR]=Sensor_BallR
    mbs_data.sensors[id_HeelR]=Sensor_HeelR
    

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

    # Example: The velocities are saved to a file, then the fields created in
    #          the MbsData instance during the function `user_dirdyn_init` are
    #          removed.
    #
    # import numpy as np # Should be done outside the function.
    #
    # np.savetxt("myfile.txt", mbs_data.my_sensor_v)
    # del mbs_data.my_sensor, mbs_data.my_sensor_v

    return
