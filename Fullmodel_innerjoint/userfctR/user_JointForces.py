# -*- coding: utf-8 -*-
"""Module for the definition of joint forces."""
# Author: Robotran Team
# (c) Universite catholique de Louvain, 2020
import sys

import os
# Get the directory where your script is located
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0,  os.path.join(parent_dir, "User_function"))
sys.path.insert(1,  os.path.join(parent_dir, "userfctR"))

import gait_controller


def user_JointForces(mbs_data, tsim):
    """Compute the force and torques in the joint.

    It fills the MBsysPy.MbsData.Qq array.

    Parameters
    ----------
    mbs_data : MBsysPy.MbsData
        The multibody system associated to this computation.
    tsim : float
        The current time of the simulation.

    Notes
    -----
    The numpy.ndarray MBsysPy.MbsData.Qq is 1D array with index starting at 1.
    The first index (array[0]) must not be modified. The first index to be
    filled is array[1].

    Robotran calls this function at every Runge-Kutta stage (5 times per step),
    so it only converts the neuromuscular state into torques: muscle torques
    (ankle, knee, hip), joint limits and the inner-thigh pressure sheet. The
    state itself advances once per accepted step in user_dirdyn_loop.

    Returns
    -------
    None
    """
    gait_controller.get(mbs_data).joint_torques(mbs_data)
    return


sys.path.insert(2,  os.path.join(parent_dir, "workR"))
import TestworkR

if __name__ == "__main__":
    TestworkR.runtest(1000e-7,0.4,c=False)
