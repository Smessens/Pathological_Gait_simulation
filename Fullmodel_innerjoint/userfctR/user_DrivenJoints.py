# -*- coding: utf-8 -*-
"""Module for the definition of driven joints."""
# Author: Robotran Team
# (c) Universite catholique de Louvain, 2020


def user_DrivenJoints(mbs_data, tsim):
    """Set the values of the driven joints directly in the MbsData structure.

    The position, velocity and acceleration of the driven joints must be set in
    the attributes mbs_data.q, mbs_data.qd and mbs_data.qdd .

    Parameters
    ----------
    mbs_data : MBsysPy.MbsData
        The multibody system associated to this computation.
    tsim : float
        The current time of the simulation.

    Returns
    -------
    None
    """

    # get the joint id
    id_j1 = mbs_data.joint_id["innerthighL"]
    id_j2 = mbs_data.joint_id["innerthighR"]

    # impose the position, velocity and acceleration
    
    # mbs_data.q[id_j1]   = 0
    # mbs_data.qd[id_j1]  = 0
    # mbs_data.qdd[id_j1] = 0
    
    # mbs_data.q[id_j2]   = 0
    # mbs_data.qd[id_j2]  = 0
    # mbs_data.qdd[id_j2] = 0
    
    

    return
