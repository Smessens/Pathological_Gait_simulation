# -*- coding: utf-8 -*-
"""Module for the definition of joint forces."""
# Author: Robotran Team
# (c) Universite catholique de Louvain, 2020
import sys

import os
import time
# Get the directory where your script is located
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0,  os.path.join(parent_dir, "User_function"))
sys.path.insert(1,  os.path.join(parent_dir, "userfctR"))



import useful_functions as u_f
import Muscle_actuation_layer as muscle
import Neural_control_layer as neural
import MBsysPy
import matplotlib.pyplot as plt
import numpy as np
import gait_graph 
import metrics
from datetime import datetime
import math

### Initial parameters :
    
ground_limit = 0
n_muscle = 7

VAS =0
SOL =1
GAS = 2
TA = 3
HAM = 4
GLU = 5
HFL = 6

ankle=0
knee=1
hip=2

tau =0.01

### Memories initialisation and globals parameters

# Ldx/Rdx  / theta-dtheta trunk / stance memories

LDx_memory = np.array([0,0])
RDx_memory = np.array([0,0])

act_memory_TA_L = np.array([0,0])
act_memory_GAS_L = np.array([0,0])
act_memory_SOL_L = np.array([0,0])
act_memory_VAS_L = np.array([0,0])
act_memory_HAM_L = np.array([0,0])
act_memory_GLU_L = np.array([0,0])
act_memory_HFL_L = np.array([0,0])

act_memory_TA_R = np.array([0,0])
act_memory_GAS_R = np.array([0,0])
act_memory_SOL_R = np.array([0,0])
act_memory_VAS_R = np.array([0,0])
act_memory_HAM_R = np.array([0,0])
act_memory_GLU_R = np.array([0,0])
act_memory_HFL_R = np.array([0,0])

lmtc_memory_TA_L = np.array([0,0])
lmtc_memory_GAS_L = np.array([0,0])
lmtc_memory_SOL_L = np.array([0,0])
lmtc_memory_VAS_L = np.array([0,0])
lmtc_memory_HAM_L = np.array([0,0])
lmtc_memory_GLU_L = np.array([0,0])
lmtc_memory_HFL_L = np.array([0,0])

lmtc_memory_TA_R = np.array([0,0])
lmtc_memory_GAS_R = np.array([0,0])
lmtc_memory_SOL_R = np.array([0,0])
lmtc_memory_VAS_R = np.array([0,0])
lmtc_memory_HAM_R = np.array([0,0])
lmtc_memory_GLU_R = np.array([0,0])
lmtc_memory_HFL_R = np.array([0,0])








theta_trunk_memory = np.array([0, 0])
dtheta_trunk_memory = np.array([0, 0])

Stance_memory = np.array([0,0,0])
Stance_memory_delayed = 0
# Fm, lce

f_m_memoryL = np.array([0, 0, 0, 0, 0, 0, 0, 0])
lce_memoryL = np.array([0, 0, 0, 0, 0, 0, 0, 0])

f_m_memoryR = np.array([0, 0, 0, 0, 0, 0, 0, 0])
lce_memoryR = np.array([0, 0, 0, 0, 0, 0, 0, 0])

# Stimulations
StimL = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
StimR = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

#global parameters

Ldx_prec = 0
Rdx_prec = 0

A_prec_TA_L=0
A_prec_GAS_L=0
A_prec_SOL_L=0
A_prec_VAS_L=0
A_prec_HAM_L=0
A_prec_GLU_L=0
A_prec_HFL_L=0

A_prec_TA_R=0
A_prec_GAS_R=0
A_prec_SOL_R=0
A_prec_VAS_R=0
A_prec_HAM_R=0
A_prec_GLU_R=0
A_prec_HFL_R=0

lce_prec_TA_L=0
lce_prec_GAS_L=0
lce_prec_SOL_L=0
lce_prec_VAS_L=0
lce_prec_HAM_L=0
lce_prec_GLU_L=0
lce_prec_HFL_L=0

lce_prec_TA_R=0
lce_prec_GAS_R=0
lce_prec_SOL_R=0
lce_prec_VAS_R=0
lce_prec_HAM_R=0
lce_prec_GLU_R=0
lce_prec_HFL_R=0




stim_left = np.zeros(7)
stim_right = np.zeros(7)

elapsed_time1=0
elapsed_time2=0
elapsed_time3=0
elapsed_time4=0

flag_graph=True
flag_initiated=False
prev_time = 0#datetime.now()
prev_datetime = datetime.now()

#prev_duplicate_tsim=-1

def user_JointForces(mbs_data, tsim):
    
    #global prev_duplicate_tsim 
    #if(prev_duplicate_tsim==tsim):
    #    return
    #else:
    #    prev_duplicate_tsim=tsim
    
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

    Returns
    -------
    None
    """
    
    #global variable
    global LDx_memory
    global RDx_memory
    global theta_trunk_memory
    global dtheta_trunk_memory
    global Stance_memory
    global Stance_memory_delayed
    
    global act_memory_TA_L 
    global act_memory_GAS_L
    global act_memory_SOL_L 
    global act_memory_VAS_L
    global act_memory_HAM_L 
    global act_memory_GLU_L 
    global act_memory_HFL_L 

    global act_memory_TA_R 
    global act_memory_GAS_R 
    global act_memory_SOL_R
    global act_memory_VAS_R 
    global act_memory_HAM_R 
    global act_memory_GLU_R 
    global act_memory_HFL_R 

    global lmtc_memory_TA_L
    global lmtc_memory_GAS_L 
    global lmtc_memory_SOL_L 
    global lmtc_memory_VAS_L 
    global lmtc_memory_HAM_L 
    global lmtc_memory_GLU_L 
    global lmtc_memory_HFL_L

    global lmtc_memory_TA_R 
    global lmtc_memory_GAS_R 
    global lmtc_memory_SOL_R 
    global lmtc_memory_VAS_R 
    global lmtc_memory_HAM_R 
    global lmtc_memory_GLU_R 
    global lmtc_memory_HFL_R 
    
    global f_m_memoryL
    global lce_memoryL
    
    global f_m_memoryR
    global lce_memoryR
    
    global StimL
    global StimR
    
    global Ldx_prec
    global Rdx_prec

    global A_prec_TA_L
    global A_prec_GAS_L
    global A_prec_SOL_L
    global A_prec_VAS_L
    global A_prec_HAM_L
    global A_prec_GLU_L
    global A_prec_HFL_L

    global A_prec_TA_R
    global A_prec_GAS_R
    global A_prec_SOL_R
    global A_prec_VAS_R
    global A_prec_HAM_R
    global A_prec_GLU_R
    global A_prec_HFL_R

    global lce_prec_TA_L
    global lce_prec_GAS_L
    global lce_prec_SOL_L
    global lce_prec_VAS_L
    global lce_prec_HAM_L
    global lce_prec_GLU_L
    global lce_prec_HFL_L

    global lce_prec_TA_R
    global lce_prec_GAS_R
    global lce_prec_SOL_R
    global lce_prec_VAS_R
    global lce_prec_HAM_R
    global lce_prec_GLU_R
    global lce_prec_HFL_R
    
    
    global elapsed_time1
    global elapsed_time2
    global elapsed_time3
    global elapsed_time4

    
    global flag_initiated
    



    

    
    parameters = mbs_data.user_model
    dt = parameters.get("dt", 0)
    tf = parameters.get("tf", 0)
    flag_graph = parameters.get("flag_graph", 0)


    if (flag_initiated==False):
       flag_initiated=True
       Stance_memory_delayed = np.zeros((round((0.1/dt)),3))
       muscle.set_parameters(parameters)
    flag_initiated=True


    
    #### sensors index :
        
    id_trunk = mbs_data.sensor_id["Sensor_trunk"]
    id_hip = mbs_data.sensor_id["Sensor_hip"]
    
    id_BallL = mbs_data.sensor_id["Sensor_BallL"]
    id_HeelL = mbs_data.sensor_id["Sensor_HeelL"]
    id_BallR = mbs_data.sensor_id["Sensor_BallR"]
    id_HeelR = mbs_data.sensor_id["Sensor_HeelR"]
    
    #### joints index :
        
    id_ankleL = mbs_data.joint_id["ankleL"]
    id_ankleR = mbs_data.joint_id["ankleR"]
    id_kneeL = mbs_data.joint_id["kneeL"]
    id_kneeR = mbs_data.joint_id["kneeR"]
    id_hipL = mbs_data.joint_id["hipL"]
    id_hipR = mbs_data.joint_id["hipR"]
    
    id_innerthighL = mbs_data.joint_id["innerthighL"]
    id_innerthighR = mbs_data.joint_id["innerthighR"]
    
    #### Change of the reference of angles
    
    ankleL_q = mbs_data.q[id_ankleL]
    ankleL_qd = mbs_data.qd[id_ankleL]
    ankleR_q = - mbs_data.q[id_ankleR]
    ankleR_qd = - mbs_data.qd[id_ankleR]
    
    kneeL_q = - mbs_data.q[id_kneeL]
    kneeL_qd = - mbs_data.qd[id_kneeL]
    kneeR_q = mbs_data.q[id_kneeR]
    kneeR_qd = mbs_data.qd[id_kneeR]
    
    hipL_q = mbs_data.q[id_hipL]
    hipL_qd = mbs_data.qd[id_hipL]
    hipR_q = - mbs_data.q[id_hipR]
    hipR_qd = - mbs_data.qd[id_hipR]
    
    
    
    
    #### Trunk angle and angular velocity 
    
    theta_trunk, dtheta_trunk = u_f.trunk_angle(mbs_data.sensors[id_hip].P, mbs_data.sensors[id_trunk].P,mbs_data.sensors[id_hip].V,mbs_data.sensors[id_trunk].V,tsim)
    # MBsysPy.set_output_value(mbs_data.sensors[id_hip].P[3], 1, "hip_position_Z")
    #### Obtention Stance
    
    # control of contact for BallL,HeelL,BallR,HeelR

        
    BallL_cnt,HeelL_cnt,BallR_cnt,HeelR_cnt = u_f.contact_cnt(mbs_data.sensors[id_BallL].P[3], mbs_data.sensors[id_HeelL].P[3], mbs_data.sensors[id_BallR].P[3], mbs_data.sensors[id_HeelR].P[3],ground_limit)
    StanceL,StanceR = u_f.Stance_cnt(BallL_cnt,HeelL_cnt,BallR_cnt,HeelR_cnt)

    #BallL_cnt,HeelL_cnt,BallR_cnt,HeelR_cnt = u_f.force_contact_cnt(mbs_data)

    """     
    global counter
    counter+=1
    a=np.array((BallL_cnt,HeelL_cnt,BallR_cnt,HeelR_cnt))
    b=np.array(u_f.contact_cnt(mbs_data.sensors[id_BallL].P[3], mbs_data.sensors[id_HeelL].P[3], mbs_data.sensors[id_BallR].P[3], mbs_data.sensors[id_HeelR].P[3],ground_limit))
    if(counter%100==0 and np.sum(a-b)!=0):
        print("\n")

        print(tsim)
                
        print(mbs_data.SWr[mbs_data.extforce_id["Force_BallL"]][3])
        print(mbs_data.SWr[mbs_data.extforce_id["Force_HeelL"]][3])
        print(mbs_data.SWr[mbs_data.extforce_id["Force_BallR"]][3])
        print(mbs_data.SWr[mbs_data.extforce_id["Force_HeelR"]][3])

        print(a)
        print(b)
    
    # Stance or not ?
     """
     


    if tsim < 0.002:
        StanceL = 1
    
    #### Ldx/Rdx filtered
    
    dx_innerjointL = mbs_data.q[id_innerthighL]
    # MBsysPy.set_output_value(dx_innerjointL, 1, "Ldx_Rdx_no_filter")
    vx_innerjointL = mbs_data.qd[id_innerthighL]
    F_innerjointL = u_f.pressure_sheet(dx_innerjointL,vx_innerjointL)
    
    dx_innerjointR = - mbs_data.q[id_innerthighR]
    # MBsysPy.set_output_value(dx_innerjointR, 2, "Ldx_Rdx_no_filter")
    vx_innerjointR = - mbs_data.qd[id_innerthighR]
    F_innerjointR = - u_f.pressure_sheet(dx_innerjointR,vx_innerjointR)
    
    # Forces and torques application
    mbs_data.Qq[id_innerthighL] = F_innerjointL
    mbs_data.Qq[id_innerthighR] = F_innerjointR
    
    # Filtre 
    
    if tsim == 0 :
        Ldx = u_f.low_filter(dx_innerjointL, 0.01, 0, Ldx_prec)
        Rdx = u_f.low_filter(dx_innerjointR, 0.01, 0, Rdx_prec)
    else: 
        if tsim!=0 :
            Ldx = u_f.low_filter(dx_innerjointL, 0.02 , dt, Ldx_prec)
            Rdx = u_f.low_filter(dx_innerjointR, 0.02 , dt, Rdx_prec)

        else :
            Ldx = Ldx_prec
            Rdx = Rdx_prec
    
    Ldx_prec = Ldx
    Rdx_prec = Rdx
    
    #### jusque ici ca parait plus ou moins correct
    
    #### update memory Stance, ldx/Rdx filtered, theta-dtheta trunk
    max_ts=2
    
    if tsim == 0:
        
        LDx_memory = np.array([Ldx,tsim])
        RDx_memory = np.array([Rdx,tsim])
        theta_trunk_memory = np.array([theta_trunk, tsim])
        dtheta_trunk_memory = np.array([dtheta_trunk, tsim])
        Stance_memory = np.array([StanceL,StanceR, tsim])
        
        f_m_memoryL = np.array([0, 0, 0, 0, 0, 0, 0, tsim])
        lce_memoryL = np.array([0, 0, 0, 0, 0, 0, 0, tsim])
        f_m_memoryR = np.array([0, 0, 0, 0, 0, 0, 0, tsim])
        lce_memoryR = np.array([0, 0, 0, 0, 0, 0, 0, tsim])
    

    else :
        if tsim!=0 :
            max_ts=round((0.1/dt))+1
            
            
            LDx_memory =  np.vstack([LDx_memory,[Ldx,tsim]])[-max_ts:]
            RDx_memory = np.vstack([RDx_memory, [Rdx, tsim]])[-max_ts:]
            theta_trunk_memory = np.vstack( [theta_trunk_memory, [theta_trunk, tsim]])[-max_ts:]
            dtheta_trunk_memory = np.vstack([dtheta_trunk_memory, [dtheta_trunk, tsim]])[-max_ts:]
            Stance_memory = np.vstack([Stance_memory, [StanceL, StanceR, tsim]])[-max_ts:]
            
            f_m_memoryL = np.vstack([f_m_memoryL, [0, 0, 0, 0, 0, 0, 0, tsim]])[-max_ts:]
            lce_memoryL = np.vstack([lce_memoryL, [0, 0, 0, 0, 0, 0, 0, tsim]])[-max_ts:]
            f_m_memoryR = np.vstack([f_m_memoryR, [0, 0, 0, 0, 0, 0, 0, tsim]])[-max_ts:]
            lce_memoryR = np.vstack([lce_memoryR, [0, 0, 0, 0, 0, 0, 0, tsim]])[-max_ts:]
            
            if tsim >= 0.01:
                    
                Stance_memory_delayed = np.vstack([Stance_memory_delayed, [Stance_memory[-1-round((0.01/dt)),0], Stance_memory[-1-round((0.01/dt)),1], tsim]])[-max_ts:]
            else: 
                Stance_memory_delayed[-1,2] = tsim
                
    
    #### Obtention des stimulations
    if tsim!=0 :

        StimL = neural.Feedback(tsim, dt, Stance_memory_delayed,f_m_memoryL, LDx_memory, RDx_memory, lce_memoryL, kneeL_q, kneeL_qd, theta_trunk_memory, dtheta_trunk_memory,0,parameters)
        StimR = neural.Feedback(tsim, dt, Stance_memory_delayed,f_m_memoryR, RDx_memory, LDx_memory, lce_memoryR, kneeR_q, kneeR_qd, theta_trunk_memory, dtheta_trunk_memory,1,parameters)
                    

    # L = np.hstack([StimL,tsim])
    # R = np.hstack([StimR,tsim])
    # stim_left = StimL
    #stim_left = np.vstack([stim_left,[StimL]])
    # stim_right = np.vstack([stim_right,[StimR]])
    
    # print(np.shape(stim_left))
    #### Compute Fm, lce
    
    #Lmtu update for each muscle
    

    lmtu_TA_L = muscle.lmtu_updateTA(ankleL_q, TA)
    lmtu_GAS_L = muscle.lmtu_updateGAS(ankleL_q,kneeL_q,GAS)
    lmtu_SOL_L = muscle.lmtu_updateSOL(ankleL_q, SOL)
    lmtu_VAS_L = muscle.lmtu_updateVAS(kneeL_q, VAS)
    lmtu_HAM_L = muscle.lmtu_updateHAM(kneeL_q, hipL_q, HAM)
    lmtu_GLU_L = muscle.lmtu_updateGLU(hipL_q, GLU)
    lmtu_HFL_L = muscle.lmtu_updateHFL(hipL_q, HFL)
    
    lmtu_TA_R = muscle.lmtu_updateTA(ankleR_q, TA)
    lmtu_GAS_R = muscle.lmtu_updateGAS(ankleR_q,kneeR_q,GAS)
    lmtu_SOL_R = muscle.lmtu_updateSOL(ankleR_q, SOL)
    lmtu_VAS_R = muscle.lmtu_updateVAS(kneeR_q, VAS)
    lmtu_HAM_R = muscle.lmtu_updateHAM(kneeR_q, hipR_q, HAM)
    lmtu_GLU_R = muscle.lmtu_updateGLU(hipR_q, GLU)
    lmtu_HFL_R = muscle.lmtu_updateHFL(hipR_q, HFL)
    

    if tsim == 0 :

        #TA
        x_0_TA_L = lmtu_TA_L - muscle.l_slack_muscle[TA]
        A_TA_L = u_f.low_filter(StimL[TA],tau,0,0)
        act_memory_TA_L = np.array(([A_TA_L,round(tsim,10)]))
        lmtc_memory_TA_L =np.array(([lmtu_TA_L,round(tsim,10)]))
        lce_TA_L = x_0_TA_L
        Fm_TA_L = muscle.vce_compute(lce_TA_L, lmtu_TA_L, A_TA_L, TA)[1]
        
        x_0_TA_R = lmtu_TA_R - muscle.l_slack_muscle[TA]
        A_TA_R = u_f.low_filter(StimR[TA],tau,0,0)
        act_memory_TA_R = np.array(([A_TA_R,round(tsim,10)]))
        lmtc_memory_TA_R =np.array(([lmtu_TA_R,round(tsim,10)]))
        lce_TA_R = x_0_TA_R
        Fm_TA_R = muscle.vce_compute(lce_TA_R, lmtu_TA_R, A_TA_R, TA)[1]
        
        #GAS
        x_0_GAS_L = lmtu_GAS_L - muscle.l_slack_muscle[GAS]
        A_GAS_L = u_f.low_filter(StimL[GAS],tau,0,0)
        act_memory_GAS_L = np.array(([A_GAS_L,round(tsim,10)]))
        lmtc_memory_GAS_L = np.array(([lmtu_GAS_L,round(tsim,10)]))
        lce_GAS_L = x_0_GAS_L
        Fm_GAS_L = muscle.vce_compute(lce_GAS_L, lmtu_GAS_L, A_GAS_L, GAS)[1]
        
        x_0_GAS_R = lmtu_GAS_R - muscle.l_slack_muscle[GAS]
        A_GAS_R = u_f.low_filter(StimR[GAS],tau,0,0)
        act_memory_GAS_R = np.array(([A_GAS_R,round(tsim,10)]))
        lmtc_memory_GAS_R =np.array(([lmtu_GAS_R,round(tsim,10)]))
        lce_GAS_R = x_0_GAS_R
        Fm_GAS_R = muscle.vce_compute(lce_GAS_R, lmtu_GAS_R, A_GAS_R, GAS)[1]
        
        #SOL
        x_0_SOL_L = lmtu_SOL_L - muscle.l_slack_muscle[SOL]
        A_SOL_L = u_f.low_filter(StimL[SOL],tau,0,0)
        act_memory_SOL_L = np.array(([A_SOL_L,round(tsim,10)]))
        lmtc_memory_SOL_L = np.array(([lmtu_SOL_L,round(tsim,10)]))
        lce_SOL_L = x_0_SOL_L
        Fm_SOL_L = muscle.vce_compute(lce_SOL_L, lmtu_SOL_L, A_SOL_L, SOL)[1]
        
        x_0_SOL_R = lmtu_SOL_R - muscle.l_slack_muscle[SOL]
        A_SOL_R = u_f.low_filter(StimR[SOL],tau,0,0)
        act_memory_SOL_R = np.array(([A_SOL_R,round(tsim,10)]))
        lmtc_memory_SOL_R =np.array(([lmtu_SOL_R,round(tsim,10)]))
        lce_SOL_R = x_0_SOL_R
        Fm_SOL_R = muscle.vce_compute(lce_SOL_R, lmtu_SOL_R, A_SOL_R, SOL)[1]
        
        #VAS
        x_0_VAS_L = lmtu_VAS_L - muscle.l_slack_muscle[VAS]
        A_VAS_L = u_f.low_filter(StimL[VAS],tau,0,0)
        act_memory_VAS_L = np.array(([A_VAS_L,round(tsim,10)]))
        lmtc_memory_VAS_L = np.array(([lmtu_VAS_L,round(tsim,10)]))
        lce_VAS_L = x_0_VAS_L
        Fm_VAS_L = muscle.vce_compute(lce_VAS_L, lmtu_VAS_L, A_VAS_L, VAS)[1]
        
        x_0_VAS_R = lmtu_VAS_R - muscle.l_slack_muscle[VAS]
        A_VAS_R = u_f.low_filter(StimR[VAS],tau,0,0)
        act_memory_VAS_R = np.array(([A_VAS_R,round(tsim,10)]))
        lmtc_memory_VAS_R =np.array(([lmtu_VAS_R,round(tsim,10)]))
        lce_VAS_R = x_0_VAS_R
        Fm_VAS_R = muscle.vce_compute(lce_VAS_R, lmtu_VAS_R, A_VAS_R, VAS)[1]
        
        #HAM
        x_0_HAM_L = lmtu_HAM_L - muscle.l_slack_muscle[HAM]
        A_HAM_L = u_f.low_filter(StimL[HAM],tau,0,0)
        act_memory_HAM_L = np.array(([A_HAM_L,round(tsim,10)]))
        lmtc_memory_HAM_L =np.array(([lmtu_HAM_L,round(tsim,10)]))
        lce_HAM_L = x_0_HAM_L
        Fm_HAM_L = muscle.vce_compute(lce_HAM_L, lmtu_HAM_L, A_HAM_L, HAM)[1]
        
        x_0_HAM_R = lmtu_HAM_R - muscle.l_slack_muscle[HAM]
        A_HAM_R = u_f.low_filter(StimR[HAM],tau,0,0)
        act_memory_HAM_R = np.array(([A_HAM_R,round(tsim,10)]))
        lmtc_memory_HAM_R =np.array(([lmtu_HAM_R,round(tsim,10)]))
        lce_HAM_R = x_0_HAM_R
        Fm_HAM_R = muscle.vce_compute(lce_HAM_R, lmtu_HAM_R, A_HAM_R, HAM)[1]
        
        #GLU
        x_0_GLU_L = lmtu_GLU_L - muscle.l_slack_muscle[GLU]
        A_GLU_L = u_f.low_filter(StimL[GLU],tau,0,0)
        act_memory_GLU_L = np.array(([A_GLU_L,round(tsim,10)]))
        lmtc_memory_GLU_L =np.array(([lmtu_GLU_L,round(tsim,10)]))
        lce_GLU_L = x_0_GLU_L
        Fm_GLU_L = muscle.vce_compute(lce_GLU_L, lmtu_GLU_L, A_GLU_L, GLU)[1]
        
        x_0_GLU_R = lmtu_GLU_R - muscle.l_slack_muscle[GLU]
        A_GLU_R = u_f.low_filter(StimR[GLU],tau,0,0)
        act_memory_GLU_R = np.array(([A_GLU_R,round(tsim,10)]))
        lmtc_memory_GLU_R =np.array(([lmtu_GLU_R,round(tsim,10)]))
        lce_GLU_R = x_0_GLU_R
        Fm_GLU_R = muscle.vce_compute(lce_GLU_R, lmtu_GLU_R, A_GLU_R, GLU)[1]
        
        #HFL
        x_0_HFL_L = lmtu_HFL_L - muscle.l_slack_muscle[HFL]
        A_HFL_L = u_f.low_filter(StimL[HFL],tau,0,0)
        act_memory_HFL_L = np.array(([A_HFL_L,round(tsim,10)]))
        lmtc_memory_HFL_L =np.array(([lmtu_HFL_L,round(tsim,10)]))
        lce_HFL_L = x_0_HFL_L
        Fm_HFL_L = muscle.vce_compute(lce_HFL_L, lmtu_HFL_L, A_HFL_L, HFL)[1]
        
        x_0_HFL_R = lmtu_HFL_R - muscle.l_slack_muscle[HFL]
        A_HFL_R = u_f.low_filter(StimR[HFL],tau,0,0)
        act_memory_HFL_R = np.array(([A_HFL_R,round(tsim,10)]))
        lmtc_memory_HFL_R =np.array(([lmtu_HFL_R,round(tsim,10)]))
        lce_HFL_R = x_0_HFL_R
        Fm_HFL_R = muscle.vce_compute(lce_HFL_R, lmtu_HFL_R, A_HFL_R, HFL)[1]


    
    else:
        if tsim!=0 :


            #TA
            A_TA_L = u_f.low_filter(StimL[TA],tau,dt,A_prec_TA_L)
            act_memory_TA_L = np.vstack([act_memory_TA_L,[A_TA_L,round(tsim,10)]])[-max_ts:]
            lmtc_memory_TA_L =np.vstack([lmtc_memory_TA_L,[lmtu_TA_L,round(tsim,10)]])[-max_ts:]
            lce_TA_L = muscle.integrateur2000(lce_prec_TA_L, dt, 5, lmtc_memory_TA_L[:,0], act_memory_TA_L[:,0], tsim-dt, TA,lmtc_memory_TA_L[:,1])
            
            A_TA_R = u_f.low_filter(StimR[TA],tau,dt,A_prec_TA_R)
            act_memory_TA_R = np.vstack([act_memory_TA_R,[A_TA_R,round(tsim,10)]])[-max_ts:]
            lmtc_memory_TA_R =np.vstack([lmtc_memory_TA_R,[lmtu_TA_R,round(tsim,10)]])[-max_ts:]
            lce_TA_R = muscle.integrateur2000(lce_prec_TA_R, dt, 5, lmtc_memory_TA_R[:,0], act_memory_TA_R[:,0], tsim-dt, TA,lmtc_memory_TA_R[:,1])
            
            #GAS
            A_GAS_L = u_f.low_filter(StimL[GAS],tau,dt,A_prec_GAS_L)
            act_memory_GAS_L = np.vstack([act_memory_GAS_L,[A_GAS_L,round(tsim,10)]])[-max_ts:]
            lmtc_memory_GAS_L =np.vstack([lmtc_memory_GAS_L,[lmtu_GAS_L,round(tsim,10)]])[-max_ts:]
            lce_GAS_L = muscle.integrateur2000(lce_prec_GAS_L, dt, 5, lmtc_memory_GAS_L[:,0], act_memory_GAS_L[:,0], tsim-dt, GAS,lmtc_memory_GAS_L[:,1])
            
            A_GAS_R = u_f.low_filter(StimR[GAS],tau,dt,A_prec_GAS_R)
            act_memory_GAS_R = np.vstack([act_memory_GAS_R,[A_GAS_R,round(tsim,10)]])[-max_ts:]
            lmtc_memory_GAS_R =np.vstack([lmtc_memory_GAS_R,[lmtu_GAS_R,round(tsim,10)]])[-max_ts:]
            lce_GAS_R = muscle.integrateur2000(lce_prec_GAS_R, dt, 5, lmtc_memory_GAS_R[:,0], act_memory_GAS_R[:,0], tsim-dt, GAS,lmtc_memory_GAS_R[:,1])
            
            #SOL
            A_SOL_L = u_f.low_filter(StimL[SOL],tau,dt,A_prec_SOL_L)
            act_memory_SOL_L = np.vstack([act_memory_SOL_L,[A_SOL_L,round(tsim,10)]])[-max_ts:]
            lmtc_memory_SOL_L =np.vstack([lmtc_memory_SOL_L,[lmtu_SOL_L,round(tsim,10)]])[-max_ts:]
            lce_SOL_L = muscle.integrateur2000(lce_prec_SOL_L, dt, 5, lmtc_memory_SOL_L[:,0], act_memory_SOL_L[:,0], tsim-dt, SOL,lmtc_memory_SOL_L[:,1])
            
            A_SOL_R = u_f.low_filter(StimR[SOL],tau,dt,A_prec_SOL_R)
            act_memory_SOL_R = np.vstack([act_memory_SOL_R,[A_SOL_R,round(tsim,10)]])[-max_ts:]
            lmtc_memory_SOL_R =np.vstack([lmtc_memory_SOL_R,[lmtu_SOL_R,round(tsim,10)]])[-max_ts:]
            lce_SOL_R = muscle.integrateur2000(lce_prec_SOL_R, dt, 5, lmtc_memory_SOL_R[:,0], act_memory_SOL_R[:,0], tsim-dt, SOL,lmtc_memory_SOL_R[:,1])
            
            #VAS
            A_VAS_L = u_f.low_filter(StimL[VAS],tau,dt,A_prec_VAS_L)
            act_memory_VAS_L = np.vstack([act_memory_VAS_L,[A_VAS_L,round(tsim,10)]])[-max_ts:]
            lmtc_memory_VAS_L =np.vstack([lmtc_memory_VAS_L,[lmtu_VAS_L,round(tsim,10)]])[-max_ts:]
            lce_VAS_L = muscle.integrateur2000(lce_prec_VAS_L, dt, 8, lmtc_memory_VAS_L[:,0], act_memory_VAS_L[:,0], tsim-dt, VAS,lmtc_memory_VAS_L[:,1])
            
            A_VAS_R = u_f.low_filter(StimR[VAS],tau,dt,A_prec_VAS_R)
            act_memory_VAS_R = np.vstack([act_memory_VAS_R,[A_VAS_R,round(tsim,10)]])[-max_ts:]
            lmtc_memory_VAS_R =np.vstack([lmtc_memory_VAS_R,[lmtu_VAS_R,round(tsim,10)]])[-max_ts:]
            lce_VAS_R = muscle.integrateur2000(lce_prec_VAS_R, dt, 5, lmtc_memory_VAS_R[:,0], act_memory_VAS_R[:,0], tsim-dt, VAS,lmtc_memory_VAS_R[:,1])
            
            #HAM
            A_HAM_L = u_f.low_filter(StimL[HAM],tau,dt,A_prec_HAM_L)
            act_memory_HAM_L = np.vstack([act_memory_HAM_L,[A_HAM_L,round(tsim,10)]])[-max_ts:]
            lmtc_memory_HAM_L =np.vstack([lmtc_memory_HAM_L,[lmtu_HAM_L,round(tsim,10)]])[-max_ts:]
            lce_HAM_L = muscle.integrateur2000(lce_prec_HAM_L, dt, 5, lmtc_memory_HAM_L[:,0], act_memory_HAM_L[:,0], tsim-dt, HAM,lmtc_memory_HAM_L[:,1])
            
            A_HAM_R = u_f.low_filter(StimR[HAM],tau,dt,A_prec_HAM_R)
            act_memory_HAM_R = np.vstack([act_memory_HAM_R,[A_HAM_R,round(tsim,10)]])[-max_ts:]
            lmtc_memory_HAM_R =np.vstack([lmtc_memory_HAM_R,[lmtu_HAM_R,round(tsim,10)]])[-max_ts:]
            lce_HAM_R = muscle.integrateur2000(lce_prec_HAM_R, dt, 5, lmtc_memory_HAM_R[:,0], act_memory_HAM_R[:,0], tsim-dt, HAM,lmtc_memory_HAM_R[:,1])
            
            #GLU
            A_GLU_L = u_f.low_filter(StimL[GLU],tau,dt,A_prec_GLU_L)
            act_memory_GLU_L = np.vstack([act_memory_GLU_L,[A_GLU_L,round(tsim,10)]])[-max_ts:]
            lmtc_memory_GLU_L =np.vstack([lmtc_memory_GLU_L,[lmtu_GLU_L,round(tsim,10)]])[-max_ts:]
            lce_GLU_L = muscle.integrateur2000(lce_prec_GLU_L, dt, 5, lmtc_memory_GLU_L[:,0], act_memory_GLU_L[:,0], tsim-dt, GLU,lmtc_memory_GLU_L[:,1])
            
            A_GLU_R = u_f.low_filter(StimR[GLU],tau,dt,A_prec_GLU_R)
            act_memory_GLU_R = np.vstack([act_memory_GLU_R,[A_GLU_R,round(tsim,10)]])[-max_ts:]
            lmtc_memory_GLU_R =np.vstack([lmtc_memory_GLU_R,[lmtu_GLU_R,round(tsim,10)]])[-max_ts:]
            lce_GLU_R = muscle.integrateur2000(lce_prec_GLU_R, dt, 5, lmtc_memory_GLU_R[:,0], act_memory_GLU_R[:,0], tsim-dt, GLU,lmtc_memory_GLU_R[:,1])
            
            #HFL
            A_HFL_L = u_f.low_filter(StimL[HFL],tau,dt,A_prec_HFL_L)
            act_memory_HFL_L = np.vstack([act_memory_HFL_L,[A_HFL_L,round(tsim,10)]])[-max_ts:]
            lmtc_memory_HFL_L =np.vstack([lmtc_memory_HFL_L,[lmtu_HFL_L,round(tsim,10)]])[-max_ts:]
            lce_HFL_L = muscle.integrateur2000(lce_prec_HFL_L, dt, 5, lmtc_memory_HFL_L[:,0], act_memory_HFL_L[:,0], tsim-dt, HFL,lmtc_memory_HFL_L[:,1])
            
            A_HFL_R = u_f.low_filter(StimR[HFL],tau,dt,A_prec_HFL_R)
            act_memory_HFL_R = np.vstack([act_memory_HFL_R,[A_HFL_R,round(tsim,10)]])[-max_ts:]
            lmtc_memory_HFL_R =np.vstack([lmtc_memory_HFL_R,[lmtu_HFL_R,round(tsim,10)]])[-max_ts:]
            lce_HFL_R = muscle.integrateur2000(lce_prec_HFL_R, dt, 5, lmtc_memory_HFL_R[:,0], act_memory_HFL_R[:,0], tsim-dt, HFL,lmtc_memory_HFL_R[:,1])
            

                
        else :
            
            #TA
            A_TA_L = A_prec_TA_L
            lce_TA_L = lce_prec_TA_L
            
            A_TA_R = A_prec_TA_R
            lce_TA_R = lce_prec_TA_R
            
            #GAS
            A_GAS_L = A_prec_GAS_L
            lce_GAS_L = lce_prec_GAS_L
            
            A_GAS_R = A_prec_GAS_R
            lce_GAS_R = lce_prec_GAS_R
            
            #SOL
            A_SOL_L = A_prec_SOL_L
            lce_SOL_L = lce_prec_SOL_L
            
            A_SOL_R = A_prec_SOL_R
            lce_SOL_R = lce_prec_SOL_R
                     
            #VAS
            A_VAS_L = A_prec_VAS_L
            lce_VAS_L = lce_prec_VAS_L
            
            A_VAS_R = A_prec_VAS_R
            lce_VAS_R = lce_prec_VAS_R
            
            #HAM
            A_HAM_L = A_prec_HAM_L
            lce_HAM_L = lce_prec_HAM_L
            
            A_HAM_R = A_prec_HAM_R
            lce_HAM_R = lce_prec_HAM_R
            
            #GLU
            A_GLU_L = A_prec_GLU_L
            lce_GLU_L = lce_prec_GLU_L
            
            A_GLU_R = A_prec_GLU_R
            lce_GLU_R = lce_prec_GLU_R
            
            #HFL
            A_HFL_L = A_prec_HFL_L
            lce_HFL_L = lce_prec_HFL_L
            
            A_HFL_R = A_prec_HFL_R
            lce_HFL_R = lce_prec_HFL_R
        
        start=time.time()

        #TA
        Fm_TA_L = muscle.vce_compute(lce_TA_L, lmtu_TA_L, A_TA_L, TA)[1]
        Fm_TA_R = muscle.vce_compute(lce_TA_R, lmtu_TA_R, A_TA_R, TA)[1]
        
        #GAS
        Fm_GAS_L = muscle.vce_compute(lce_GAS_L, lmtu_GAS_L, A_GAS_L, GAS)[1]
        Fm_GAS_R = muscle.vce_compute(lce_GAS_R, lmtu_GAS_R, A_GAS_R, GAS)[1]
        
        #SOL
        Fm_SOL_L = muscle.vce_compute(lce_SOL_L, lmtu_SOL_L, A_SOL_L, SOL)[1]
        Fm_SOL_R = muscle.vce_compute(lce_SOL_R, lmtu_SOL_R, A_SOL_R, SOL)[1]
        
        #VAS
        Fm_VAS_L = muscle.vce_compute(lce_VAS_L, lmtu_VAS_L, A_VAS_L, VAS)[1]
        Fm_VAS_R = muscle.vce_compute(lce_VAS_R, lmtu_VAS_R, A_VAS_R, VAS)[1]

        #HAM
        Fm_HAM_L = muscle.vce_compute(lce_HAM_L, lmtu_HAM_L, A_HAM_L, HAM)[1]
        Fm_HAM_R = muscle.vce_compute(lce_HAM_R, lmtu_HAM_R, A_HAM_R, HAM)[1]
        
        #GLU
        Fm_GLU_L = muscle.vce_compute(lce_GLU_L, lmtu_GLU_L, A_GLU_L, GLU)[1]
        Fm_GLU_R = muscle.vce_compute(lce_GLU_R, lmtu_GLU_R, A_GLU_R, GLU)[1]
        
        #HFL
        Fm_HFL_L = muscle.vce_compute(lce_HFL_L, lmtu_HFL_L, A_HFL_L, HFL)[1]
        Fm_HFL_R = muscle.vce_compute(lce_HFL_R, lmtu_HFL_R, A_HFL_R, HFL)[1]

    #TA
    A_prec_TA_L = A_TA_L
    lce_prec_TA_L = lce_TA_L
    
    A_prec_TA_R = A_TA_R
    lce_prec_TA_R = lce_TA_R
    
    #GAS
    A_prec_GAS_L = A_GAS_L
    lce_prec_GAS_L = lce_GAS_L
    
    A_prec_GAS_R = A_GAS_R
    lce_prec_GAS_R = lce_GAS_R
    
    #SOL
    A_prec_SOL_L = A_SOL_L
    lce_prec_SOL_L = lce_SOL_L
    
    A_prec_SOL_R = A_SOL_R
    lce_prec_SOL_R = lce_SOL_R
    
    #VAS
    A_prec_VAS_L = A_VAS_L
    lce_prec_VAS_L = lce_VAS_L
    
    A_prec_VAS_R = A_VAS_R
    lce_prec_VAS_R = lce_VAS_R
    
    #HAM
    A_prec_HAM_L = A_HAM_L
    lce_prec_HAM_L = lce_HAM_L
    
    A_prec_HAM_R = A_HAM_R
    lce_prec_HAM_R = lce_HAM_R
    
    #GLU
    A_prec_GLU_L = A_GLU_L
    lce_prec_GLU_L = lce_GLU_L
    
    A_prec_GLU_R = A_GLU_R
    lce_prec_GLU_R = lce_GLU_R
    
    #HFL
    A_prec_HFL_L = A_HFL_L
    lce_prec_HFL_L = lce_HFL_L
    
    A_prec_HFL_R = A_HFL_R
    lce_prec_HFL_R = lce_HFL_R
    
    ### update memory Fm, lce
    if tsim == 0 :
        f_m_memoryL[VAS]= Fm_VAS_L
        f_m_memoryL[SOL]= Fm_SOL_L
        f_m_memoryL[GAS]= Fm_GAS_L
        f_m_memoryL[TA]= Fm_TA_L
        f_m_memoryL[HAM]= Fm_HAM_L
        f_m_memoryL[GLU]= Fm_GLU_L
        f_m_memoryL[HFL]= Fm_HFL_L
        
        lce_memoryL[VAS]= lce_VAS_L
        lce_memoryL[SOL]= lce_SOL_L
        lce_memoryL[GAS]= lce_GAS_L
        lce_memoryL[TA]= lce_TA_L
        lce_memoryL[HAM]= lce_HAM_L
        lce_memoryL[GLU]= lce_GLU_L
        lce_memoryL[HFL]= lce_HFL_L
        
        f_m_memoryR[VAS]= Fm_VAS_R
        f_m_memoryR[SOL]= Fm_SOL_R
        f_m_memoryR[GAS]= Fm_GAS_R
        f_m_memoryR[TA]= Fm_TA_R
        f_m_memoryR[HAM]= Fm_HAM_R
        f_m_memoryR[GLU]= Fm_GLU_R
        f_m_memoryR[HFL]= Fm_HFL_R
        
        lce_memoryR[VAS]= lce_VAS_R
        lce_memoryR[SOL]= lce_SOL_R
        lce_memoryR[GAS]= lce_GAS_R
        lce_memoryR[TA]= lce_TA_R
        lce_memoryR[HAM]= lce_HAM_R
        lce_memoryR[GLU]= lce_GLU_R
        lce_memoryR[HFL]= lce_HFL_R
        
    else :
        if tsim !=0:
        
            f_m_memoryL[-1,VAS]= Fm_VAS_L
            f_m_memoryL[-1,SOL]= Fm_SOL_L
            f_m_memoryL[-1,GAS]= Fm_GAS_L
            f_m_memoryL[-1,TA]= Fm_TA_L
            f_m_memoryL[-1,HAM]= Fm_HAM_L
            f_m_memoryL[-1,GLU]= Fm_GLU_L
            f_m_memoryL[-1,HFL]= Fm_HFL_L
        
            lce_memoryL[-1,VAS]= lce_VAS_L
            lce_memoryL[-1,SOL]= lce_SOL_L
            lce_memoryL[-1,GAS]= lce_GAS_L
            lce_memoryL[-1,TA]= lce_TA_L
            lce_memoryL[-1,HAM]= lce_HAM_L
            lce_memoryL[-1,GLU]= lce_GLU_L
            lce_memoryL[-1,HFL]= lce_HFL_L
            
            f_m_memoryR[-1,VAS]= Fm_VAS_R
            f_m_memoryR[-1,SOL]= Fm_SOL_R
            f_m_memoryR[-1,GAS]= Fm_GAS_R
            f_m_memoryR[-1,TA]= Fm_TA_R
            f_m_memoryR[-1,HAM]= Fm_HAM_R
            f_m_memoryR[-1,GLU]= Fm_GLU_R
            f_m_memoryR[-1,HFL]= Fm_HFL_R
        
            lce_memoryR[-1,VAS]= lce_VAS_R
            lce_memoryR[-1,SOL]= lce_SOL_R
            lce_memoryR[-1,GAS]= lce_GAS_R
            lce_memoryR[-1,TA]= lce_TA_R
            lce_memoryR[-1,HAM]= lce_HAM_R
            lce_memoryR[-1,GLU]= lce_GLU_R
            lce_memoryR[-1,HFL]= lce_HFL_R
    
    #### calcul des torques
    #start=time.time()

    Torque_ankle_TA_L = muscle.torque_updateANKLE(ankleL_q,Fm_TA_L,TA)
    Torque_ankle_GAS_L = muscle.torque_updateANKLE(ankleL_q,Fm_GAS_L,GAS)
    Torque_knee_GAS_L =muscle.torque_updateKNEE(kneeL_q, Fm_GAS_L, GAS)
    Torque_ankle_SOL_L = muscle.torque_updateANKLE(ankleL_q,Fm_SOL_L,SOL)
    Torque_knee_VAS_L= muscle.torque_updateKNEE(kneeL_q,Fm_VAS_L,VAS)
    Torque_knee_HAM_L= muscle.torque_updateKNEE(kneeL_q,Fm_HAM_L,HAM)
    Torque_hip_HAM_L=muscle.torque_updateHIP(hipL_q, Fm_HAM_L, HAM)
    Torque_hip_GLU_L=muscle.torque_updateHIP(hipL_q, Fm_GLU_L, GLU)
    Torque_hip_HFL_L=muscle.torque_updateHIP(hipL_q, Fm_HFL_L, HFL)
    
    Torque_ankle_TA_R = muscle.torque_updateANKLE(ankleR_q,Fm_TA_R,TA)
    Torque_ankle_GAS_R = muscle.torque_updateANKLE(ankleR_q,Fm_GAS_R,GAS)
    Torque_knee_GAS_R =muscle.torque_updateKNEE(kneeR_q, Fm_GAS_R, GAS)
    Torque_ankle_SOL_R = muscle.torque_updateANKLE(ankleR_q,Fm_SOL_R,SOL)
    Torque_knee_VAS_R= muscle.torque_updateKNEE(kneeR_q,Fm_VAS_R,VAS)
    Torque_knee_HAM_R= muscle.torque_updateKNEE(kneeR_q,Fm_HAM_R,HAM)
    Torque_hip_HAM_R=muscle.torque_updateHIP(hipR_q, Fm_HAM_R, HAM)
    Torque_hip_GLU_R=muscle.torque_updateHIP(hipR_q, Fm_GLU_R, GLU)
    Torque_hip_HFL_R=muscle.torque_updateHIP(hipR_q, Fm_HFL_R, HFL)
    
    Torque=[Torque_ankle_TA_L,Torque_ankle_GAS_L ,  Torque_knee_GAS_L ,  Torque_ankle_SOL_L ,
     Torque_knee_VAS_L ,  Torque_knee_HAM_L ,  Torque_hip_HAM_L ,  Torque_hip_GLU_L ,  Torque_hip_HFL_L ,
     Torque_ankle_TA_R ,  Torque_ankle_GAS_R ,  Torque_knee_GAS_R ,  Torque_ankle_SOL_R ,
     Torque_knee_VAS_R ,  Torque_knee_HAM_R ,  Torque_hip_HAM_R ,  Torque_hip_GLU_R ,  Torque_hip_HFL_R ]
    
    Fm = [Fm_TA_L, Fm_GAS_L, Fm_GAS_L, Fm_SOL_L, Fm_VAS_L, Fm_HAM_L, Fm_HAM_L, Fm_GLU_L, Fm_HFL_L, Fm_TA_R, Fm_GAS_R, Fm_GAS_R, Fm_SOL_R, Fm_VAS_R, Fm_HAM_R, Fm_HAM_R, Fm_GLU_R, Fm_HFL_R]
    
    
    if(tsim!=0):
        act =[ act_memory_TA_L [-1,0], act_memory_GAS_L  [-1,0], act_memory_SOL_L  [-1,0], act_memory_VAS_L  [-1,0],act_memory_HAM_L  [-1,0],
           act_memory_GLU_L [-1,0], act_memory_HFL_L  [-1,0],act_memory_TA_R   [-1,0], act_memory_GAS_R  [-1,0],act_memory_SOL_R  [-1,0],act_memory_VAS_R  [-1,0],act_memory_HAM_R  [-1,0],act_memory_GLU_R  [-1,0],act_memory_HFL_R  [-1,0]]
    else:
        act =[0.01]*14
        

    stim = np.append(StimL,StimR)
    stance = np.append(StanceL,StanceR)
    
    
    
    if(flag_graph):
        gait_graph.collect_muscle(Torque,Fm,act,stim,stance,tsim)

    
    #### joint limits 
    jl_ankle_L=muscle.joint_limits(ankle, ankleL_q, ankleL_qd)
    jl_knee_L=muscle.joint_limits(knee, kneeL_q, kneeL_qd)
    jl_hip_L=muscle.joint_limits(hip, hipL_q, hipL_qd)
    
    jl_ankle_R=muscle.joint_limits(ankle, ankleR_q, ankleR_qd)
    jl_knee_R=muscle.joint_limits(knee, kneeR_q, kneeR_qd)
    jl_hip_R=muscle.joint_limits(hip, hipR_q, hipR_qd)
    
    Torque_ankle_L = jl_ankle_L + Torque_ankle_GAS_L + Torque_ankle_SOL_L - Torque_ankle_TA_L
    # print(Torque_ankle_L)
    Torque_knee_L = jl_knee_L + Torque_knee_VAS_L - Torque_knee_GAS_L - Torque_knee_HAM_L
    Torque_hip_L = jl_hip_L + Torque_hip_GLU_L + Torque_hip_HAM_L - Torque_hip_HFL_L
    
    Torque_ankle_R = jl_ankle_R + Torque_ankle_GAS_R + Torque_ankle_SOL_R - Torque_ankle_TA_R
    Torque_knee_R = jl_knee_R + Torque_knee_VAS_R - Torque_knee_GAS_R - Torque_knee_HAM_R
    Torque_hip_R = jl_hip_R + Torque_hip_GLU_R + Torque_hip_HAM_R - Torque_hip_HFL_R
    
    #### Application of torques

    mbs_data.Qq[id_ankleL] = Torque_ankle_L
    mbs_data.Qq[id_kneeL] = - Torque_knee_L
    mbs_data.Qq[id_hipL] = Torque_hip_L
    
    mbs_data.Qq[id_ankleR] = - Torque_ankle_R 
    mbs_data.Qq[id_kneeR] = Torque_knee_R
    mbs_data.Qq[id_hipR] = - Torque_hip_R
    
    
    




    if(tf<tsim+dt*3 and flag_graph):
        gait_graph.show_ext()
        
        flag_graph=False 
        
    

        
        
    
    """ StanceL,StanceR = u_f.Stance_cnt(BallL_cnt,HeelL_cnt,BallR_cnt,HeelR_cnt)
    
    pos_hip   = mbs_data.sensors[id_hip].P[3]
    pos_trunk = mbs_data.sensors[id_trunk].Px
    
    metrics.register(StanceL,StanceR,kneeL_q,kneeR_q, theta_trunk, pos_trunk, pos_hip, tsim, parameters)
    """

    
    #pos_hip   = mbs_data.sensors[id_hip].P[3]
    #pos_trunk = mbs_data.sensors[id_trunk].P
    #metrics.register(StanceL,StanceR,kneeL_q,kneeR_q, theta_trunk, pos_trunk, pos_hip, tsim, parameters)
    
    
    global prev_time  
    global prev_datetime
      
    time_between_measure = 0.1
    
    if tsim >= prev_time + time_between_measure:
        
        #survived time
        mbs_data.user_model["fitness"] -= 2
        
        #cost of transport (estimated max 10000, objectif que la metric E ]0,1] )
        mbs_data.user_model["fitness"] += np.sum(Fm)/10000
        
        #objective speed 
        mbs_data.user_model["fitness"] += abs( mbs_data.sensors[id_hip].P[1]-tsim*1.3 )
        #metrics.register(StanceL,StanceR,kneeL_q,kneeR_q, theta_trunk, pos_trunk, pos_hip, tsim, parameters)
    
        mbs_data.user_model["fitness_memory"][round(tsim/time_between_measure)] = mbs_data.user_model["fitness"]
        mbs_data.user_model["fm_memory"][round(tsim/time_between_measure)] =  np.sum(Fm)/10000
    
        prev_time=tsim
        
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        time_diff = int((now - prev_datetime).total_seconds())
        print("\n",round(tsim,2), " fitness ", round(mbs_data.user_model["fitness"]),  " ct:", current_time, "I", time_diff, "s")
        print("FM" ,round(np.sum(Fm)/10000,3) , "Dist Target", round(( mbs_data.sensors[id_hip].P[1]-tsim*1.3 ),3), "speed", round(abs( mbs_data.sensors[id_hip].P[1]/tsim),3))
        prev_datetime = now
        
        
        np.save("fitness_id"+str(parameters.get("id", 0)),mbs_data.user_model["fitness"])
        #check if disqualified
        if(abs( mbs_data.sensors[id_hip].P[1]-tsim*1.3 )>0.4):
            print("DISQUALIFIED: Outside allowed area", flush=True)
            mbs_data.Qq[2] = np.inf
            
            
        pos_hip   = mbs_data.sensors[id_hip].P[3]
        if(pos_hip > -0.75):
            print("DISQUALIFIED: Hip too low", mbs_data.sensors[id_hip].P[3],flush=True)
            mbs_data.Qq[2] = np.inf
            

        if(theta_trunk < 0 or theta_trunk>0.4):
            print("DISQUALIFIED: trunk angle outside allowed range",flush=True)
            mbs_data.Qq[2] = np.inf

        print("hip",round(mbs_data.sensors[id_hip].P[3],3) , "trunk" ,round(theta_trunk,3))

  




        
    
    
    """     if tsim % 5 == 0 and prev_time.strftime("%H:%M:%S") != datetime.now().strftime("%H:%M:%S") :
            
            fitness=metrics.test_fitness(tsim)
            
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            time_diff = int((now - prev_time).total_seconds())
            print(tsim, " fitness ", round(fitness),  " ct:", current_time, "Interval", time_diff, "s", flush=True)
            prev_time = now 
            mbs_data.user_model["fitness"] = fitness
            np.save("fitness_id"+str(parameters.get("id", 0)),fitness)
            
            if(fitness > fitness_threshold):
                mbs_data.user_model["fitness"] = fitness+100*(tf-tsim)/dt
                np.save("fitness_id"+str(parameters.get("id", 0)), fitness+100*(tf-tsim)/dt)

                print("FITNESS THRESHOLD REACHED , crash it ",fitness+100*(tf-tsim)/dt)
                mbs_data.Qq[2] = np.inf


    
    print(tsim, round(tsim,8) % 0.01,  round(tsim % 0.01,8))
    if tsim % 0.01 == 0 and tsim>0.02:
        print(tsim)
        input("input")
        #print(mbs_data.sensors[id_hip].P[1]-tsim*1.3, mbs_data.sensors[id_hip].P[1])
    
    
    """
    return



import os
import sys 
import TestworkR

# Get the directory where your script is located
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(2,  os.path.join(parent_dir, "workR"))
if __name__ == "__main__":
    TestworkR.runtest(1000e-7,2,c=False)
