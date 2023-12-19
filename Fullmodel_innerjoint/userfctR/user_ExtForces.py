# -*- coding: utf-8 -*-
"""Module for defining user function required to compute external forces."""
# Author: Robotran Team
# (c) Universite catholique de Louvain, 2019


import math
import numpy as np
import MBsysPy
# Useful data for the contact force model

import sys

import os
import time
# Get the directory where your script is located
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0,  os.path.join(parent_dir, "User_function"))
sys.path.insert(1,  os.path.join(parent_dir, "userfctR"))
import gait_graph
import metrics

ground_limit = 0
v_gx_max = 0.03
v_gz_max = 0.03
kz = 78480
kx = 7848
must = 0.9
musl = 0.8
v_limit = 0.01


Q_ballL=1
Qn_ballL=0
Stiction_test_ballL=0
Stiction_prec_test_ballL=0 
prec_slide_test_ballL=0
prec_stick_test_ballL=0

Q_heelL=1
Qn_heelL=0
Stiction_test_heelL=0
Stiction_prec_test_heelL=0 
prec_slide_test_heelL=0
prec_stick_test_heelL=0

x0_ballL = 0
x0_heelL = 0

Q_ballR=1
Qn_ballR=0
Stiction_test_ballR=0
Stiction_prec_test_ballR=0 
prec_slide_test_ballR=0
prec_stick_test_ballR=0

Q_heelR=1
Qn_heelR=0
Stiction_test_heelR=0
Stiction_prec_test_heelR=0 
prec_slide_test_heelR=0
prec_stick_test_heelR=0

x0_ballR = 0
x0_heelR = 0


Q=np.array([1,1,1,1,1])
Qn=np.zeros(5)
Stiction=np.array([0,0,0,0,0])
Stiction_test=np.zeros(5)
Stiction_prec_test=np.zeros(5)
Sliding=np.zeros(5)
Sticking=np.zeros(5)
x0 = np.zeros(5)
PosFP = np.zeros(5)

flag_graph = True
flag_initiated = False
# Useful function for the contact force model

def flip_flop_SR(S, R, Q, Qn):
    if S and not R: # Set
        Q = 1
        Qn = 0
    elif R and not S: # Reset
        Q = 0
        Qn = 1
    elif not R and not S: # Hold
        Q=Q
        Qn=Qn
    else: # Invalid input
        print("Invalid input: S and R cannot be both 1")
    return Q, Qn

def sr_flip_flop(S, R, current_state):
    if S and not R:  # Set
        return True
    elif R and not S:  # Reset
        return False
    else:  # No change
        return current_state
    




    
flag_fitness=False


def user_ExtForces(PxF, RxF, VxF, OMxF, AxF, OMPxF, mbs_data, tsim, ixF):
    

    """Compute an user-specified external force.

    Parameters
    ----------
    PxF : numpy.ndarray
        Position vector (index starting at 1) of the force sensor expressed in
        the inertial frame: PxF[1:4] = [P_x, P_y, P_z]
    RxF : numpy.ndarray
        Rotation matrix (index starting at 1) from the inertial frame to the
        force sensor frame: Frame_sensor = RxF[1:4,1:4] * Frame_inertial
    VxF : numpy.ndarray
        Velocity vector (index starting at 1) of the force sensor expressed in
        the inertial frame: VxF[1:4] = [V_x, V_y, V_z]
    OMxF : numpy.ndarray
        Angular velocity vector (index starting at 1) of the force sensor
        expressed in the inertial frame: OMxF[1:4] = [OM_x, OM_y, OM_z]
    AxF : numpy.ndarray
        Acceleration vector (index starting at 1) of the force sensor expressed
        in the inertial frame: AxF[1:4] = [A_x, A_y, A_z]
    OMPxF : numpy.ndarray
        Angular acceleration vector (index starting at 1) of the force sensor
        expressed in the inertial frame: OMPxF[1:4] = [OMP_x, OMP_y, OMP_z]
    mbs_data : MBsysPy.MbsData
        The multibody system associated to this computation.
    tsim : float
        The current time of the simulation.
    ixF : int
        The ID identifying the computed force sensor.

    Notes
    -----
    For 1D numpy.ndarray with index starting at 1, the first index (array[0])
    must not be modified. The first index to be filled is array[1].

    For 2D numpy.ndarray with index starting at 1, the first row (mat[0, :]) and
    line (mat[:,0]) must not be modified. The subarray to be filled is mat[1:, 1:].

    Returns
    -------
    Swr : numpy.ndarray
        An array of length 10 equal to [0., Fx, Fy, Fz, Mx, My, Mz, dxF].
        F_# are the forces components expressed in inertial frame.
        M_# are the torques components expressed in inertial frame.
        dxF is an array of length 3 containing the component of the forces/torque
        application point expressed in the BODY-FIXED frame.
        This array is a specific line of MbsData.SWr.
    """

    #if(tsim<0.0002):
    #   print(v_gx_max,np.load("parameters.npy", allow_pickle=True)[()].get("v_gx_max", 2e-4),tsim)
    #   print(flag_initiated )
    
       
    parameters = mbs_data.user_model

    flag_graph = parameters.get("flag_graph", 0)
    v_gx_max = parameters.get("v_gx_max", 0.03)
    v_gz_max = parameters.get("v_gz_max", 0.03)
    kz = parameters.get("kz", 78480)
    kx = parameters.get("kx", 7848)
    must = parameters.get("must", 0.9)
    musl = parameters.get("musl", 0.9)
    v_limit = parameters.get("v_limit",0.01)
        
        #fitness_threshold = np.load("parameters.npy", allow_pickle=True)[()].get("fitness_threshold", 10e10)
    




    #global flag_fitness
    
    #if tsim % 0.01 == 0 and tsim != 0:
    #    flag_fitness=np.load("flag_fitness.npy")
    #    print(flag_fitness)
    
    #Resetting variables
    Fx = 0.0
    Fy = 0.0
    Fz = 0.0
    Mx = 0.0
    My = 0.0
    Mz = 0.0
    

    
    idpt = mbs_data.xfidpt[ixF]
    dxF = mbs_data.dpt[1:, idpt]
    
    #if(flag_fitness):  # to stop failure
    #    print("stoped",tsim)
    #    Swr = mbs_data.SWr[ixF]
    #    Swr[1:] = [Fx, Fy, -Fz, Mx, My, Mz, 0.0, 0.0, 0.0]
    #    return
    
    #Initialization of external force sensors:
    
    Force_BallL = mbs_data.extforce_id["Force_BallL"]
    Force_HeelL = mbs_data.extforce_id["Force_HeelL"]
    Force_BallR = mbs_data.extforce_id["Force_BallR"]
    Force_HeelR = mbs_data.extforce_id["Force_HeelR"]
    
    

    
    global Q
    global Qn
    global Stiction
    global Stiction_test
    global Stiction_prec_test
    global Sliding
    global Sticking
    global x0
    global PosFP
    
    MBsysPy.set_output_value(PxF[3], ixF, "pos_Z")
    MBsysPy.set_output_value(VxF[1], ixF, "velocity_X")
    MBsysPy.set_output_value(VxF[3], ixF, "velocity_Z")
    
    #pos_z,type, tsim)
    # tsim, 
    metrics.register_ankle(PxF[3],ixF-1,tsim , parameters)
    Fx_sticking = 0
    Fx_sliding  = 0
    test_slide  = 0
    test_stick  = 0
    dx=0
    dvx=0
    
    #PxF z are negative for positive 
    
    """ 
    v_gx_max = 0.03
    v_gz_max = 0.03
    kz = 78480
    kx = 7848
    must = 0.9
    musl = 0.8
    v_limit = 0.01 """
 
        
    

    
        

    
    dz = -(PxF[3])

    if(dz < 0 ):
    
        #print(ixF,dz,VxF[3])
        dz_k=    dz*kz 
        vz_max = -VxF[3]/v_gz_max
        
        if(vz_max<1):
            Fz= - dz_k*(1-vz_max)
        else:
            Fz=0
        
        dx  = (PxF[1]-PosFP[ixF])*kx
            
        if(Stiction[ixF]==1):
            dx  = (PxF[1]-PosFP[ixF])*kx
            #print(dx,tsim)
            dvx = VxF[1]/v_gx_max
            
            
            Fx_sticking = -dx *(1 + math.copysign(1, dx) * dvx) 
            
            if(Fz==0):
                Fx_sticking=0
                
            test_slide=abs(Fx_sticking) - abs(Fz*must)
            
            if(test_slide>0):
                Sliding[ixF]=1
            else:
                Sliding[ixF]=0
                
                
            Sticking[ixF]=0
                
                    
        
        else:
            Fx_sliding = -math.copysign(1,VxF[1])*Fz*musl
            
            test_stick=(abs(VxF[1])-v_limit)
            
            if(test_stick<0):
                Sticking[ixF]=1
            else:
                Sticking[ixF]=0                
                   
            PosFP[ixF]=PxF[1]
            Sliding[ixF]=0
            
        
             
        Fx=Fx_sliding+Fx_sticking
        
        Stiction[ixF]= sr_flip_flop(Sticking[ixF],Sliding[ixF],Stiction[ixF])

    else:
        Stiction[ixF]=0
        Fx = 0
        Fz = 0 
        
    
        
                
        

    # Concatenating force, torque and force application point to returned array.
    # This must not be modified.
    Swr = mbs_data.SWr[ixF]
    Swr[1:] = [Fx, Fy, -Fz, Mx, My, Mz, dxF[0], dxF[1], -dxF[2]]
    
    
    Fx_HeelR = mbs_data.SWr[Force_HeelR][1]
    Fz_HeelR = mbs_data.SWr[Force_HeelR][3]
    
    Fx_HeelL = mbs_data.SWr[Force_HeelL][1]
    Fz_HeelL = mbs_data.SWr[Force_HeelL][3]
    
    Fx_BallR = mbs_data.SWr[Force_BallR][1]
    Fz_BallR = mbs_data.SWr[Force_BallR][3]
    
    Fx_BallL = mbs_data.SWr[Force_BallL][1]
    Fz_BallL = mbs_data.SWr[Force_BallL][3]
    
    
    MBsysPy.set_output_value(Fx_HeelR, 1, "external_force_X")
    MBsysPy.set_output_value(Fx_HeelL, 2, "external_force_X")
    MBsysPy.set_output_value(Fx_BallR, 3, "external_force_X")
    MBsysPy.set_output_value(Fx_BallL, 4, "external_force_X")


    MBsysPy.set_output_value(Fz_HeelR, 1, "external_force_Z")
    MBsysPy.set_output_value(Fz_HeelL, 2, "external_force_Z")
    MBsysPy.set_output_value(Fz_BallR, 3, "external_force_Z")
    MBsysPy.set_output_value(Fz_BallL, 4, "external_force_Z")
    
    GRF = [PxF[1],dz,VxF[1],VxF[3],Sticking[ixF],Sliding[ixF],Stiction[ixF] if (dz<= 0) else -1,PosFP[ixF],dx,dvx,Fx,Fz,Fx_sliding,Fx_sticking,test_slide,test_stick]
    
    



    if(flag_graph):
        gait_graph.collect_ext(ixF,GRF,tsim)

    return Swr






# (c) Universite catholique de Louvain, 2020
import sys

import os
import time
# Get the directory where your script is located
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0,  os.path.join(parent_dir, "User_function"))
sys.path.insert(1,  os.path.join(parent_dir, "userfctR"))
sys.path.insert(2,  os.path.join(parent_dir, "workR"))


import TestworkR


if __name__ == "__main__":
    TestworkR.runtest(250e-7,0.05,c=False)    




