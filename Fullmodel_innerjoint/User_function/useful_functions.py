#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:51:51 2022

@author: matthieuxaussems
"""

import numpy as np
import math
from scipy import signal 
import MBsysPy


################################ Usefuls functions ####################################################################################

"""
    Function that allows to determine the angle and the angular speed of the trunk in relation to the vertical
    
    input : position of the hip joint and of a point on the trunk thanks to the sensors in the MBsyspad
            sensor_hip with PxF = [0,x,y,z] and VxF = [0,vx,vy,vz]
            sensor_trunk with PxF = [0,x,y,z] and VxF = [0,vx,vy,vz]
    out : angle in radian between the bust and the vertical and angular speed of the trunk
"""

def trunk_angle(P_sensor_hip, P_sensor_trunk,V_sensor_hip,V_sensor_trunk,tsim):
    

    
    dp = P_sensor_trunk[1:] - P_sensor_hip[1:]
    dx= dp[0]
    dz= dp[2]
        
    dv = V_sensor_trunk[1:] - V_sensor_hip[1:]
    dvx = dv[0]
    dvz = dv[2]
        
        
    angle = math.atan2(dx, -dz)
    dangle = (-dz*dvx+dx*dvz)/(0.8)**2
      
    return angle,dangle





def contact_cnt(BallL_positionz,HeelL_positionz,BallR_positionz,HeelR_positionz,ground_limit):
    
    if BallL_positionz-ground_limit >=0: 
        BallL_cnt=1
    else:
        BallL_cnt=0
        
        
    if BallR_positionz-ground_limit >=0:
        BallR_cnt=1
    else:
        BallR_cnt=0
        
        
    if HeelR_positionz-ground_limit >=0:
        HeelR_cnt=1
    else:
        HeelR_cnt=0
        
        
    if HeelL_positionz-ground_limit >=0:
        HeelL_cnt=1
    else:
        HeelL_cnt=0
        

    return BallL_cnt,HeelL_cnt,BallR_cnt,HeelR_cnt


#important : z forces are inversed 

def force_contact_cnt(mbs_data):
    
    threshold=-10 
   
    
    if  mbs_data.SWr[mbs_data.extforce_id["Force_BallL"]][3] < threshold: 
        BallL_cnt=1
    else:
        BallL_cnt=0
        
        
    if  mbs_data.SWr[mbs_data.extforce_id["Force_BallR"]][3] < threshold: 
        BallR_cnt=1
    else:
        BallR_cnt=0
        
        
    if  mbs_data.SWr[mbs_data.extforce_id["Force_HeelR"]][3] < threshold: 
        HeelR_cnt=1
    else:
        HeelR_cnt=0
        
        
    if  mbs_data.SWr[mbs_data.extforce_id["Force_HeelL"]][3] < threshold: 
        HeelL_cnt=1
    else:
        HeelL_cnt=0
        

    return BallL_cnt,HeelL_cnt,BallR_cnt,HeelR_cnt


def Stance_cnt(BallL_cnt,HeelL_cnt,BallR_cnt,HeelR_cnt):
    
    if BallL_cnt == 1 or HeelL_cnt ==1 :
        
        StanceL = 1
    
    else:
        
        StanceL = 0
    
    if BallR_cnt == 1 or HeelR_cnt ==1 :
        
        StanceR = 1
    
    else:
        
        StanceR = 0
    
    return StanceL,StanceR



def pressure_sheet(p,v):
    k_pressure = 104967
    v_max_pressure = 0.5
    u1 = k_pressure*p
    u2 = v/v_max_pressure
    
    F = -u1* (1+math.copysign(1, u1)*u2)
    
    return F



def low_filter(stimulation,tau,diff_t,last_activation):

    if diff_t==0 : #lorsque current_t=last_t alors l'activation est égale à la stimulation car delta-t = 0 dans l'équation différentielle ce qui revient
                   #à Stimulation(t)-activation(t)=0 donc stimulation=activation
        activation = stimulation
        return activation
    
    else :
        f = diff_t/tau
        frac = 1/(1+f)
        activation = f*frac*stimulation +frac*last_activation
        return activation






        
import sys
import os
# Get the directory where your script is located
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(2,  os.path.join(parent_dir, "workR"))
import TestworkR


if __name__ == "__main__":
    TestworkR.runtest(250e-7,0.05,c=False)
    
    

