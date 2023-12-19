
    
import os
import sys 
import matplotlib.pyplot as plt
import numpy as np


import os
import time
# Get the directory where your script is located
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0,  os.path.join(parent_dir, "User_function"))
sys.path.insert(1,  os.path.join(parent_dir, "userfctR"))
sys.path.insert(2,  os.path.join(parent_dir, "workR"))
import TestworkR
import math




t=0

M1=0
M2=0
M3=0
M4=0
M5=0

Stance=0
dt=0
tf=0
flag_initiated=False


def metrics_initiate(parameters):
    
    global flag_initiated
    global dt , tf , t
    global M1 , M2, M3 , M4 , M5, Stance

    
    
    if (flag_initiated==False):
        flag_initiated=True
        dt = parameters.get("dt", 0)
        tf = parameters.get("tf", 0)
        t = np.arange(0, tf+ dt, dt)  # Create the time array directly
        l = len(t)  # Calculate the length based on t
        M1= np.zeros((2,l))
        M2= np.zeros((4,l))
        M3= np.zeros((l))
        M4= np.zeros((l))
        M5= np.zeros((l))

        Stance= np.zeros((2,l))


def register(StanceL,StanceR,kneeL_q,kneeR_q, theta_trunk, p_trunk , pos_hip, tsim , parameters):

    global flag_initiated
    global M1 , M2, M3 , M4 , M5, Stance
    
    
    if (flag_initiated==False):
        metrics_initiate(parameters)

    dt = parameters.get("dt", 0)
    ti = int(tsim/dt)
    
    M1[:,ti] = [kneeL_q,kneeR_q]
    M3[ti]   = pos_hip
    M4[ti]   = theta_trunk
    M5[ti]   = p_trunk[1]
    
    Stance[:,ti] = [StanceL,StanceR]


def register_ankle(pos_z,type, tsim, parameters):
    #"BallR", "HeelL", "HeelR", "BallL"
    global flag_initiated
    global M1 , M2, M3 , M4 , M5, Stance
    
    
    if (flag_initiated==False):
        metrics_initiate(parameters)
        
    ti= int(tsim/dt)
    M2[type,ti] = pos_z
    
    
def metrics_calculate(M1 , M2, M3 , M4 , M5, Stance ,dt , tf , t):

    sum_M1=0
    thresM1h=3.3
    thresM1L=1.5
    for i in range(len(t)):
        for j in range(2):
            if(Stance[j,i]):  
                if(M1[j,i]>thresM1h or M1[j,i] < thresM1L):
                    sum_M1+=1
                    
    
                    
    sum_M2=0
    angle=0
    for i in range(1, len(Stance[0])-5):
        for j in range (2):
            #BallL , HellL , HeelR , BallR
            if(j==0):
                angle=math.tan((M2[1,i]-M2[0,i])/20)
            else:
                angle=math.tan((M2[2,i]-M2[3,i])/20)
        
            if Stance[j,i-1] == 0 and Stance[j,i] == 1 and angle < 0: #1.50 is parrallel to ground
                sum_M2+=0.001/dt

            if Stance[j,i ] == 1 and Stance[j,i+1] == 0 and angle > 0:
                sum_M2+=0.001/dt
                
                
            
    sum_M3   = 0
    thresM3h = -0.7
    
    for i in range(len(t)):
        if(M3[i]>thresM3h):
            sum_M3 += (len(t)-i)*1000
            break
        

                
    
    sum_M4=0
    thresM4h=0.4
    thresM4L=0
    for i in range(len(t)):
        if(M4[i]>thresM4h or M4[i] < thresM4L):
            sum_M4+=1
                                   
    
    sum_M5=0
    for i in range(len(t)-5):
        sum_M5+= abs(M5[i]-1.1*dt*i)
        
                
    print(round(sum_M1),round(sum_M2),round(sum_M3),round(sum_M4),round(sum_M5))
    #np.save("fitness",sum_M1+sum_M2+sum_M3+sum_M4+sum_M5)
    return sum_M1+sum_M2+sum_M3+sum_M4+sum_M5
    

     
    
    
def plot():
    global dt , tf , t
    global M1 , M2, M3 , M4 , M5, Stance

    np.save("numpy_archive/Metrics/Stance",Stance)
    
    np.save("numpy_archive/Metrics/M1",M1)
    np.save("numpy_archive/Metrics/M2",M2)
    np.save("numpy_archive/Metrics/M3",M3)
    np.save("numpy_archive/Metrics/M4",M4)
    np.save("numpy_archive/Metrics/M5",M5)
    
    return metrics_calculate(M1 , M2, M3 , M4 , M5, Stance ,dt , tf , t)

                    

    


def plot_loaded():
    
    dt = np.load("parameters.npy", allow_pickle=True)[()].get("dt", 0)
    tf = np.load("parameters.npy", allow_pickle=True)[()].get("tf", 0)
    t = np.arange(0, tf+ dt, dt)  # Create the time array directly
       
    M1 =  np.load("numpy_archive/Metrics/M1.npy")
    M2 =  np.load("numpy_archive/Metrics/M2.npy")
    M3 =  np.load("numpy_archive/Metrics/M3.npy")
    M4 =  np.load("numpy_archive/Metrics/M4.npy")
    M5 =  np.load("numpy_archive/Metrics/M5.npy")

    Stance =  np.load("numpy_archive/Metrics/Stance.npy")
    metrics_calculate(M1 , M2, M3 , M4 , M5, Stance ,dt , tf , t)


    
    
def test_fitness(tsim):
    global dt , tf 
    global M1 , M2, M3 , M4 , M5, Stance
    t = np.arange(0, tsim + dt, dt)

    return   metrics_calculate(M1 , M2, M3 , M4 , M5, Stance ,dt , tf , t)
    

    

if __name__ == "__main__":
    TestworkR.runtest(1000e-7,3)
    #plot_loaded()  
       









