#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
"""Script to run a direct dynamic analysis on a multibody system.

Summary
-------
This template loads the data file *.mbs and execute:
 - the coordinate partitioning module
 - the direct dynamic module (time integration of equations of motion).
 - if available, plot the time evolution of the first generalized coordinate.

It may have to be adapted and completed by the user.


Universite catholique de Louvain
CEREM : Centre for research in mechatronics

http://www.robotran.eu
Contact : info@robotran.be

(c) Universite catholique de Louvain
"""


import os




os.system('clear')
#os.system('conda init')
#os.system('conda info --envs')
#os.system('conda activate my_x86_env')

os.system('python --version')
import sys
print(sys.executable)

'''
conda activate  /Users/messenssimon/anaconda3/envs/my_x86_env
pip install --index-url https://www.robotran.eu/dist/ MBsysPy --user

conda create -n x86_env -y
conda activate x86_env
conda config --env --set subdir osx-64
conda install python="3.10"



'''
# ============================================================================
# Packages loading
# =============================================================================
start_time = time.time()
try:
    import MBsysPy as Robotran
except:
    raise ImportError("MBsysPy not found/installed."
                   
                      "See: https://www.robotran.eu/download/how-to-install/"
                      )
    

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0,  os.path.join(parent_dir, "User_function"))
sys.path.insert(1,  os.path.join(parent_dir, "userfctR"))


# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__))) # the script is running from it's parent directory ?


# ===========================================================================
# Project loading
# =============================================================================
import pandas as pd
import time
import shutil
import numpy as np
import json


def runtest(dt0,tf,overide_parameters=False,c=False):
    mbs_data = Robotran.MbsData('../dataR/Fullmodel_innerjoint.mbs',)
    # ===========================================================================
    # Partitionning
    # =============================================================================
    mbs_data.process = 1
    mbs_part = Robotran.MbsPart(mbs_data)
    mbs_part.set_options(rowperm=1, verbose=1)
    
    parameters = {
        "dt": dt0,
        "tf": tf,
        "flag_graph": True,
        "id": 0, 

        "flag_fitness" : False, 
        "best_fitness_memory":  np.ones(200)*100,
        "fitness_memory":  np.ones(200)*100,
        "fm_memory": np.zeros(200),
        "fitness":  100,
        
        "fitness_thresold": 10e10,#placeholder to not be triggered

        
        "v_gx_max": 0.03,
        "v_gz_max": 0.03,
        "kz": 78480,
        "kx": 7848,
        "must": 0.9,
        "musl": 0.8,
        "v_limit": 0.01
    }


        


    
    #np.save("parameters", parameters)
    #print(parameters,flush=True)
    #input()
    mbs_part.run()
    
    # ===========================================================================
    # Direct Dynamics
    # =============================================================================
    mbs_data.process = 3    
    mbs_dirdyn = Robotran.MbsDirdyn(mbs_data)
    mbs_data.user_model=parameters



    from datetime import datetime

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time, flush=True)
    
    
    mbs_dirdyn.set_options(dt0=dt0, tf=tf, save2file=1)#, integrator="Bader") # 96
    start_time = time.time()

    try:
        results = mbs_dirdyn.run()
        
    except:
        print("Manually Crashed")
    
    elapsed_time = time.time() - start_time



    elapsed_time = time.time() - start_time

    elapsed_time_minutes = round(elapsed_time/60 , 3)

    print(f"Time taken to run the line: {elapsed_time_minutes:.2f} minutes")



    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    now = str(datetime.now())[:19]
    now = now.replace(":","_")

    tf=tf
    dt0=dt0
    src_dir=parent_dir+"/animationR/dirdyn_q.anim"
    dst_dir=parent_dir+"/animationR/archive/tf:"+str(tf)+"dt0"+str(dt0)+"rt"+str(elapsed_time_minutes)+".anim"

    shutil.copy(src_dir,dst_dir)
    #print(parameters)
    
    #print("fitness " , np.load("fitness_id0.npy"))





if __name__ == "__main__":
    parameters = {
        "dt": 1000e-7,
        "tf": 1,
        "flag_graph": True,
        "v_gx_max": 0.03,
        "v_gz_max": 0.03,
        "kz": 108480,
        "kx": 10848,
        "must": 0.9,
        "musl": 0.8,
        "v_limit": 0.01
    }

    
    
    runtest(125e-7,3,parameters)
    

""" 

import winsound    
winsound.Beep(1440, 200)    

import numpy as np

arr= np.zeros(2)

for i in range (100):
    arr=np.vstack([arr,[i,i]])[-20:]

print(arr)


 import numpy as np
print(np.arange(100)[:200])

dt=round(250e-7,10)
print(dt)

relevant_timesteps=int(0.21/dt)
import numpy as np

a=np.arange(100000)
print(a[-relevant_timesteps:])


counter=0
    global counter
    counter+=1
    if(counter%10000==0):
        print(time[-1],t)
    
                global elapsed_time
            start=time.time()
                        elapsed_time += time.time()-start
            if(tsim>0.095):
                print("Elapsed time"+str(elapsed_time))
 """
 