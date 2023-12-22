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


""" dt = 1000e-7
tf = 6
F_max_alpha=0.778
v_max_alpha=0.8

flag_graph=False

name="fitness_data/f"+str(F_max_alpha)+"v"+str(v_max_alpha)+"tf"+str(tf)

# Initialize empty lists or load existing ones from files
memory_fitness = np.load(str(name)+"memory_fitness.npy", allow_pickle=True).tolist() if str(name)[13:]+"memory_fitness.npy" in os.listdir("fitness_data") else []
memory_suggestion = np.load(str(name)+"memory_suggestion.npy", allow_pickle=True).tolist() if str(name)[13:]+"memory_suggestion.npy" in os.listdir("fitness_data") else []

np.save(str(name)+"memory_fitness.npy", np.array(memory_fitness[:-1]))
np.save(str(name)+"memory_suggestion.npy", np.array(memory_suggestion[:-1]))


import matplotlib.pyplot as plt
plt.plot(memory_fitness)
plt.show() """


dt = 1000e-7
tf = 20
F_max_alpha=0
v_max_alpha=0

flag_graph=False

name="fitness_data/Bayesian"+"tf"+str(tf)




# Initialize empty lists or load existing ones from files
memory_fitness = np.load(str(name)+"memory_fitness.npy", allow_pickle=True).tolist() if str(name)[13:]+"memory_fitness.npy" in os.listdir("fitness_data") else []
memory_suggestion = np.load(str(name)+"memory_suggestion.npy", allow_pickle=True).tolist() if str(name)[13:]+"memory_suggestion.npy" in os.listdir("fitness_data") else []
import matplotlib.pyplot as plt
import numpy as np

low, high = np.zeros(len(memory_fitness)), np.zeros(len(memory_fitness))
low[0], high[-1] = memory_fitness[0], memory_fitness[-1]

for i in range(1, len(memory_fitness)):
    low[i] = min(low[i-1], memory_fitness[i]) if low[i-1] > memory_fitness[i] else low[i-1]

for i in range(len(memory_fitness)-2, -1, -1):
    high[i] = max(high[i+1], memory_fitness[i]) if high[i+1] < memory_fitness[i] else high[i+1]

generation = np.arange(len(memory_fitness))
coefficients = np.polyfit(generation, memory_fitness, 1)
fit_line = np.polyval(coefficients, generation)
print("Pente:", coefficients[0])


plt.plot(generation, memory_fitness, label='Memory Fitness')
plt.plot(generation, low, label='Low')
plt.plot(generation, fit_line, label='Regression line Line')
plt.plot(generation, high, label='High')
#plt.savefig("squares.png") 

plt.savefig("plot/fitness"+name[13:]+"ft"+str(low[-1])+".png")
plt.grid()
plt.legend()




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
        "flag_graph": False,
        "fitness_thresold": 10e10,#placeholder to not be triggered
        "fitness_memory": np.zeros(200),
        "fm_memory": np.zeros(200),
        "fitness":  2*200,
        "v_gx_max": 0.03,
        "v_gz_max": 0.03,
        "kz": 78480,
        "kx": 7848,
        "must": 0.9,
        "musl": 0.8,
        "v_limit": 0.01
    }

    if(overide_parameters!=False):
        parameters=overide_parameters
        dt0=parameters['dt']
        tf=parameters['tf']
        


    
    #np.save("parameters", parameters)
    print(parameters,flush=True)
    mbs_part.run()
    
    mbs_data.user_model=parameters
    # ===========================================================================
    # Direct Dynamics
    # =============================================================================
    mbs_data.process = 3    
    mbs_dirdyn = Robotran.MbsDirdyn(mbs_data)
    from datetime import datetime

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time, flush=True)
    
    

        
    
    mbs_dirdyn.set_options(dt0=dt0, tf=tf, save2file=1)#, integrator="Bader") # 96
    
    start_time = time.time()

    try:
        results = mbs_dirdyn.run()
    except:
        print("yoooooo")
    
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
    print(parameters)
    
    print("fitness " , np.load("fitness.npy"))





if __name__ == "__main__":
    parameters = {
        "dt": 1000e-7,
        "tf": 1,
        "flag_graph": False,
        "v_gx_max": 0.03,
        "v_gz_max": 0.03,
        "kz": 108480,
        "kx": 10848,
        "must": 0.9,
        "musl": 0.8,
        "v_limit": 0.01
    }



    #runtest(10000e-7,5,parameters)
    

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
 