#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import os
os.system('clear')
os.system('python --version')
import sys
print(sys.executable)


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



def fitness_calculator(parameters_pakaged):
    mbs_data = Robotran.MbsData('../dataR/Fullmodel_innerjoint.mbs',)
    mbs_data.process = 1
    mbs_part = Robotran.MbsPart(mbs_data)
    mbs_part.set_options(rowperm=1, verbose=1)
    
    
    global dt
    global tf 
    global flag_graph    
    global fitness_thresold
     
    parameters = {
        "dt": dt,
        "tf": tf,
        "flag_graph": flag_graph,
        "fitness_thresold": fitness_thresold,
        "fitness": 0,
        "v_gx_max": parameters_pakaged[0],
        "v_gz_max": parameters_pakaged[1],
        "kz": parameters_pakaged[2],
        "kx": parameters_pakaged[3],
        "must": parameters_pakaged[4],
        "musl": parameters_pakaged[5],
        "v_limit": parameters_pakaged[6]
        
        }

    
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
    
    mbs_dirdyn.set_options(dt0=dt, tf=tf, save2file=1)#, integrator="Bader") # 96
    start_time = time.time()

    results = mbs_dirdyn.run()
    
    

    elapsed_time = time.time() - start_time

    elapsed_time_minutes = round(elapsed_time/60 , 3)

    print(f"Time taken to run the line: {elapsed_time_minutes:.2f} minutes")


    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    now = str(datetime.now())[:19]
    now = now.replace(":","_")

    fitness= float(np.load("fitness_id"+str(id)+".npy")) 

    
    src_dir=parent_dir+"/animationR/dirdyn_q.anim"
    dst_dir=parent_dir+"/animationR/archive/tf:"+str(tf)+"dt0"+str(dt)+"rt"+str(elapsed_time_minutes)+"ft"+str(np.round(fitness))+".anim"

    shutil.copy(src_dir,dst_dir)

    return fitness



fitness_thresold =  10e10#2000000000 #change it 




dt = 2500e-7
tf = 0.54


name="fitness_data/Ground" + str(dt)

 

flag_graph=False


import numpy as np
from skopt import Optimizer
from skopt.space import Real


v_gx_max = 0.03
v_gz_max = 0.03
kz = 90000
kx = 7800
must = 0.9
musl = 0.8
v_limit = 0.01
        
# Define the parameter bounds
a = 0.4
b = 2.5 # Adjust upper bound as needed

# Define the parameter space for Bayesian optimization
space = [
    Real(v_gx_max * a, v_gx_max * b, name='v_gx_max', prior='uniform', transform='normalize'),
    Real( v_gz_max * a,  v_gz_max * b, name=' v_gz_max', prior='uniform', transform='normalize'),
    Real( kz * a,  kz * b, name=' kz', prior='uniform', transform='normalize'),
    Real( kx * a,  kx * b, name=' kx', prior='uniform', transform='normalize'),
    Real( must * a,  must * b, name=' must', prior='uniform', transform='normalize'),
    Real(musl * a, musl * b, name='musl', prior='uniform', transform='normalize'),
    Real(v_limit * a, v_limit * b, name='v_limit', prior='uniform', transform='normalize')
]

best = [0.028255368998069727, 0.03881766968949231, 74036.16471541824, 9750.88523726196,
        0.7085848423301961, 0.577162570567703, 0.00772486492389127]


v_gx_max = best[0] 
v_gz_max = best[1] 
kz = best[2] 
kx = best[3] 
must = best[4] 
musl = best[5] 
v_limit = best[6] 




# Initialize empty lists or load existing ones from files
memory_fitness = np.load(str(name)+"memory_fitness.npy", allow_pickle=True).tolist() if str(name)[13:]+"memory_fitness.npy" in os.listdir("fitness_data") else []
memory_suggestion = np.load(str(name)+"memory_suggestion.npy", allow_pickle=True).tolist() if str(name)[13:]+"memory_suggestion.npy" in os.listdir("fitness_data") else []

print(memory_fitness)

# Create the optimizer
optimizer = Optimizer(space,base_estimator="GBRT", acq_func="PI",random_state=42)


for i in range (len(memory_fitness)):
    optimizer.tell(memory_suggestion[i],memory_fitness[i])
    print(i)



import matplotlib.pyplot as plt
print("show")
plt.plot(memory_fitness)
plt.show()

# Run Bayesian optimization
while(True):
    suggestion = optimizer.ask()
    
    fitness = fitness_calculator(suggestion)
    print("\n",len(memory_fitness),"fitness", fitness)
    optimizer.tell(suggestion, fitness)
    result = optimizer.get_result()

    # Append values to lists
    memory_fitness.append(fitness)
    memory_suggestion.append(suggestion)

    # Save lists
    np.save(str(name)+"memory_fitness.npy", np.array(memory_fitness))
    np.save(str(name)+"memory_suggestion.npy", np.array(memory_suggestion))

    print("\nBest results:", result.x)
    print("Best Fitness:", result.fun)

# Get the best result from Bayesian optimization
result = optimizer.get_result()

print("Best results:",result.x)
print("Best Fitness:", result.fun)


