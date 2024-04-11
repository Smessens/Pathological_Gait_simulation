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



def fitness_calculator(parameters_pakaged,id=0 , best_fitness_memory = np.ones(200)*10):
    mbs_data = Robotran.MbsData('../dataR/Fullmodel_innerjoint.mbs',)
    mbs_data.process = 1
    mbs_part = Robotran.MbsPart(mbs_data)
    mbs_part.set_options(rowperm=1, verbose=1)
    
    
    global dt
    global tf 
    global flag_graph    
     

    
    
    parameters = {
        "dt": dt,
        "tf": tf,
        "flag_graph": True,
        "id": id, 

        "flag_fitness" : False, 
        "best_fitness_memory":  best_fitness_memory ,
        "fitness_memory":  np.ones(200)*tf*10,
        "fm_memory": np.zeros(200),
        "fitness": tf*10,
        

        "v_gx_max": 0.03,
        "v_gz_max": 0.03,
        "kz": 90000,
        "kx":7800,
        "must": 0.9,
        "musl": 0.8,
        "v_limit": 0.01,
        
        
        "G_VAS": parameters_pakaged.get("G_VAS", 2e-4),
        "G_SOL" : parameters_pakaged.get("G_SOL", 1.2 / 4000),
        "G_GAS" : parameters_pakaged.get("G_GAS", 1.1 / 1500),
        "G_TA" : parameters_pakaged.get("G_TA", 1.1),
        "G_SOL_TA" : parameters_pakaged.get("G_SOL_TA", 0.0001),
        "G_HAM" : parameters_pakaged.get("G_HAM", 2.166666666666667e-04),
        "G_GLU" : parameters_pakaged.get("G_GLU", 1 / 3000.),
        "G_HFL" : parameters_pakaged.get("G_HFL", 0.5),
        "G_HAM_HFL" : parameters_pakaged.get("G_HAM_HFL", 4),
        "G_delta_theta" :  parameters_pakaged.get("G_delta_theta", 1.145915590261647),

        "theta_ref" : parameters_pakaged.get("theta_ref", 0.104719755119660),
        "k_swing" : parameters_pakaged.get("k_swing", 0.25),
        
        "k_p" : parameters_pakaged.get("k_p", 1.909859317102744),
        "k_d" : parameters_pakaged.get("k_d", 0.2),
        "phi_k_off": parameters_pakaged.get("phi_k_off", 2.967059728390360),
        
        "loff_TA" : parameters_pakaged.get("loff_TA", 0.72),
        "loff_HAM" : parameters_pakaged.get("loff_HAM", 0.85),
        "loff_HFL" : parameters_pakaged.get("loff_HFL", 0.65)
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

    try:
        results = mbs_dirdyn.run()
        
    except:
        print("Manually Crashed")
    
    

    elapsed_time = time.time() - start_time

    elapsed_time_minutes = round(elapsed_time/60 , 3)

    print(f"Time taken to run the line: {elapsed_time_minutes:.2f} minutes")


    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    now = str(datetime.now())[:19]
    now = now.replace(":","_")

    fitness= float(np.load("fitness_id"+str(id)+".npy"))
    fitness_memory = np.load("fitness_memory"+str(id)+".npy")

    
    import platform
    # do not trigger in colab
    if platform.system() == 'Darwin':  # Darwin is the system name for macOS
        src_dir=parent_dir+"/animationR/dirdyn_q.anim"
        dst_dir=parent_dir+"/animationR/archive/tf:"+str(tf)+"dt0"+str(dt)+"ft"+str(np.round(fitness,2))+"rt"+str(elapsed_time_minutes)+".anim"
        shutil.copy(src_dir,dst_dir)

    return fitness , fitness_memory



import numpy as np
from skopt import Optimizer
from skopt import forest_minimize
from skopt.space import Real

import matplotlib.pyplot as plt
import numpy as np
#fitness_thresold = 20000 #change it 




dt = 1000e-7
tf = 60

F_max_alpha=0
v_max_alpha=0

flag_graph=False



specific_parameters = {
    'G_VAS': [4.9e-4, 5 * 2.1e-4],  # original: 2e-4
    'G_SOL': [0.000275,  0.0004],  # original: 1.2 / 4000
    'G_GAS': [0.2 * 1.05 / 1500, 2 * 1.15 / 1500],  # original: 1.1 / 1500
    'G_TA': [2,4.5],  # original: 1.1
    'G_SOL_TA': [0.8 * 9e-5, 1.2 * 1.1e-4],  # original: 0.0001
    'G_HAM': [0.2 * 2e-4, 1.2 * 2.3e-4],  # original: 2.166666666666667e-04
    'G_GLU': [0.3 * 0.95 / 3000, 0.85 / 3000],  # original: 1 / 3000.
    'G_HFL': [0.7 * 0.45, 2 * 0.55],  # original: 0.5
    'G_HAM_HFL': [5,12],  # original: 4
    'G_delta_theta': [0.4 , 2],  # original: 1.145915590261647
    'theta_ref': [0.162, 0.22],  # original: 0.104719755119660

}


space = [ Real(specific_parameters[k][0],specific_parameters[k][1], prior='uniform', transform='normalize') for k in specific_parameters.keys() ] 

best_fitness_memory = np.ones(200)*10*tf
best_fitness_session = tf



suggestion= [0.0006820967664971395, 0.00035349204804382885, 0.0007515645757628333, 3.715356814772865, 0.00012735605555843976, 0.00020004060562716971, 0.00013087472798702774, 0.9172262105998026, 8.723161175630677, 1.0209169546721757, 0.1999103493115993]

results = fitness_calculator({k: v for k, v in zip(list(specific_parameters.keys()), suggestion)}, best_fitness_memory=best_fitness_memory)
