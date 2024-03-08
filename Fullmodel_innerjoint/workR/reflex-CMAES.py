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
        "flag_graph": flag_graph,
        "id": id, 

        "flag_fitness" : True, 
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
        
        
        "G_VAS": parameters_pakaged[0],
        "G_SOL" : parameters_pakaged[1],
        "G_GAS" : parameters_pakaged[2],
        "G_TA" : parameters_pakaged[3],
        "G_SOL_TA" : parameters_pakaged[4],
        "G_HAM" : parameters_pakaged[5],
        "G_GLU" : parameters_pakaged[6],
        "G_HFL" : parameters_pakaged[7],
        "G_HAM_HFL" : parameters_pakaged[8],
        "G_delta_theta" :  parameters_pakaged[9],

        "theta_ref" :  parameters_pakaged[10],
        "k_swing" : parameters_pakaged[11],
        
        "k_p" : parameters_pakaged[12],
        "k_d" : parameters_pakaged[13],
        "phi_k_off": parameters_pakaged[14],
        
        "loff_TA" : parameters_pakaged[15],
        "lopt_TA" : parameters_pakaged[16],
        "loff_HAM" : parameters_pakaged[17],
        "lopt_HAM": parameters_pakaged[18],
        "loff_HFL" : parameters_pakaged[19],
        "lopt_HFL" : parameters_pakaged[20]    
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

    
    src_dir=parent_dir+"/animationR/dirdyn_q.anim"
    dst_dir=parent_dir+"/animationR/archive/tf:"+str(tf)+"dt0"+str(dt)+"ft"+str(np.round(fitness,2))+"rt"+str(elapsed_time_minutes)+".anim"

    shutil.copy(src_dir,dst_dir)

    return fitness , fitness_memory



import numpy as np
from skopt import Optimizer
from skopt import forest_minimize
from skopt.space import Real

#fitness_thresold = 20000 #change it 




dt = 1000e-7
tf = 5
F_max_alpha=0
v_max_alpha=0

flag_graph=False

name="fitness_data/validationV3_tf"+str(tf)

if os.path.exists(str(name)+"memory_fitness.npy"):

    # Initialize empty lists or load existing ones from files
    memory_fitness = np.load(str(name)+"memory_fitness.npy", allow_pickle=True).tolist() if str(name)[13:]+"memory_fitness.npy" in os.listdir("fitness_data") else []
    memory_fitness_breakdown = np.load(str(name)+"memory_fitness_breakdown.npy", allow_pickle=True).tolist() if str(name)[13:]+"memory_fitness.npy" in os.listdir("fitness_data") else []
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
    plt.savefig(name+"ft"+str(low[-1])+".png")
    plt.grid()
    plt.legend()
    #plt.show()

else:
    
    memory_fitness = []
    memory_fitness_breakdown = []
    memory_suggestion = []
    




# Define initial parameter values
initial_G_VAS = 2e-4
initial_G_SOL = 1.2 / 4000
initial_G_GAS = 1.1 / 1500
initial_G_TA = 1.1
initial_G_SOL_TA = 0.0001
initial_G_HAM = 2.166666666666667e-04
initial_G_GLU = 1 / 3000.
initial_G_HFL = 0.5
initial_G_HAM_HFL = 4
initial_G_delta_theta = 1.145915590261647

initial_theta_ref = 0.104719755119660
initial_k_swing = 0.26
initial_k_p = 1.909859317102744
initial_k_d = 0.2
initial_phi_k_off = 2.967059728390360

# Offset parameters with default values
initial_loff_TA =  0.72
initial_lopt_TA =  0.06
initial_loff_HAM = 0.85
initial_lopt_HAM = 0.10
initial_loff_HFL = 0.65
initial_lopt_HFL = 0.11

# Pre-stimulation parameters with default values
""" initial_So     = 0.01
initial_So_VAS = 0.08
initial_So_BAL = 0.05 """







#fitness , fitness_memory = fitness_calculator(best,best_fitness_memory=np.ones(200)*400) # evaluate points in parallel


""" 
 19.801  fitness  141  ct: 11:43:41 I 6 s
FM 0.355 Dist Target 0.317 speed 1.284


Best Fitness: 139.7854923078354

"""   
        
#best = [0.00023725448394776403, 0.0003146167106615956, 0.0005371782039878938, 1.0996407314305618, 0.00011943960621704116, 0.00023329286319415322, 0.00026820579487105313, 0.6228140997410778, 3.832458473316, 1.3350141507149667, 0.114269443189251, 0.3250687765967386, 2.467205206542504, 0.24572574715990603, 2.9245508819983925, 0.763941391504197, 0.04360965734734056, 0.9899210250212649, 0.1214108697986023, 0.7398225640695794, 0.12264733904567945, 0.011612226392038539, 0.06628084057764845, 0.05059880840670715] 

#Best results: [0.0002473427103517884, 0.000321598903277249, 0.0005415670503787884, 1.1491988235031547, 0.00011568345238729708, 0.0002359577594866942, 0.00027039944338621235, 0.6363467451923275, 3.8191765845692824, 1.3918911218951076, 0.1097470780740622, 0.3411216324380119, 2.5626565999650044, 0.24280857238021, 3.0482447495840566, 0.7854981903756897, 0.042728312084087644, 0.9781972461671607, 0.12528465604623568, 0.7218363247445218, 0.1173802928264711, 0.01147217528636999, 0.06649227576523722, 0.05048190684594042]
#Best Fitness: 115.4005164979152
#20s , full

best = [0.00022855271187823933, 0.0003709767222688006, 0.0010280292401355596, 1.477396789027207, 7.26252655900307e-05, 0.00020006350350802764, 0.0002333333333333333, 0.4575489188387377, 4.37081686192501, 0.898274557929061, 0.12219296888380718, 0.27689360968045695, 2.860130691646921, 0.2786656396304239, 2.4911813327038415, 0.9309597927321951, 0.05120766665198147, 0.8781145421594301, 0.06999999999999999, 0.7647617439170884, 0.16309304038971356]

best_fitness_memory = np.ones(200)*50
best_fitness_session = 50

fitness , fitness_memory = fitness_calculator(best,best_fitness_memory=best_fitness_memory) # evaluate points in parallel

input (fitness)

#placeholder , best_fitness_memory =fitness_calculator(best)
 

""" initial_G_VAS = best[0]
initial_G_SOL = best[1]
initial_G_GAS = best[2]
initial_G_TA = best[3]
initial_G_SOL_TA = best[4]
initial_G_HAM = best[5]
initial_G_GLU = best[6]
initial_G_HFL =best[7]
initial_G_HAM_HFL = best[8]
initial_G_delta_theta = best[9]
initial_theta_ref = best[10]
initial_k_swing =   best[11]
initial_k_p = best[12]
initial_k_d = best[13]
initial_phi_k_off = best[14]

# Offset parameters with default values
initial_loff_TA =  best[15]
initial_lopt_TA =  best[16]
initial_loff_HAM = best[17]
initial_lopt_HAM = best[18]
initial_loff_HFL = best[19]
initial_lopt_HFL = best[20]


# Pre-stimulation parameters with default values
initial_So     = best[21]
initial_So_VAS = best[22]
initial_So_BAL = best[23]  """
        
        
        
# Define the parameter bounds
a = 0.7
b = 1.6

# Define the parameter space for Bayesian optimization
space = [
    Real(initial_G_VAS * a, initial_G_VAS * b, name='G_VAS', prior='uniform', transform='normalize'),
    Real(initial_G_SOL * a, initial_G_SOL * b, name='G_SOL', prior='uniform', transform='normalize'),
    Real(initial_G_GAS * a, initial_G_GAS * b, name='G_GAS', prior='uniform', transform='normalize'),
    Real(initial_G_TA * a, initial_G_TA * b, name='G_TA', prior='uniform', transform='normalize'),
    Real(initial_G_SOL_TA * a, initial_G_SOL_TA * b, name='G_SOL_TA', prior='uniform', transform='normalize'),
    Real(initial_G_HAM * a, initial_G_HAM * b, name='G_HAM', prior='uniform', transform='normalize'),
    Real(initial_G_GLU * a, initial_G_GLU * b, name='G_GLU', prior='uniform', transform='normalize'),
    Real(initial_G_HFL * a, initial_G_HFL * b, name='G_HFL', prior='uniform', transform='normalize'),
    Real(initial_G_HAM_HFL * a, initial_G_HAM_HFL * b, name='G_HAM_HFL', prior='uniform', transform='normalize'),
    Real(initial_G_delta_theta * a, initial_G_delta_theta * b, name='G_delta_theta', prior='uniform', transform='normalize'),
    
    
    Real(initial_theta_ref  * a, initial_theta_ref  * b, name='theta_ref', prior='uniform', transform='normalize'),
    Real(initial_k_swing  * a, initial_k_swing  * b, name='k_swing', prior='uniform', transform='normalize'),
    Real(initial_k_p * a, initial_k_p  * b, name='k_p', prior='uniform', transform='normalize'),
    Real(initial_k_d * a, initial_k_d  * b, name='k_d', prior='uniform', transform='normalize'),
    Real(initial_phi_k_off * a, initial_phi_k_off * b, name='phi_k_off', prior='uniform', transform='normalize'),
    
    Real(initial_loff_TA * a, initial_loff_TA  * b, name='loff_TA', prior='uniform', transform='normalize'),
    Real(initial_lopt_TA * a, initial_lopt_TA  * b, name='lopt_TA', prior='uniform', transform='normalize'),
    Real(initial_loff_HAM * a, initial_loff_HAM  * b, name='loff_HAM', prior='uniform', transform='normalize'),
    Real(initial_lopt_HAM * a, initial_lopt_HAM  * b, name='lopt_HAM', prior='uniform', transform='normalize'),
    Real(initial_loff_HFL * a, initial_loff_HFL  * b, name='loff_HFL', prior='uniform', transform='normalize'),
    Real(initial_lopt_HFL * a, initial_lopt_HFL  * b, name='lopt_HFL', prior='uniform', transform='normalize')
]


# Create the optimizer
optimizer = Optimizer(space,base_estimator="GP", acq_func="PI",random_state=42)
count=0
for i in range (len(memory_fitness)):
    try:
        optimizer.tell(memory_suggestion[i],memory_fitness[i])
        count+=1
        print(i, "accepted")
    except:
        print(i,"rejected") 
        
    #print(len(memory_fitness),i,memory_fitness[i])
    

print(count)


from joblib import Parallel, delayed
# example objective taken from skopt
parallel_jobs=1


best_fitness_memory = np.ones(200)*100
best_fitness_session = 100

# Run Bayesian optimization
while(True):
    
    suggestion = optimizer.ask(n_points=parallel_jobs) 

    #fitness = Parallel(n_jobs=parallel_jobs)(delayed(fitness_calculator)(suggestion[i],id=i) for i in range(parallel_jobs)) # evaluate points in parallel
    fitness , fitness_breakdown = fitness_calculator(suggestion[0],best_fitness_memory=best_fitness_memory) # evaluate points in parallel
        
    if(fitness < best_fitness_session ):
        best_fitness_memory  = fitness_breakdown
        best_fitness_session = fitness
        print("new fitness memory target",best_fitness_session)
        
        
    optimizer.tell(suggestion[0], fitness)
    print("\n",len(memory_fitness),"fitness", fitness)
    

    # Save lists
    memory_fitness.append(fitness)
    memory_fitness_breakdown.append(fitness_breakdown)
    
    for s in suggestion:
            memory_suggestion.append(s)

    np.save(str(name)+"memory_fitness.npy", np.array(memory_fitness))
    np.save(str(name)+"memory_suggestion.npy", np.array(memory_suggestion))
    np.save(str(name)+"memory_fitness_breakdown.npy", np.array(memory_fitness_breakdown))

    result = optimizer.get_result()
    print("\nBest results:", result.x)
    print("Best Fitness:", result.fun)
    
    
















































""" 




dt = 1000e-7
tf = 6
F_max_alpha=0.778
v_max_alpha=0.8

flag_graph=False

name="fitness_data/f"+str(F_max_alpha)+"v"+str(v_max_alpha)+"tf"+str(tf)


import numpy as np
from skopt import Optimizer
from skopt import forest_minimize
from skopt.space import Real


# Define initial parameter values
initial_G_VAS = 2e-4
initial_G_SOL = 1.2 / 4000
initial_G_GAS = 1.1 / 1500
initial_G_TA = 1.1
initial_G_SOL_TA = 0.0001
initial_G_HAM = 2.166666666666667e-04
initial_G_GLU = 1 / 3000.
initial_G_HFL = 0.5
initial_G_HAM_HFL = 4
initial_G_delta_theta = 1.145915590261647

initial_theta_ref = 0.104719755119660
initial_k_swing = 0.26
initial_k_p = 1.909859317102744
initial_k_d = 0.2
initial_phi_k_off = 2.967059728390360

# Offset parameters with default values
initial_loff_TA =  0.72
initial_lopt_TA =  0.06
initial_loff_HAM = 0.85
initial_lopt_HAM = 0.10
initial_loff_HFL = 0.65
initial_lopt_HFL = 0.11


# Pre-stimulation parameters with default values
initial_So     = 0.01
initial_So_VAS = 0.08
initial_So_BAL = 0.05





best= [0.00021451021347858762, 0.0002818105038724765, 0.0008056483781169555, 0.9900000000000001, 9.202038020883363e-05, 0.00019764000216095177, 0.00030421289890817657, 0.5486559450900268, 3.74259067255584, 1.2412782367429336]
best= [0.00022723251336198097, 0.000263968223597214, 0.0008507148982656886, 1.0091763312734046, 9.102348207337796e-05, 0.0001818278104605188, 0.0003017334963664198, 0.5304085932668562, 3.475270009334734, 1.278737152272291]
best= [0.00023728077499771596, 0.00026563883144114603, 0.0008409671094272204, 0.9603287910937125, 8.85739839694462e-05, 0.00017711893321276646, 0.00030726317059558166, 0.5362429105187098, 3.591064237736697, 1.2369690769917965, 0.13157894446987853,0.32]
#16306
best=[0.00027287289124737333, 0.0002304932432771713, 0.0009671121758413034, 1.1043781097577692, 7.528788637402927e-05, 0.00020368677319468142, 0.00033366245078424726, 0.49698348945516785, 3.052404602076192, 1.0514242560487264, 0.11184210279939674, 0.272]
#1201859.0062338933
best=[0.0002034821064863745, 0.0003019072795460582, 0.0007520905719879207, 1.1099710836831285, 0.0001024897580532627, 0.0002212171681779341, 0.0003323223468160974, 0.4878623034947123, 3.968996380527584, 1.157524259684039, 0.10576227603289168, 0.26142424548824056, 1.8840443361468564, 0.20073492111017244, 2.94621795107867]
#0.95 4700
best=[0.00020032244458211386, 0.00030898258567887537, 0.0007684125048874361, 1.1148397204033387, 0.00010518013764591801, 0.0002199339501314751, 0.0003368207691181484, 0.49621350795025837, 3.8502466462059344, 1.130077126127288, 0.10384893307696859, 0.25764592174238377, 1.8843262401748322, 0.19507013140282387, 2.9151996215211726]
#0.88 8562 
best=[0.0002007931282461503, 0.00030967256332341313, 0.0007608369765844455, 1.1042055413490053, 0.00010523225368944018, 0.0002194934706904153, 0.0003337669207184673, 0.5009151857085606, 3.8296687212003677, 1.1208242000489248, 0.10409481829217822, 0.25704025797038904, 1.9025375329386054, 0.19494046006719254, 2.936185584270931]
#O.88 7473
best=[0.00020288975746899607, 0.00031090633261031224, 0.0007382447765840968, 1.1531056780106368, 0.00010627263727450483, 0.0002237688727895238, 0.0003464656786245205, 0.5071442688200157, 3.751403193098936, 1.07660704198488, 0.10364236569342092, 0.2498030437052075, 1.8866532371582425, 0.20241214306802424, 2.884610022557958]
#0.86 12370

best =[0.00020838521563748896, 0.00030279121848843195, 0.0007536397392827817, 1.1006262668757263, 0.00011073504382599255, 0.00022909564683972286, 0.00032950641735506405, 0.5148517558703398, 3.6211593717123503, 1.1185470630914691, 0.10411986033826364, 0.2533722788993626, 1.8954501438339189, 0.1966672911189784, 2.787493979457837, 0.686204425257249, 0.05765531360027646, 0.8230895358812534, 0.10008248985783516, 0.6574382688521642, 0.10483168362812152, 0.010336493974787, 0.08357898418375251, 0.049787806661221456, 0.8800015482041182]
#0.88  7469

best= [0.0002017530049513596, 0.00032588222600278744, 0.0008243232594716532, 1.1996884020326075, 9.970861943851622e-05, 0.00023894389225832054, 0.00034837393693679195, 0.5289967835355902, 3.2772153806042565, 1.153331923671437, 0.09514159632970996, 0.25449905623587704, 1.7273536649111285, 0.20024740950138284, 2.881366827714655, 0.6232463238202345, 0.06325040813715892, 0.7849906458881816, 0.0956646967130869, 0.6909725005263179, 0.10340066217711921, 0.009797077736039635, 0.0852783437864845, 0.048845948693298114, 0.86]
#0.84 12878

best = [0.00019858554018624598, 0.00038408665352797935, 0.0008501144622447181, 1.3408911649658672, 0.00010945751145615829, 0.0002775146853079911, 0.0002875649232749087, 0.5931577026550473, 2.769071421563315, 1.0380215065059217, 0.11226140004898291, 0.21118343271515758, 1.7386757298254432, 0.16327561240200292, 3.3856727163265776, 0.7298904592938643, 0.06444104829238949, 0.931010333691888, 0.08211225900249854, 0.6951927881766085, 0.1157929729162852, 0.01049935385940784, 0.08939094669845936, 0.04307643515807261, 0.8400000000034299]
#0.84 6854

best =[0.00020925763393115352, 0.0004118492743927525, 0.0008606336369092829, 1.3128843276513, 0.00011177332981734889, 0.0002878045661705798, 0.00023935217085328234, 0.5622315172974713, 2.483480403063275, 1.1638877411456408, 0.11092776310592121, 0.25201981605440404, 1.668311135276158, 0.18394185555536607, 3.789712296210273, 0.6279152863905196, 0.06465238339954436, 1.0039318323284976, 0.09388251940117454, 0.646795942596604, 0.10283532459535288, 0.011386127322733945, 0.10045758330589645, 0.040468863243817106, 0.8200000000000475]
#0.82 2s 2843

best = [0.000233546212745655, 0.000429305089884046, 0.0010073891871063092, 1.3922229912711717, 0.00013032589968753095, 0.00032810164658008377, 0.00023451253450120597, 0.47124224366195006, 2.355152267423648, 1.2424926468155792, 0.11828992145345794, 0.2612233567201142, 1.5179774787602687, 0.18844794731176354, 3.6122429112897416, 0.7463933805785682, 0.0736756275354852, 1.0929723621142462, 0.08396795904362601, 0.5836863349531, 0.08393146019817826, 0.012345581116388917, 0.08482199619254269, 0.03948687011986283, 0.8200000000009995]
#0.82 5s 4633

best= [0.0002476595996011982, 0.000500562810207077, 0.001148439218440515, 1.3640726178079512, 0.00010923448345838676, 0.000311147748915347, 0.0002503506905374064, 0.502518093104393, 2.441160344115895, 1.130530040536522, 0.12118771346668405, 0.24899046281670248, 1.804396813664059, 0.2147487849004199, 3.9326192677103347, 0.6675697372696486, 0.06648689977694805, 0.8920550080380278, 0.09104353220088526, 0.4928392506743195, 0.08189482980652411, 0.010872601203177096, 0.0982497794746892, 0.039097849000547164, 0.7780000000040426, 0.8000000000053206]
#full 2s 3797

best= [0.00025161335148950447, 0.0004107460355699797, 0.001191269108005828, 1.2444504051975525, 9.036957414903379e-05, 0.00028112908494196695, 0.00020174277677774546, 0.40526260886199994, 2.134645896217124, 1.2079762828986662, 0.10995821187307848, 0.28557198599088873, 1.7261518685162862, 0.19528886045878097, 3.445798759155944, 0.620655267012408, 0.0782721648490077, 0.8239828068954215, 0.0816367970858522, 0.5907131470195318, 0.06990631274098111, 0.011620527440629969, 0.08164021470763207, 0.03603571942844179, 0.7780000000054695, 0.8000000000034387]
#full 5s 5373


best = [0.0002536243264431705, 0.00041783085685604966, 0.0011677387909938238, 1.2204685269091167, 9.216504665128866e-05, 0.0002759930293467014, 0.00020354146598049714, 0.4030891117315347, 2.1677897249933653, 1.1891156637022235, 0.11106146345218056, 0.28060954745986183, 1.7112765555695757, 0.19789940752036345, 3.452189126366633, 0.6239076465732398, 0.07730816865533373, 0.829929329755522, 0.0824103731606878, 0.5920955389470945, 0.06948672988116915, 0.011548402203892548, 0.08226737396382561, 0.03667399736274654, 0.7779999999993815, 0.800000000004609]
#full 10s Best Fitness: 1520282.2384506825


best= [0.00025161335148950447, 0.0004107460355699797, 0.001191269108005828, 1.2444504051975525, 9.036957414903379e-05, 0.00028112908494196695, 0.00020174277677774546, 0.40526260886199994, 2.134645896217124, 1.2079762828986662, 0.10995821187307848, 0.28557198599088873, 1.7261518685162862, 0.19528886045878097, 3.445798759155944, 0.620655267012408, 0.0782721648490077, 0.8239828068954215, 0.0816367970858522, 0.5907131470195318, 0.06990631274098111, 0.011620527440629969, 0.08164021470763207, 0.03603571942844179, 0.7780000000054695, 0.8000000000034387]
#full 5s 5373

best =  [0.00025721945374747535, 0.00041017065847990804, 0.0012088311525866582, 1.2444426578161107, 9.456329022102047e-05, 0.00026757164030952017, 0.00019171219591902837, 0.4104646378335758, 2.0313107499326053, 1.254688458861997, 0.11243654515919223, 0.28787406059681553, 1.7866557114069959, 0.20377397637827838, 3.385410153548072, 0.5896225036617876, 0.07479750020959994, 0.8129673605116439, 0.08198122675052542, 0.5852723779901078, 0.06717674548053261, 0.011584512287610909, 0.08182008764648747, 0.03643897388166518, 0.7780000000064132, 0.8000000000003628]
#full 6s 11000

best = [0.00026078184460466856, 0.00042358083026587494, 0.0011741674372193565, 1.2731132201021962, 8.597886193040449e-05, 0.0002670726306948686, 0.00020251447357547782, 0.39992404977493595, 2.044694627777905, 1.1645473823681178, 0.1076801172900888, 0.2712933866913443, 1.7481057276085277, 0.20410664859743333, 3.4859632569273233, 0.5896225036617876, 0.07435855660655731, 0.7867206765458591, 0.07755495723155958, 0.5885087766320153, 0.07327661021371795, 0.01159347363027351, 0.08218808954384062, 0.03783750539986388, 0.778000000005377, 0.7999999999992614]
#full 6s 8181


#print(fitness_calculator(best))

initial_G_VAS = best[0]
initial_G_SOL = best[1]
initial_G_GAS = best[2]
initial_G_TA = best[3]
initial_G_SOL_TA = best[4]
initial_G_HAM = best[5]
initial_G_GLU = best[6]
initial_G_HFL =best[7]
initial_G_HAM_HFL = best[8]
initial_G_delta_theta = best[9]
initial_theta_ref = best[10]
initial_k_swing =   best[11]
initial_k_p = best[12]
initial_k_d = best[13]
initial_phi_k_off = best[14]

# Offset parameters with default values
initial_loff_TA =  best[15]
initial_lopt_TA =  best[16]
initial_loff_HAM = best[17]
initial_lopt_HAM = best[18]
initial_loff_HFL = best[19]
initial_lopt_HFL = best[20]


# Pre-stimulation parameters with default values
initial_So     = best[21]
initial_So_VAS = best[22]
initial_So_BAL = best[23]
        
        
# Define the parameter bounds
a = 0.95
b = 1.05 # Adjust upper bound as needed

# Define the parameter space for Bayesian optimization
space = [
    Real(initial_G_VAS * a, initial_G_VAS * b, name='G_VAS', prior='uniform', transform='normalize'),
    Real(initial_G_SOL * a, initial_G_SOL * b, name='G_SOL', prior='uniform', transform='normalize'),
    Real(initial_G_GAS * a, initial_G_GAS * b, name='G_GAS', prior='uniform', transform='normalize'),
    Real(initial_G_TA * a, initial_G_TA * b, name='G_TA', prior='uniform', transform='normalize'),
    Real(initial_G_SOL_TA * a, initial_G_SOL_TA * b, name='G_SOL_TA', prior='uniform', transform='normalize'),
    Real(initial_G_HAM * a, initial_G_HAM * b, name='G_HAM', prior='uniform', transform='normalize'),
    Real(initial_G_GLU * a, initial_G_GLU * b, name='G_GLU', prior='uniform', transform='normalize'),
    Real(initial_G_HFL * a, initial_G_HFL * b, name='G_HFL', prior='uniform', transform='normalize'),
    Real(initial_G_HAM_HFL * a, initial_G_HAM_HFL * b, name='G_HAM_HFL', prior='uniform', transform='normalize'),
    Real(initial_G_delta_theta * a, initial_G_delta_theta * b, name='G_delta_theta', prior='uniform', transform='normalize'),
    
    
    Real(initial_theta_ref  * a, initial_theta_ref  * b, name='theta_ref', prior='uniform', transform='normalize'),
    Real(initial_k_swing  * a, initial_k_swing  * b, name='k_swing', prior='uniform', transform='normalize'),
    Real(initial_k_p * a, initial_k_p  * b, name='k_p', prior='uniform', transform='normalize'),
    Real(initial_k_d * a, initial_k_d  * b, name='k_d', prior='uniform', transform='normalize'),
    Real(initial_phi_k_off * a, initial_phi_k_off * b, name='phi_k_off', prior='uniform', transform='normalize'),
    
    Real(initial_loff_TA * a, initial_loff_TA  * b, name='loff_TA', prior='uniform', transform='normalize'),
    Real(initial_lopt_TA * a, initial_lopt_TA  * b, name='lopt_TA', prior='uniform', transform='normalize'),
    Real(initial_loff_HAM * a, initial_loff_HAM  * b, name='loff_HAM', prior='uniform', transform='normalize'),
    Real(initial_lopt_HAM * a, initial_lopt_HAM  * b, name='lopt_HAM', prior='uniform', transform='normalize'),
    Real(initial_loff_HFL * a, initial_loff_HFL  * b, name='loff_HFL', prior='uniform', transform='normalize'),
    Real(initial_lopt_HFL * a, initial_lopt_HFL  * b, name='lopt_HFL', prior='uniform', transform='normalize'),
    
    Real(initial_So * a, initial_So  * b, name='So', prior='uniform', transform='normalize'),
    Real(initial_So_VAS * a, initial_So_VAS  * b, name='_So_VAS', prior='uniform', transform='normalize'),
    Real(initial_So_BAL * a, initial_So_BAL * b, name='So_BAL', prior='uniform', transform='normalize'),
    
#    Real(F_max_alpha *0.999999999999, F_max_alpha *1.00000000001, name='F_max_alpha', prior='uniform', transform='normalize'),
#    Real(v_max_alpha *0.999999999999, v_max_alpha *1.00000000001, name='v_max_alpha', prior='uniform', transform='normalize')
]




# Initialize empty lists or load existing ones from files
memory_fitness = np.load(str(name)+"memory_fitness.npy", allow_pickle=True).tolist() if str(name)[13:]+"memory_fitness.npy" in os.listdir("fitness_data") else []
memory_suggestion = np.load(str(name)+"memory_suggestion.npy", allow_pickle=True).tolist() if str(name)[13:]+"memory_suggestion.npy" in os.listdir("fitness_data") else []




# Create the optimizer
optimizer = Optimizer(space,base_estimator="GP", acq_func="PI",random_state=42)
#optimizer = forest_minimize(fitness_calculator , dimensions=space , n_calls=12, random_state=42)
#print(optimizer)
#input()

import matplotlib.pyplot as plt
plt.plot(memory_fitness)
plt.show()

for i in range (len(memory_fitness)):
    optimizer.tell(memory_suggestion[i],memory_fitness[i])
    print(i,memory_fitness[i])
    #if(memory_fitness[i]<6000 and memory_fitness[i]> 5000):
    #print(memory_fitness[i])   
    #    print(memory_suggestion[i])
    

from joblib import Parallel, delayed
# example objective taken from skopt
parallel_jobs=1


# Run Bayesian optimization
while(True):
    
    suggestion = optimizer.ask(n_points=parallel_jobs)  # x is a list of n_points points

    fitness = Parallel(n_jobs=parallel_jobs)(delayed(fitness_calculator)(suggestion[i],id=i) for i in range(parallel_jobs)) # evaluate points in parallel
    optimizer.tell(suggestion, fitness)
    
    print("\n",len(memory_fitness),"fitness", fitness)
    

    # Save lists
    for f in fitness:
            memory_fitness.append(f)

    for s in suggestion:
            memory_suggestion.append(s)

    np.save(str(name)+"memory_fitness.npy", np.array(memory_fitness))
    np.save(str(name)+"memory_suggestion.npy", np.array(memory_suggestion))

    result = optimizer.get_result()
    print("\nBest results:", result.x)
    print("Best Fitness:", result.fun)
    
    


# Get the best result from Bayesian optimization
result = optimizer.get_result()

print("Best results:",result.x)
print("Best Fitness:", result.fun)




 """



"""
    
    
    
# Initialize empty lists or load existing ones from files
memory_fitness = np.load(str(name)+"memory_fitness.npy", allow_pickle=True).tolist() if str(name)[13:]+"memory_fitness.npy" in os.listdir("fitness_data") else []
memory_suggestion = np.load(str(name)+"memory_suggestion.npy", allow_pickle=True).tolist() if str(name)[13:]+"memory_suggestion.npy" in os.listdir("fitness_data") else []
print(memory_fitness)

# Create the optimizer
optimizer = Optimizer(space,base_estimator="GBRT", acq_func="PI",random_state=42)

for i in range (len(memory_fitness)):
    optimizer.tell(memory_suggestion[i],memory_fitness[i])
    #if(memory_fitness[i]<6000 and memory_fitness[i]> 5000):
    #    print(memory_fitness[i])   
    #    print(memory_suggestion[i])

import matplotlib.pyplot as plt
from joblib import Parallel, delayed
# example objective taken from skopt
parallel_jobs=4

# Run Bayesian optimization
while(True):
    
    suggestion = optimizer.ask(n_points=parallel_jobs)  # x is a list of n_points points

    fitness = Parallel(n_jobs=parallel_jobs)(delayed(fitness_calculator)(v) for v in suggestion)  # evaluate points in parallel
    optimizer.tell(suggestion, fitness)
    
    print(fitness)
    
    
    suggestion = optimizer.ask()
    fitness = fitness_calculator(suggestion)
    optimizer.tell(suggestion, fitness)

    print(suggestion[0])
    print("\n",len(memory_fitness),"fitness", fitness)
    
    # Append values to lists
    memory_fitness.append(fitness)
    memory_suggestion.append(suggestion)

    # Save lists
    np.save(str(name)+"memory_fitness.npy", np.array(memory_fitness))
    np.save(str(name)+"memory_suggestion.npy", np.array(memory_suggestion))

    result = optimizer.get_result()
    print("\nBest results:", result.x)
    print("Best Fitness:", result.fun)
    
    


# Get the best result from Bayesian optimization
result = optimizer.get_result()

print("Best results:",result.x)
print("Best Fitness:", result.fun)
    """