
    
import os
import sys 
import matplotlib.pyplot as plt
import numpy as np
import time


# Get the directory where your script is located
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0,  os.path.join(parent_dir, "User_function"))
sys.path.insert(1,  os.path.join(parent_dir, "userfctR"))
sys.path.insert(2,  os.path.join(parent_dir, "workR"))
import TestworkR


flag_initiated=False
t=np.zeros((0))
dt=0


GRF_data = 0 
muscle_data = 0
fm_data = 0
act_data = 0
stim_data =0
stance_data =0


def initiate(tf, dt):
    
    global GRF_data, muscle_data ,fm_data , act_data, stim_data , stance_data

    t = np.arange(0, tf + dt, dt)  # Create the time array directly
    l = len(t)  # Calculate the length based on t

    
    GRF_data = np.zeros((16,5,l))
    muscle_data = np.zeros((18,l))
    fm_data = np.zeros((18,l))
    act_data = np.zeros((14,l))
    stim_data = np.zeros((14,l))
    stance_data = np.zeros((2,l))
    
    
def collect_ext(ixF,GRF,tsim , dt , tf):
    global flag_initiated, GRF_data 

    if (flag_initiated==False):
       initiate(tf , dt)
       flag_initiated=True
    ti= int(tsim/dt)
    
    GRF_data[:,ixF,ti]=GRF
       

    

        
        

    
def collect_muscle(torque,fm,act,stim,stance,tsim , dt , tf ):
    
    global muscle_data , flag_initiated , fm_data , act_data , stim_data, stance_data
    
    if (flag_initiated==False):
       initiate(tf, dt)
       flag_initiated=True
       
    ti= int(tsim/dt)
    muscle_data[:,ti] = torque
    fm_data[:,ti]     = fm
    act_data[:,ti]    = act
    stim_data[:,ti]   = stim
    stance_data[:,ti]   = stance

       
    
def show_ext(tsim , dt , flag_comparator=False):
    global GRF_data, muscle_data, fm_data, act_data, stim_data , stance_data

    np.save("numpy_archive/GRF",GRF_data)
    np.save("numpy_archive/muscle",muscle_data)
    np.save("numpy_archive/fm",fm_data)
    np.save("numpy_archive/act",act_data)    
    np.save("numpy_archive/stim",stim_data)    
    np.save("numpy_archive/stance",stance_data)    

        
    name_muscle=["Torque_ankle_TA_L", "Torque_ankle_GAS_L", "Torque_knee_GAS_L", "Torque_ankle_SOL_L",
    "Torque_knee_VAS_L", "Torque_knee_HAM_L", "Torque_hip_HAM_L", "Torque_hip_GLU_L", "Torque_hip_HFL_L",
    "Torque_ankle_TA_R", "Torque_ankle_GAS_R", "Torque_knee_GAS_R", "Torque_ankle_SOL_R",
    "Torque_knee_VAS_R", "Torque_knee_HAM_R", "Torque_hip_HAM_R", "Torque_hip_GLU_R", "Torque_hip_HFL_R"]

    #print(GRF_data)
    
    """ 
    for i in range(len(name_muscle)):
        gait_plot("muscle/",name_muscle[i],muscle_data[i],stance_data[int(i*2/len(name_muscle))], dt , tsim)


    Fm_names = ["Fm_TA_L", "Fm_GAS_L", "Fm_GAS_L", "Fm_SOL_L", "Fm_VAS_L", "Fm_HAM_L", "Fm_HAM_L", "Fm_GLU_L", "Fm_HFL_L", "Fm_TA_R"," Fm_GAS_R", "Fm_GAS_R", "Fm_SOL_R", "Fm_VAS_R", "Fm_HAM_R", "Fm_HAM_R", "Fm_GLU_R", "Fm_HFL_R"]
    for i in range(len(Fm_names)):
        gait_plot("fm/",Fm_names[i],fm_data[i],stance_data[int(i*2/len(Fm_names))], dt , tsim)

  
    stim_names=["VASL","SOLL","GASL","TAL","HAML","GLUL","HFLL","VASR","SOLR","GASR","TAR","HAMR","GLUR","HFLR"] 
    for i in range(len(stim_names)):
        gait_plot("stim/",stim_names[i],stim_data[i],stance_data[int(i*2/len(stim_names))], dt , tsim)

    act_names = ["act_memory_TA_L","act_memory_GAS_L","act_memory_SOL_L","act_memory_VAS_L","act_memory_HAM_L","act_memory_GLU_L",    "act_memory_HFL_L",   "act_memory_TA_R",    "act_memory_GAS_R",    "act_memory_SOL_R",    "act_memory_VAS_R",    "act_memory_HAM_R",    "act_memory_GLU_R",    "act_memory_HFL_R",]
    for i in range(len(act_names)):
        gait_plot("act/",act_names[i],act_data[i],stance_data[int(i*2/len(act_names))], dt , tsim)
        
    point_name = ["temp ", "BallR", "HeelL", "HeelR", "BallL"]
    GRF_names = ["Px", "Pz", "Vx", "Vz", "Sticking", "Sliding", "Stiction", "PosFP", "dx", "dvx", "Fx", "Fz", "Fx_sliding", "Fx_sticking", "test_slide", "test_stick"]
    for j in range(1,5):
        for i in range(len(GRF_names)):
            if(j==2 or j==4):
                gait_plot("GRF/"+point_name[j]+"/",point_name[j]+" "+GRF_names[i],GRF_data[i][j],stance_data[0], dt , tsim)
            else:
                gait_plot("GRF/"+point_name[j]+"/",point_name[j]+" "+GRF_names[i],GRF_data[i][j],stance_data[0], dt , tsim)
     """
    
    
        
        
        



        
def gait_plot(dir_name,name,data,stance , dt , tsim):

    stance_change_indices = []
    previous_value = 0
    for i, value in enumerate(stance):
        if value == 1 and previous_value == 0:
            stance_change_indices.append(i)
        previous_value = value
    
    id = "plot/"+dir_name+ name
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    for i in range(2,len(stance_change_indices)):
        intensity=0.01
        t = np.arange(0, len(data[stance_change_indices[i-1]:stance_change_indices[i]]), 1)*dt
        ax1.grid()
        ax1.plot(t,data[stance_change_indices[i-1]:stance_change_indices[i]], color=(intensity,intensity,intensity),alpha=0.050)
        ax1.set_title(name + " gait cycle overlayed")

    t = np.arange(0, tsim + dt, dt)   
    ax2.plot(t, data.T, label="Full gait",color='b')
    ax2.set_title(name)
    ax2.legend()
    ax2.grid()
    plt.tight_layout()
    fig.savefig(id)
    plt.close()  
    


def replay_show_ext():
    dt = np.load("parameters.npy", allow_pickle=True)[()].get("dt", 0)
    tsim = np.load("parameters.npy", allow_pickle=True)[()].get("tf", 0)

    t = np.arange(0, tsim + dt, dt)  # Create the time array directly
    
    stance_data = np.load(os.getcwd()+"/numpy_archive/stance.npy")
    GRF_data    = np.load(os.getcwd()+"/numpy_archive/GRF.npy")
    muscle_data = np.load(os.getcwd()+"/numpy_archive/muscle.npy")
    fm_data     = np.load(os.getcwd()+"/numpy_archive/fm.npy")
    act_data    = np.load(os.getcwd()+"/numpy_archive/act.npy")
    stim_data   = np.load(os.getcwd()+"/numpy_archive/stim.npy")

    
    name_muscle=["Torque_ankle_TA_L", "Torque_ankle_GAS_L", "Torque_knee_GAS_L", "Torque_ankle_SOL_L",
    "Torque_knee_VAS_L", "Torque_knee_HAM_L", "Torque_hip_HAM_L", "Torque_hip_GLU_L", "Torque_hip_HFL_L",
    "Torque_ankle_TA_R", "Torque_ankle_GAS_R", "Torque_knee_GAS_R", "Torque_ankle_SOL_R",
    "Torque_knee_VAS_R", "Torque_knee_HAM_R", "Torque_hip_HAM_R", "Torque_hip_GLU_R", "Torque_hip_HFL_R"]

    for i in range(len(name_muscle)):
        gait_plot("muscle/",name_muscle[i],muscle_data[i],stance_data[int(i*2/len(name_muscle))])

 
    Fm_names = ["Fm_TA_L", "Fm_GAS_L", "Fm_GAS_L", "Fm_SOL_L", "Fm_VAS_L", "Fm_HAM_L", "Fm_HAM_L", "Fm_GLU_L", "Fm_HFL_L", "Fm_TA_R"," Fm_GAS_R", "Fm_GAS_R", "Fm_SOL_R", "Fm_VAS_R", "Fm_HAM_R", "Fm_HAM_R", "Fm_GLU_R", "Fm_HFL_R"]
    for i in range(len(Fm_names)):
        gait_plot("fm/",Fm_names[i],fm_data[i],stance_data[int(i*2/len(Fm_names))])

  
    stim_names=["VASL","SOLL","GASL","TAL","HAML","GLUL","HFLL","VASR","SOLR","GASR","TAR","HAMR","GLUR","HFLR"] 
    for i in range(len(stim_names)):
        gait_plot("stim/",stim_names[i],stim_data[i],stance_data[int(i*2/len(stim_names))])

    act_names = ["act_memory_TA_L","act_memory_GAS_L","act_memory_SOL_L","act_memory_VAS_L","act_memory_HAM_L","act_memory_GLU_L",    "act_memory_HFL_L",   "act_memory_TA_R",    "act_memory_GAS_R",    "act_memory_SOL_R",    "act_memory_VAS_R",    "act_memory_HAM_R",    "act_memory_GLU_R",    "act_memory_HFL_R",]
    for i in range(len(act_names)):
        gait_plot("act/",act_names[i],act_data[i],stance_data[int(i*2/len(act_names))])
        
    point_name = ["temp ", "BallR", "HeelL", "HeelR", "BallL"]
    
    GRF_names = ["Px", "Pz", "Vx", "Vz", "Sticking", "Sliding", "Stiction", "PosFP", "dx", "dvx", "Fx", "Fz", "Fx_sliding", "Fx_sticking", "test_slide", "test_stick"]
    for j in range(1,5):
        for i in range(len(GRF_names)):
            if(j==2 or j==4):
                gait_plot("GRF/"+point_name[j]+"/",point_name[j]+" "+GRF_names[i],GRF_data[i][j],stance_data[0])
            else:
                gait_plot("GRF/"+point_name[j]+"/",point_name[j]+" "+GRF_names[i],GRF_data[i][j],stance_data[0])

     
if __name__ == "__main__":
    #replay_show_ext()
    TestworkR.runtest(1000e-7,5,c=False)   
    #60 , 154 print(154/60) print(1.3*300/60) print(cos())




















""" 


BallL=np.zeros((0))
BallR=np.zeros((0))
HeelL=np.zeros((0))
HeelR=np.zeros((0))

BallR_stiction=np.zeros(0)
Px=np.zeros(0)

#HeelL_Px=np.zeros((0,2))



def initiate_temp():
    global dt, t, BallL, BallR, HeelL, HeelR, BallR_stiction, Px
    dt = np.load("parameters.npy", allow_pickle=True)[()].get("dt", 0)
    tsim = np.load("parameters.npy", allow_pickle=True)[()].get("tf", 0)
    t = np.arange(0, tsim + dt, dt)  # Create the time array directly
    l = len(t)  # Calculate the length based on t
    BallL=(np.zeros((3,l)))
    BallR=(np.zeros((3,l)))
    HeelL=np.zeros((3,l))
    HeelR=np.zeros((3,l))
    BallR_stiction=np.zeros((11,l))
    Px=np.zeros((4,l))

    
    


flag_initiated=False

def collect_ext(mbs_data,type,fx,px,vx,tsim):
    global flag_initiated
    if (flag_initiated==False):
       initiate()
       flag_initiated=True
    
    global t, dt,  BallL , BallR , HeelL , HeelR
    ti= int(tsim/dt)
    if(type==mbs_data.extforce_id["Force_BallL"]):
        BallL[:,ti]=[fx,px,vx]
        
    if(type==mbs_data.extforce_id["Force_BallR"]):
        BallR[:,ti]=[fx,px,vx]
        
    if(type==mbs_data.extforce_id["Force_HeelL"]):
        HeelL[:,ti]=[fx,px,vx]
        
    if(type==mbs_data.extforce_id["Force_HeelR"]):
        HeelR[:,ti]=[fx,px,vx]    

def collect_stiction(mbs_data,Stiction_test_ballR,Force_slide_ballR,delta_x,delta_vx,Fx_mod, Force_stick_ballR,d_z, Fz,  slide_test, stick_test,delta_test, tsim):
    global flag_initiated
    if (flag_initiated==False):
       initiate()
       flag_initiated=True
       
    global t, dt, BallR_stiction
    ti= int(tsim/dt)
    
    BallR_stiction[:,ti]=[Stiction_test_ballR, Force_slide_ballR,delta_x, delta_vx,Fx_mod, Force_stick_ballR, d_z, Fz, slide_test, stick_test,delta_test]

    
      
def collect_Px(PxF,tsim):
    global flag_initiated
    if (flag_initiated==False):
       initiate()
       flag_initiated=True
       
    global t, dt, Px
    ti= int(tsim/dt)
    
    Px[:,ti]=PxF


    
 initiate()
    BallR_stiction = np.load(os.getcwd()+"/numpy_archive/stiction_BallR.npy")
    print(BallR_stiction)
    
    id="plot_archive/Force_stick_ballR"
    print(t.shape,BallR_stiction[5].shape)
    
    plt.plot(t[10:], BallR_stiction[5,10:], label="Force_stick_ballR")
    plt.legend()
    plt.title(id)
    plt.savefig(id)
plt.close()  



def show_ext(flag_comparator=False):
    global t, dt, GRF_data, muscle_data, fm_data, act_data, stim_data , stance_data

    name = ["temp", "BallR", "HeelL", "HeelR", "BallL"]
    
    variable_names = ["Px", "Pz", "Vx", "Vz", "Sticking", "Sliding", "Stiction", "PosFP", "dx", "dvx", "Fx", "Fz", "Fx_sliding", "Fx_sticking", "test_slide", "test_stick"]

    np.save("numpy_archive/GRF",GRF_data)
    np.save("numpy_archive/muscle",muscle_data)
    np.save("numpy_archive/fm",fm_data)
    np.save("numpy_archive/act",act_data)    
    np.save("numpy_archive/stim",stim_data)    
    np.save("numpy_archive/stance",stance_data)    


    for i in range(1, 5):
        for j in range(len(GRF_data)):
            id = "plot/GRF/"+ name[i] + "/" + name[i] + " " + variable_names[j]           

            if(flag_comparator):
                ratio=int(dt/(250.0*(10.0**-7.0)))
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
                comparator=np.load(os.getcwd()+"/numpy_archive_comparator/GRF.npy")[j][i][::ratio][:len(GRF_data[j][i])]
                ax1.plot(t, GRF_data[j][i], label="-6",linestyle="dashed",color="black")
                ax1.plot(t, comparator, label="-7",color="green")

                ax1.grid()
                ax1.legend()
                ax1.set_title(id)
                

                ax2.plot(t, -GRF_data[j][i]+comparator, label="Diff",color='r')
                ax2.legend()
                ax2.grid()
                plt.tight_layout()
                fig.savefig(id)
                plt.close()    
                
            else:
                ratio=int(dt/(250.0*(10.0**-7.0)))
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
                ax1.plot(t, GRF_data[j][i], label="-6",linestyle="dashed",color="black")

                ax1.grid()
                ax1.legend()
                ax1.set_title(id)
                

                ax2.plot(t, -GRF_data[j][i], label="Diff",color='r')
                ax2.legend()
                ax2.grid()
                plt.tight_layout()
                fig.savefig(id)
                plt.close()  

            

    name_muscle=["Torque_ankle_TA_L", "Torque_ankle_GAS_L", "Torque_knee_GAS_L", "Torque_ankle_SOL_L",
    "Torque_knee_VAS_L", "Torque_knee_HAM_L", "Torque_hip_HAM_L", "Torque_hip_GLU_L", "Torque_hip_HFL_L",
    "Torque_ankle_TA_R", "Torque_ankle_GAS_R", "Torque_knee_GAS_R", "Torque_ankle_SOL_R",
    "Torque_knee_VAS_R", "Torque_knee_HAM_R", "Torque_hip_HAM_R", "Torque_hip_GLU_R", "Torque_hip_HFL_R"]

    ratio=int(dt/(250.0*(10.0**-7.0)))
    

    comparator=np.load(os.getcwd()+"/numpy_archive_comparator/muscle.npy")
    for i in range(len(name_muscle)):
        id = "plot/muscle/" + name_muscle[i]
        if(flag_comparator):
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

            ax1.plot(t, comparator[i][::ratio][:len(muscle_data[i])], label="-7",color="green")
            ax1.plot(t, muscle_data[i], label="-6",linestyle="dashed",color="black")
            ax1.grid()
            ax1.legend()
            ax1.set_title(id)
            

            ax2.plot(t, -muscle_data[i]+comparator[i][::ratio][:len(muscle_data[i])], label="Diff",color='r')
            ax2.legend()
            ax2.grid()
            plt.tight_layout()
            fig.savefig(id)
            plt.close()  
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

            ax1.plot(t, muscle_data[i], label="-6",linestyle="dashed",color="black")
            ax1.grid()
            ax1.legend()
            ax1.set_title(id)
            

            ax2.plot(t, -muscle_data[i], label="Diff",color='r')
            ax2.legend()
            ax2.grid()
            plt.tight_layout()
            fig.savefig(id)
            plt.close()  
    
    comparator=np.load(os.getcwd()+"/numpy_archive_comparator/fm.npy")
    Fm_names = ["Fm_TA_L", "Fm_GAS_L", "Fm_GAS_L", "Fm_SOL_L", "Fm_VAS_L", "Fm_HAM_L", "Fm_HAM_L", "Fm_GLU_L", "Fm_HFL_L", "Fm_TA_R"," Fm_GAS_R", "Fm_GAS_R", "Fm_SOL_R", "Fm_VAS_R", "Fm_HAM_R", "Fm_HAM_R", "Fm_GLU_R", "Fm_HFL_R"]
    for i in range(len(Fm_names)):
        id = "plot/Fm/" + Fm_names[i]
        if(flag_comparator):
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

            ax1.plot(t, comparator[i][::ratio][:len(fm_data[i])], label="-7",color="green")
            ax1.plot(t, fm_data[i], label="-6",linestyle="dashed",color="black")
            ax1.grid()
            ax1.legend()
            ax1.set_title(id)
            

            ax2.plot(t, -fm_data[i]+comparator[i][::ratio][:len(fm_data[i])], label="Diff",color='r')
            ax2.legend()
            ax2.grid()
            plt.tight_layout()
            fig.savefig(id)
            plt.close()  
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

            ax1.plot(t, fm_data[i], label="-6",linestyle="dashed",color="black")
            ax1.grid()
            ax1.legend()
            ax1.set_title(id)
            

            ax2.plot(t, -fm_data[i], label="Diff",color='r')
            ax2.legend()
            ax2.grid()
            plt.tight_layout()
            fig.savefig(id)
            plt.close()      
        
        
        
    act_names = ["act_memory_TA_L","act_memory_GAS_L","act_memory_SOL_L","act_memory_VAS_L","act_memory_HAM_L","act_memory_GLU_L",    "act_memory_HFL_L",   "act_memory_TA_R",    "act_memory_GAS_R",    "act_memory_SOL_R",    "act_memory_VAS_R",    "act_memory_HAM_R",    "act_memory_GLU_R",    "act_memory_HFL_R",]
    comparator=np.load(os.getcwd()+"/numpy_archive_comparator/act.npy")
    for i in range(len(act_names)):
        id = "plot/act/" + act_names[i]
        if(flag_comparator):
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

            ax1.plot(t, comparator[i][::ratio][:len(fm_data[i])], label="-7",color="green")
            ax1.plot(t, act_data[i], label="-6",linestyle="dashed",color="black")
            ax1.grid()
            ax1.legend()
            ax1.set_title(id)
            

            ax2.plot(t, -act_data[i]+comparator[i][::ratio][:len(act_data[i])], label="Diff",color='r')
            ax2.legend()
            ax2.grid()
            plt.tight_layout()
            fig.savefig(id)
            plt.close()  
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

            ax1.plot(t, act_data[i], label="-6",linestyle="dashed",color="black")
            ax1.grid()
            ax1.legend()
            ax1.set_title(id)
            

            ax2.plot(t, -act_data[i], label="Diff",color='r')
            ax2.legend()
            ax2.grid()
            plt.tight_layout()
            fig.savefig(id)
            plt.close()  
            
        
    dir_name="stim"
    stim_names=["VASL","SOLL","GASL","TAL","HAML","GLUL","HFLL","VASR","SOLR","GASR","TAR","HAMR","GLUR","HFLR"]   
    comparator=np.load(os.getcwd()+"/numpy_archive_comparator/stim.npy")
    for i in range(len(stim_names)):
        id = "plot/stim/" + stim_names[i]
        if(flag_comparator):
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

            ax1.plot(t, comparator[i][::ratio][:len(stim_data[i])], label="-7",color="green")
            ax1.plot(t, stim_data[i], label="-6",linestyle="dashed",color="black")
            ax1.grid()
            ax1.legend()
            ax1.set_title(id)
            

            ax2.plot(t, -stim_data[i]+comparator[i][::ratio][:len(stim_data[i])], label="Diff",color='r')
            ax2.legend()
            ax2.grid()
            plt.tight_layout()
            fig.savefig(id)
            plt.close()  
            
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

            ax1.plot(t, stim_data[i], label="-6",linestyle="dashed",color="black")
            ax1.grid()
            ax1.legend()
            ax1.set_title(id)
            

            ax2.plot(t, -stim_data[i], label="Diff",color='r')
            ax2.legend()
            ax2.grid()
            plt.tight_layout()
            fig.savefig(id)
            plt.close()  
        
    stance_names=["StanceL","StanceR"]
    comparator=np.load(os.getcwd()+"/numpy_archive_comparator/stance.npy")
    
    for i in range(len(stance_names)):
        id = "plot/stance/" + stance_names[i]
        if(flag_comparator):
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

            ax1.plot(t, comparator[i][::ratio][:len(stance_data[i])], label="-7",color="green")
            ax1.plot(t, stance_data[i], label="-6",linestyle="dashed",color="black")
            ax1.grid()
            ax1.legend()
            ax1.set_title(id)
            

            ax2.plot(t, -stim_data[i]+comparator[i][::ratio][:len(stance_data[i])], label="Diff",color='r')
            ax2.legend()
            ax2.grid()
            plt.tight_layout()
            fig.savefig(id)
            plt.close()  
            
        else:
            print("WESH")
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

            ax1.plot(t, stance_data[i], label="-6",linestyle="dashed",color="black")
            ax1.grid()
            ax1.legend()
            ax1.set_title(id)
            

            ax2.plot(t, -stance_data[i], label="Diff",color='r')
            ax2.legend()
            ax2.grid()
            plt.tight_layout()
            fig.savefig(id)
            plt.close()  
"""

     