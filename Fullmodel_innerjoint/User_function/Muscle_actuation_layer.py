#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:29:35 2022

@author: matthieuxaussems
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import useful_functions as u_f
from scipy.interpolate import interp1d

################################ Informations générales ####################################################################################
#ici nous avons les informations qui sont communes à tous les muscles

w_muscle = 0.56 #width (portion of l_opt)
epsilon_ref = 0.04 #reference strain
epsilon_be = w_muscle/2 #BE reference compression
epsilon_pe = w_muscle  #PE reference strain
c = np.log(0.05) # residual force factor
fl_inf = 0.001 #lower bound for fl
fv_inf = 0 #lower bound for fv
fv_sup = 1.5 #upper bound for fv
K_muscle = 5 #shape factor
N_muscle = 1.5 #force enhancement

Left = 0 #indice pour la jambe gauche
Right = 1 # indice pour la jambe droite

################################ Informations sur les muscles #############################################################################
#indexs permettant de determiner les caractéristiques de quel muscle on est entrain de modifier
VAS =0
SOL =1
GAS = 2
TA = 3
HAM = 4
GLU = 5
HFL = 6

n_muscle = 7





    

#Données sur les longueurs des muscles sous la forme de vecteur avec l'informations placés à l'indice correspondant à son muscle
l_opt_muscle=np.array([0.08,0.04,0.05,0.06,0.1,0.11,0.11]) #[m]
l_slack_muscle=np.array([0.23,0.26,0.4,0.24,0.31,0.13,0.10]) #[m]
F_max_muscle = np.array([6000,4000,1500,800,3000,1500,2000]) 
v_max_muscle = np.array([12,6,12,12,12,12,12])
l_min_muscle = np.zeros(n_muscle)

def set_parameters(parameters):
    global F_max_muscle , v_max_muscle
    F_max_alpha = parameters.get("F_max_alpha", 1)
    v_max_alpha = parameters.get("v_max_alpha", 1)
    F_max_muscle = np.array([6000,4000,1500,800,3000,1500,2000]) * F_max_alpha
    v_max_muscle = np.array([12,6,12,12,12,12,12]) * v_max_alpha
    

for i in range (n_muscle):
    l_min_muscle[i] = (1-w_muscle)*l_opt_muscle[i]


#initial length, comme c'est des données qui change durant la simulation, il faut differencier les muscles de la jambe droite des muscles 
#de la jambe gauche

l_mtu = np.zeros((2,n_muscle)) #longueur de l'unité motrice (i,j) i : jambe , j : muscle
l_ce = np.zeros((2,n_muscle)) #longueur de l'élément contractile (i,j) i : jambe , j : muscle
l_se = np.zeros((2,n_muscle)) #longueur de l'élément élastique en série (i,j) i : jambe , j : muscle

for i in range (2):
    l_mtu[i]= l_opt_muscle+ l_slack_muscle
    l_ce[i] = l_opt_muscle
    l_se[i] = l_mtu[i] - l_ce[i]


#initial velocities and forces of muscles, c'est des données qui change pendant la simulation, il faut differencier les muscles de la jambe droite 
#des muscles de la jambes gauche 

v_ce=np.zeros((2,n_muscle)) # vitesse de contraction de l'élément contractile (i,j) i : jambe , j : muscle
F_ce=np.zeros((2,n_muscle)) # Force de l'élément contractile (i,j) i : jambe , j : muscle
F_se=np.zeros((2,n_muscle)) # force de l'élément élastique en série (i,j) i : jambe , j : muscle
F_m=np.zeros((2,n_muscle)) # Force de l'unité motrice (i,j) i : jambe , j : muscle



torque_L = np.zeros((3,n_muscle)) # Torque correspondant à la jambe gauche (i,j) i : articulation , j : muscle
torque_R = np.zeros((3,n_muscle)) # Torque correspondant à la jambe droite (i,j) i : articulation , j : muscle


################################ Informations sur les articulations #############################################################################
#indexs permettant de déterminer quelle articulation est impliquée par le muscle étudié
ankle=0
knee=1
hip=2

#données sur les angles maximaux et de références, sous la forme d'un tableau à 2 entrées cela dépend de l'articulation et du muscle (articulation,muscle)

r_0 = np.array([[0,0.05,0.05,0.04,0,0,0],[0.06,0,0.05,0,0.05,0,0],[0,0,0,0,0.08,0.1,0.1]]) #[m]
phi_max = np.array([[0,110,110,80,0,0,0],[165,0,140,0,180,0,0],[0,0,0,0,0,0,0]])*np.pi/180 #[rad]
phi_ref = np.array([[0,80,80,110,0,0,0],[125,0,165,0,180,0,0],[0,0,0,0,155,150,180]])*np.pi/180 #[rad]
rho = np.array([[0,0.5,0.7,0.7,0,0,0],[0.7,0,0.7,0,0.7,0,0],[0,0,0,0,0.7,0.5,0.5]])



################################ useful function ##########################################################################################
# fonctions utiles permetant de simplifier certaines opérations
    
def pos(x) : #fonction permettant de soit return la valeurs si celle-ci est positive, soit de return 0
    if x>0 : 
        return x 
    else : 
        return 0
    


################################ Caclul de la longueur de l'élément contractile et longueur totale par muscle ################################################

# 1) première fonction utile pour obtenir la longueur de l'élément contactile à la prochaine itérations. La fonction réalise aussi l'update de la force de l'unité motrice
# parametre in : longeur actuelle de l'element contractile[m], index du muscle dont on veut updater la longueur de l'élément contractile
# parametre out : v_ce qui est la vitesse de contraction de l'élément contractile, soit la dérivé de la longueur de l'élément contractile
#

def vce_compute(l_ce_current, l_mtu_current, Act, muscle):  

        
    # SE force-length relation
    l_se = l_mtu_current - l_ce_current
    #SE
    l_se_norm = l_se/l_slack_muscle[muscle]
    
    if l_se_norm > 1:
        f_se = ((l_se_norm-1)/epsilon_ref)**2
    else:
        f_se = 0    
    #BE
    l_ce_norm = l_ce_current/l_opt_muscle[muscle]
    if l_ce_norm -1 + w_muscle < 0 :
        f_be = (2*(l_ce_norm-1+w_muscle)/w_muscle)**2 
    else: 
        f_be = 0 
    #PE
    if l_ce_norm > 1 :
        f_pe =  ((l_ce_norm-1)/w_muscle)**2
    else: 
        f_pe = 0
    #CE
    f_ce = np.exp(c*(np.abs(l_ce_norm-1)/w_muscle)**3)
    #force vitesse
    f_v = (f_se + f_be)/(f_pe + f_ce * Act)
    #vitesse normalisée
    if f_v <= 1 :
        v_norm = (f_v-1)/(f_v * K_muscle + 1)
        
    elif  f_v > 1 and f_v <= N_muscle:
        v_norm = ((f_v-N_muscle)/(N_muscle-1)+1)/(1-7.56*K_muscle*(f_v-N_muscle)/(N_muscle-1))  
        
    elif f_v > N_muscle:
        v_norm = 0.01*(f_v-N_muscle)+1
        
    #vitesse elem contractil
    v_ce_norm = v_norm * v_max_muscle[muscle] * l_opt_muscle[muscle]
    # force muscle
    f_m = f_se *F_max_muscle[muscle]
    
    return v_ce_norm, f_m



# 2) deuxième fonction qui permet d'integrer v_ce en utilisant la méthode des trapèzes à 2 step pour obtenir lce;
#
# parametre in : pas de temps (que l'on peut obtenir), nombre d'itérations (5), index du muscle, activation, index de la jambe, current time 
# parametre out : v_ce qui est la vitesse de contraction de l'élément contractile, soit la dérivé de la longueur de l'élément contractile
#


counter=0
import matplotlib.pyplot as plt

def interpol(time,signal,t):   #,N_points,dt):

    t=round(t,10)
    #print('t',t)
    f= interp1d(time,signal, kind = "linear", fill_value="extrapolate")
    signal_interpol = f(t)

        
 

    return signal_interpol



def integrateur2000(x0,dt,N_iter,l_mtc_memory,l_act_memory,tsim,muscle,time_vector):

    lce_curr = x0
        
    vce_curr = vce_compute(lce_curr, l_mtc_memory[-1], l_act_memory[-1], muscle)[0]
    
    if lce_curr + vce_curr * dt> 0 :
        lce_curr = (lce_curr + vce_curr * dt)
    else: 
        lce_curr = 0
        
    return lce_curr
        




    


# 3) 3ème fonction qui permet d'updater la valeur de la longueur de l'unité musculaire l_mtu[m] en fonction de l'angle de l'articulation
#
# parametre in : Angle de l'articulation qu'on peut obtenir grace à Robotran [rad], angle de la deuxième articulation si y en a une [rad], 
#index du muscle
# parametre out : nouvelle longueur de l'unité musculaire [m]

def lmtu_updateVAS(phi,muscle):
    delta_lmtu = + rho[knee,VAS]*r_0[knee,VAS]*(  np.sin(phi_ref[knee,VAS]-phi_max[knee,VAS]) - np.sin(phi-phi_max[knee,VAS])  )
    return l_opt_muscle[muscle] + l_slack_muscle[muscle] + delta_lmtu

def lmtu_updateSOL(phi,muscle):
    
    delta_lmtu = + rho[ankle,SOL]*r_0[ankle,SOL]*(np.sin(phi_ref[ankle,SOL]-phi_max[ankle,SOL]) - np.sin(phi-phi_max[ankle,SOL]))
    return l_opt_muscle[muscle] + l_slack_muscle[muscle] + delta_lmtu

def lmtu_updateGAS(phi,phi2,muscle):
    
    delta_lmtu = (+ rho[ankle,GAS]*r_0[ankle,GAS]*(np.sin(phi_ref[ankle,GAS]-phi_max[ankle,GAS]) - np.sin(phi-phi_max[ankle,GAS])) 
    - rho[knee,GAS]*r_0[knee,GAS]*(np.sin(phi_ref[knee,GAS]-phi_max[knee,GAS]) - np.sin(phi2-phi_max[knee,GAS])))
    return l_opt_muscle[muscle] + l_slack_muscle[muscle] + delta_lmtu

def lmtu_updateTA(phi,muscle):
    
    delta_lmtu = - rho[ankle,TA]*r_0[ankle,TA]*(np.sin(phi_ref[ankle,TA]-phi_max[ankle,TA]) - np.sin(phi-phi_max[ankle,TA]))
    return l_opt_muscle[muscle] + l_slack_muscle[muscle] + delta_lmtu
    
def lmtu_updateHAM(phi,phi2,muscle):
    
    delta_lmtu = (- rho[knee,HAM]*r_0[knee,HAM]*(np.sin(phi_ref[knee,HAM]-phi_max[knee,HAM]) - np.sin(phi-phi_max[knee,HAM]))
    - rho[hip,HAM]*r_0[hip,HAM]*(phi2-phi_ref[hip,HAM]))
    return l_opt_muscle[muscle] + l_slack_muscle[muscle] + delta_lmtu

def lmtu_updateGLU(phi,muscle):
    
    delta_lmtu = - rho[hip,GLU]*r_0[hip,GLU]*(phi-phi_ref[hip,GLU])
    return l_opt_muscle[muscle] + l_slack_muscle[muscle] + delta_lmtu

def lmtu_updateHFL(phi,muscle):
    
    delta_lmtu = + rho[hip,HFL]*r_0[hip,HFL]*(phi-phi_ref[hip,HFL])
    return l_opt_muscle[muscle] + l_slack_muscle[muscle] + delta_lmtu



#4) fonction qui permet de calculer le torque creer par la force musculaire d'un muscle sur une articulation spécifique 
#
# parametre in : Angle de l'articulation que l'on peut obtenir sur Robotran, Forces motrice , index du muscle
# parametre out : Torque
#

def torque_updateHIP(phi,F_m,muscle):
    
    lever = r_0[hip,muscle]
    
    return lever*F_m

def torque_updateANKLE(phi,F_m,muscle):
    
    lever = r_0[ankle,muscle]*np.cos(phi-phi_max[ankle,muscle])
    
    return lever*F_m


def torque_updateKNEE(phi,F_m,muscle):
    
    lever = r_0[knee,muscle]*np.cos(phi-phi_max[knee,muscle])
    
    return lever*F_m


# 5) fonction qui permet d'appliquer les joint limits lorsque l'angle de l'articulation dépasse les limites, il y a un torque 
# qui est additionné pour contrecarer
#
# parametre in : articulation, l'angle de l'articulation [rad] et la vitesse angulaire de l'articulation
# parametre out : Torque joints limits
#

def joint_limits(articulation,phi,dphi):
    
    phi_a_up = 2.2689
    phi_a_low = 1.2217
    phi_k_up = 3.0543
    phi_h_up = 4.0143
    c_joint = 17.1887
    w_max = 0.0175
    
    if articulation == ankle : # ankle joint limits 
    
        if phi> 30 * np.pi/180 and phi < 170 * np.pi/180 : #check that the angle values do not take unrealistic values
            
            u1 = (phi- phi_a_up)*c_joint
            u2 = - dphi / w_max
            u3 = (phi- phi_a_low)*c_joint
            u4 = dphi/w_max
            
            if u2<1 and u1>0 :
                
                
                ext_torque = -u1*(1-u2)
            
            else : 
                
                ext_torque = 0
            
            if u4<1 and u3 < 0 :
                
                
                flex_torque = -u3*(1-u4)
            
            else :
                
                flex_torque = 0
            
            return (ext_torque + flex_torque)
        
        else : # raise an error if unrealistic values 
            
            #raise NameError('unrealistic values for the ankle joint')
            return 0
    
    elif articulation == knee : # knee joint limits
    
        if phi> 40 * np.pi/180 and phi < 185 * np.pi/180 : #check that the angle values do not take unrealistic values
    
            #print("knee")
    
            u1 = (phi- phi_k_up)*c_joint
            u2 = - dphi / w_max
            
            if u2<1 and u1>0 :
                
                ext_torque = -u1*(1-u2)
            
            else : 
                
                ext_torque = 0
            
            return ext_torque
        
        else : # raise an error if unrealistic values 
            
            #raise NameError('unrealistic values for the knee joint')
            return 0
    
    elif articulation == hip : # hip joint limits 
    
        if phi> 10 * np.pi/180 and phi < 260 * np.pi/180 : #check that the angle values do not take unrealistic values
            
            #print("hip")
            
            u1 = (phi- phi_h_up)*c_joint
            u2 = - dphi / w_max
            
            if u2<1 and u1>0 :
                
                ext_torque = -u1*(1-u2)
            
            else : 
                
                ext_torque = 0
            
            return ext_torque
        
        else : # raise an error if unrealistic values 
        
           #raise NameError('unrealistic values for the hip joint')
           return 0




        
import sys
import os
# Get the directory where your script is located
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(2,  os.path.join(parent_dir, "workR"))
import TestworkR


if __name__ == "__main__":
    TestworkR.runtest(1000e-7,3,c=False)
    
