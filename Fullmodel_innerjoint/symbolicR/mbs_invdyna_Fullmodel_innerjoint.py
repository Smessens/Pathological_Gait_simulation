#
#	MBsysTran - Release 8.1
#
#	Copyright 
#	Universite catholique de Louvain (UCLouvain) 
#	Mechatronic, Electrical Energy, and Dynamic systems (MEED Division) 
#	2, Place du Levant
#	1348 Louvain-la-Neuve 
#	Belgium 
#
#	http://www.robotran.be 
#
#	==> Generation Date: Sat May  6 16:31:24 2023
#
#	==> Project name: Fullmodel_innerjoint
#
#	==> Number of joints: 14
#
#	==> Function: F2 - Recursive Inverse Dynamics of tree-like MBS
#
#	==> Git hash: 8f46effdc05c898000a15b4c4dfc8f70efce4fc0
#

from math import sin, cos

def invdyna(phi,s,tsim):
    Qq = phi  # compatibility with symbolic generation
    q = s.q
    qd = s.qd
    qdd = s.qdd
 
# Trigonometric functions

    S4 = sin(q[4])
    C4 = cos(q[4])
    S5 = sin(q[5])
    C5 = cos(q[5])
    S6 = sin(q[6])
    C6 = cos(q[6])
    S7 = sin(q[7])
    C7 = cos(q[7])
    S8 = sin(q[8])
    C8 = cos(q[8])
    S10 = sin(q[10])
    C10 = cos(q[10])
    S11 = sin(q[11])
    C11 = cos(q[11])
    S13 = sin(q[13])
    C13 = cos(q[13])
    S14 = sin(q[14])
    C14 = cos(q[14])
 
# Augmented Joint Position Vectors

    Dz91 = q[9]+s.dpt[1,5]
    Dz121 = q[12]+s.dpt[1,10]
 
# Augmented Joint Position Vectors

 
# Forward Kinematics

    ALPHA33 = qdd[3]-s.g[3]
    ALPHA24 = qdd[2]*C4+ALPHA33*S4
    ALPHA34 = -qdd[2]*S4+ALPHA33*C4
    OM15 = qd[4]*C5
    OM35 = qd[4]*S5
    OMp15 = -qd[4]*qd[5]*S5+qdd[4]*C5
    OMp35 = qd[4]*qd[5]*C5+qdd[4]*S5
    ALPHA15 = qdd[1]*C5-ALPHA34*S5
    ALPHA35 = qdd[1]*S5+ALPHA34*C5
    OM16 = qd[5]*S6+OM15*C6
    OM26 = qd[5]*C6-OM15*S6
    OM36 = qd[6]+OM35
    OMp16 = C6*(OMp15+qd[5]*qd[6])+S6*(qdd[5]-qd[6]*OM15)
    OMp26 = C6*(qdd[5]-qd[6]*OM15)-S6*(OMp15+qd[5]*qd[6])
    OMp36 = qdd[6]+OMp35
    BS16 = -OM26*OM26-OM36*OM36
    BS26 = OM16*OM26
    BS36 = OM16*OM36
    BETA46 = BS26+OMp36
    BETA76 = BS36-OMp26
    ALPHA16 = ALPHA15*C6+ALPHA24*S6
    ALPHA26 = -ALPHA15*S6+ALPHA24*C6
    OM17 = OM16*C7-OM36*S7
    OM27 = qd[7]+OM26
    OM37 = OM16*S7+OM36*C7
    OMp17 = C7*(OMp16-qd[7]*OM36)-S7*(OMp36+qd[7]*OM16)
    OMp27 = qdd[7]+OMp26
    OMp37 = C7*(OMp36+qd[7]*OM16)+S7*(OMp16-qd[7]*OM36)
    BS17 = -OM27*OM27-OM37*OM37
    BS27 = OM17*OM27
    BS37 = OM17*OM37
    BETA47 = BS27+OMp37
    BETA77 = BS37-OMp27
    ALPHA17 = C7*(ALPHA16+BS16*s.dpt[1,2])-S7*(ALPHA35+BETA76*s.dpt[1,2])
    ALPHA27 = ALPHA26+BETA46*s.dpt[1,2]
    ALPHA37 = C7*(ALPHA35+BETA76*s.dpt[1,2])+S7*(ALPHA16+BS16*s.dpt[1,2])
    OM18 = OM17*C8-OM37*S8
    OM28 = qd[8]+OM27
    OM38 = OM17*S8+OM37*C8
    OMp18 = C8*(OMp17-qd[8]*OM37)-S8*(OMp37+qd[8]*OM17)
    OMp28 = qdd[8]+OMp27
    OMp38 = C8*(OMp37+qd[8]*OM17)+S8*(OMp17-qd[8]*OM37)
    BS18 = -OM28*OM28-OM38*OM38
    BS28 = OM18*OM28
    BS38 = OM18*OM38
    BETA48 = BS28+OMp38
    BETA78 = BS38-OMp28
    ALPHA18 = C8*(ALPHA17+BS17*s.dpt[1,4])-S8*(ALPHA37+BETA77*s.dpt[1,4])
    ALPHA28 = ALPHA27+BETA47*s.dpt[1,4]
    ALPHA38 = C8*(ALPHA37+BETA77*s.dpt[1,4])+S8*(ALPHA17+BS17*s.dpt[1,4])
    BS19 = -OM28*OM28-OM38*OM38
    BS29 = OM18*OM28
    BS39 = OM18*OM38
    BETA49 = BS29+OMp38
    BETA79 = BS39-OMp28
    ALPHA19 = qdd[9]+ALPHA18+BS18*Dz91
    ALPHA29 = ALPHA28+(2.0)*qd[9]*OM38+BETA48*Dz91
    ALPHA39 = ALPHA38-(2.0)*qd[9]*OM28+BETA78*Dz91
    OM110 = OM18*C10-OM38*S10
    OM210 = qd[10]+OM28
    OM310 = OM18*S10+OM38*C10
    OMp110 = C10*(OMp18-qd[10]*OM38)-S10*(OMp38+qd[10]*OM18)
    OMp210 = qdd[10]+OMp28
    OMp310 = C10*(OMp38+qd[10]*OM18)+S10*(OMp18-qd[10]*OM38)
    BS110 = -OM210*OM210-OM310*OM310
    BS210 = OM110*OM210
    BS310 = OM110*OM310
    BETA410 = BS210+OMp310
    BETA710 = BS310-OMp210
    ALPHA110 = C10*(ALPHA19+BS19*s.dpt[1,6])-S10*(ALPHA39+BETA79*s.dpt[1,6])
    ALPHA210 = ALPHA29+BETA49*s.dpt[1,6]
    ALPHA310 = C10*(ALPHA39+BETA79*s.dpt[1,6])+S10*(ALPHA19+BS19*s.dpt[1,6])
    OM111 = OM110*C11-OM310*S11
    OM211 = qd[11]+OM210
    OM311 = OM110*S11+OM310*C11
    OMp111 = C11*(OMp110-qd[11]*OM310)-S11*(OMp310+qd[11]*OM110)
    OMp211 = qdd[11]+OMp210
    OMp311 = C11*(OMp310+qd[11]*OM110)+S11*(OMp110-qd[11]*OM310)
    BS111 = -OM211*OM211-OM311*OM311
    BS211 = OM111*OM211
    BS311 = OM111*OM311
    BETA411 = BS211+OMp311
    BETA711 = BS311-OMp211
    ALPHA111 = ALPHA110*C11-ALPHA310*S11
    ALPHA311 = ALPHA110*S11+ALPHA310*C11
    BS112 = -OM211*OM211-OM311*OM311
    BS212 = OM111*OM211
    BS312 = OM111*OM311
    BETA412 = BS212+OMp311
    BETA712 = BS312-OMp211
    ALPHA112 = qdd[12]+ALPHA111+BS111*Dz121
    ALPHA212 = ALPHA210+(2.0)*qd[12]*OM311+BETA411*Dz121
    ALPHA312 = ALPHA311-(2.0)*qd[12]*OM211+BETA711*Dz121
    OM113 = OM111*C13-OM311*S13
    OM213 = qd[13]+OM211
    OM313 = OM111*S13+OM311*C13
    OMp113 = C13*(OMp111-qd[13]*OM311)-S13*(OMp311+qd[13]*OM111)
    OMp213 = qdd[13]+OMp211
    OMp313 = C13*(OMp311+qd[13]*OM111)+S13*(OMp111-qd[13]*OM311)
    BS113 = -OM213*OM213-OM313*OM313
    BS213 = OM113*OM213
    BS313 = OM113*OM313
    BETA413 = BS213+OMp313
    BETA713 = BS313-OMp213
    ALPHA113 = C13*(ALPHA112+BS112*s.dpt[1,11])-S13*(ALPHA312+BETA712*s.dpt[1,11])
    ALPHA213 = ALPHA212+BETA412*s.dpt[1,11]
    ALPHA313 = C13*(ALPHA312+BETA712*s.dpt[1,11])+S13*(ALPHA112+BS112*s.dpt[1,11])
    OM114 = OM113*C14-OM313*S14
    OM214 = qd[14]+OM213
    OM314 = OM113*S14+OM313*C14
    OMp214 = qdd[14]+OMp213
    OMp314 = C14*(OMp313+qd[14]*OM113)+S14*(OMp113-qd[14]*OM313)
    BS114 = -OM214*OM214-OM314*OM314
    BS214 = OM114*OM214
    BS314 = OM114*OM314
    BETA414 = BS214+OMp314
    BETA714 = BS314-OMp214
    ALPHA114 = C14*(ALPHA113+BS113*s.dpt[1,12])-S14*(ALPHA313+BETA713*s.dpt[1,12])
    ALPHA214 = ALPHA213+BETA413*s.dpt[1,12]
    ALPHA314 = C14*(ALPHA313+BETA713*s.dpt[1,12])+S14*(ALPHA113+BS113*s.dpt[1,12])
 
# Backward Dynamics

    Fs114 = -s.frc[1,14]+s.m[14]*(ALPHA114+BS114*s.l[1,14])
    Fs214 = -s.frc[2,14]+s.m[14]*(ALPHA214+BETA414*s.l[1,14])
    Fs314 = -s.frc[3,14]+s.m[14]*(ALPHA314+BETA714*s.l[1,14])
    Cq114 = -s.trq[1,14]-s.In[5,14]*OM214*OM314
    Cq214 = -s.trq[2,14]+s.In[5,14]*OMp214-Fs314*s.l[1,14]
    Cq314 = -s.trq[3,14]+s.In[5,14]*OM114*OM214+Fs214*s.l[1,14]
    Fs113 = -s.frc[1,13]+s.m[13]*(ALPHA113+BS113*s.l[1,13])
    Fs213 = -s.frc[2,13]+s.m[13]*(ALPHA213+BETA413*s.l[1,13])
    Fs313 = -s.frc[3,13]+s.m[13]*(ALPHA313+BETA713*s.l[1,13])
    Fq113 = Fs113+Fs114*C14+Fs314*S14
    Fq213 = Fs213+Fs214
    Fq313 = Fs313-Fs114*S14+Fs314*C14
    Cq113 = -s.trq[1,13]-s.In[5,13]*OM213*OM313+Cq114*C14+Cq314*S14
    Cq213 = -s.trq[2,13]+Cq214+s.In[5,13]*OMp213-Fs313*s.l[1,13]-s.dpt[1,12]*(-Fs114*S14+Fs314*C14)
    Cq313 = -s.trq[3,13]+s.In[5,13]*OM113*OM213-Cq114*S14+Cq314*C14+Fs213*s.l[1,13]+Fs214*s.dpt[1,12]
    Fs112 = -s.frc[1,12]+s.m[12]*(ALPHA112+BS112*s.l[1,12])
    Fs212 = -s.frc[2,12]+s.m[12]*(ALPHA212+BETA412*s.l[1,12])
    Fs312 = -s.frc[3,12]+s.m[12]*(ALPHA312+BETA712*s.l[1,12])
    Fq112 = Fs112+Fq113*C13+Fq313*S13
    Fq212 = Fq213+Fs212
    Fq312 = Fs312-Fq113*S13+Fq313*C13
    Cq112 = -s.trq[1,12]-s.In[5,12]*OM211*OM311+Cq113*C13+Cq313*S13
    Cq212 = -s.trq[2,12]+Cq213+s.In[5,12]*OMp211-Fs312*s.l[1,12]-s.dpt[1,11]*(-Fq113*S13+Fq313*C13)
    Cq312 = -s.trq[3,12]+s.In[5,12]*OM111*OM211-Cq113*S13+Cq313*C13+Fq213*s.dpt[1,11]+Fs212*s.l[1,12]
    Fs111 = -s.frc[1,11]+s.m[11]*(ALPHA111+BS111*s.l[1,11])
    Fs211 = -s.frc[2,11]+s.m[11]*(ALPHA210+BETA411*s.l[1,11])
    Fs311 = -s.frc[3,11]+s.m[11]*(ALPHA311+BETA711*s.l[1,11])
    Fq111 = Fq112+Fs111
    Fq211 = Fq212+Fs211
    Fq311 = Fq312+Fs311
    Cq111 = -s.trq[1,11]+Cq112-s.In[5,11]*OM211*OM311
    Cq211 = -s.trq[2,11]+Cq212+s.In[5,11]*OMp211-Fq312*Dz121-Fs311*s.l[1,11]
    Cq311 = -s.trq[3,11]+Cq312+s.In[5,11]*OM111*OM211+Fq212*Dz121+Fs211*s.l[1,11]
    Fs110 = -s.frc[1,10]+s.m[10]*(ALPHA110+BS110*s.l[1,10])
    Fs210 = -s.frc[2,10]+s.m[10]*(ALPHA210+BETA410*s.l[1,10])
    Fs310 = -s.frc[3,10]+s.m[10]*(ALPHA310+BETA710*s.l[1,10])
    Fq110 = Fs110+Fq111*C11+Fq311*S11
    Fq210 = Fq211+Fs210
    Fq310 = Fs310-Fq111*S11+Fq311*C11
    Cq110 = -s.trq[1,10]-s.In[5,10]*OM210*OM310+Cq111*C11+Cq311*S11
    Cq210 = -s.trq[2,10]+Cq211+s.In[5,10]*OMp210-Fs310*s.l[1,10]
    Cq310 = -s.trq[3,10]+s.In[5,10]*OM110*OM210-Cq111*S11+Cq311*C11+Fs210*s.l[1,10]
    Fs19 = -s.frc[1,9]+s.m[9]*(ALPHA19+BS19*s.l[1,9])
    Fs29 = -s.frc[2,9]+s.m[9]*(ALPHA29+BETA49*s.l[1,9])
    Fs39 = -s.frc[3,9]+s.m[9]*(ALPHA39+BETA79*s.l[1,9])
    Fq19 = Fs19+Fq110*C10+Fq310*S10
    Fq29 = Fq210+Fs29
    Fq39 = Fs39-Fq110*S10+Fq310*C10
    Cq19 = -s.trq[1,9]-s.In[5,9]*OM28*OM38+Cq110*C10+Cq310*S10
    Cq29 = -s.trq[2,9]+Cq210+s.In[5,9]*OMp28-Fs39*s.l[1,9]-s.dpt[1,6]*(-Fq110*S10+Fq310*C10)
    Cq39 = -s.trq[3,9]+s.In[5,9]*OM18*OM28-Cq110*S10+Cq310*C10+Fq210*s.dpt[1,6]+Fs29*s.l[1,9]
    Fs18 = -s.frc[1,8]+s.m[8]*(ALPHA18+BS18*s.l[1,8])
    Fs28 = -s.frc[2,8]+s.m[8]*(ALPHA28+BETA48*s.l[1,8])
    Fs38 = -s.frc[3,8]+s.m[8]*(ALPHA38+BETA78*s.l[1,8])
    Fq18 = Fq19+Fs18
    Fq28 = Fq29+Fs28
    Fq38 = Fq39+Fs38
    Cq18 = -s.trq[1,8]+Cq19-s.In[5,8]*OM28*OM38
    Cq28 = -s.trq[2,8]+Cq29+s.In[5,8]*OMp28-Fq39*Dz91-Fs38*s.l[1,8]
    Cq38 = -s.trq[3,8]+Cq39+s.In[5,8]*OM18*OM28+Fq29*Dz91+Fs28*s.l[1,8]
    Fs17 = -s.frc[1,7]+s.m[7]*(ALPHA17+BS17*s.l[1,7])
    Fs27 = -s.frc[2,7]+s.m[7]*(ALPHA27+BETA47*s.l[1,7])
    Fs37 = -s.frc[3,7]+s.m[7]*(ALPHA37+BETA77*s.l[1,7])
    Fq17 = Fs17+Fq18*C8+Fq38*S8
    Fq27 = Fq28+Fs27
    Fq37 = Fs37-Fq18*S8+Fq38*C8
    Cq17 = -s.trq[1,7]-s.In[5,7]*OM27*OM37+Cq18*C8+Cq38*S8
    Cq27 = -s.trq[2,7]+Cq28+s.In[5,7]*OMp27-Fs37*s.l[1,7]-s.dpt[1,4]*(-Fq18*S8+Fq38*C8)
    Cq37 = -s.trq[3,7]+s.In[5,7]*OM17*OM27-Cq18*S8+Cq38*C8+Fq28*s.dpt[1,4]+Fs27*s.l[1,7]
    Fs16 = -s.frc[1,6]+s.m[6]*(ALPHA16+BS16*s.l[1,6])
    Fs26 = -s.frc[2,6]+s.m[6]*(ALPHA26+BETA46*s.l[1,6])
    Fs36 = -s.frc[3,6]+s.m[6]*(ALPHA35+BETA76*s.l[1,6])
    Fq16 = Fs16+Fq17*C7+Fq37*S7
    Fq26 = Fq27+Fs26
    Fq36 = Fs36-Fq17*S7+Fq37*C7
    Cq16 = -s.trq[1,6]-s.In[5,6]*OM26*OM36+Cq17*C7+Cq37*S7
    Cq26 = -s.trq[2,6]+Cq27+s.In[5,6]*OMp26-Fs36*s.l[1,6]-s.dpt[1,2]*(-Fq17*S7+Fq37*C7)
    Cq36 = -s.trq[3,6]+s.In[5,6]*OM16*OM26-Cq17*S7+Cq37*C7+Fq27*s.dpt[1,2]+Fs26*s.l[1,6]
    Fq15 = Fq16*C6-Fq26*S6
    Fq25 = Fq16*S6+Fq26*C6
    Cq15 = Cq16*C6-Cq26*S6
    Cq25 = Cq16*S6+Cq26*C6
    Fq14 = Fq15*C5+Fq36*S5
    Fq34 = -Fq15*S5+Fq36*C5
    Cq14 = Cq15*C5+Cq36*S5
    Fq23 = Fq25*C4-Fq34*S4
    Fq33 = Fq25*S4+Fq34*C4
 
# Symbolic model output

    Qq[1] = Fq14
    Qq[2] = Fq23
    Qq[3] = Fq33
    Qq[4] = Cq14
    Qq[5] = Cq25
    Qq[6] = Cq36
    Qq[7] = Cq27
    Qq[8] = Cq28
    Qq[9] = Fq19
    Qq[10] = Cq210
    Qq[11] = Cq211
    Qq[12] = Fq112
    Qq[13] = Cq213
    Qq[14] = Cq214

# Number of continuation lines = 0


