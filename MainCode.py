import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
import scipy.optimize as optimize

import matplotlib.pyplot as plt
from functools import partial
import multiprocessing 
import TransitionFunc
from scipy.stats import frechet_r
print("BaseTradeLC")

################################################################################################################
######################################  Basic Parameters #######################################################
################################################################################################################
No_Sectors = 4
Age_Periods = 40
Beta = 0.98
UnempBenefit=0.7
High_Ability=0.35
Swith_Cost0=0
Switch_Cost =5.
Switch_Cost1=10.
Switch_Cost2=14.
Switch_All = np.array((Swith_Cost0,Switch_Cost,Switch_Cost1,Switch_Cost2))

Calib_Timing = 4 #quarterly

################################################################################################################
######################################  Production Function ####################################################
################################################################################################################


alpha_out=0.7  #25-30% went to manufacturing
eta_mfg=3.   
eta_ins = 2.
TFP_L=1.     
TFP_M = 1.1
TFP_M_B = 1.5
TFP_H = 1.6
AR1 = 0.36
Price_Multiplier_Foreign=1.25
Non_Pec=0.15

################################################################################################################
######################################  Ability Types ##########################################################
################################################################################################################
#
Skills = 2
SkillLow = 1.
SkillHigh = 1.5
Skill_vals = np.array((SkillHigh,SkillLow))

################################################################################################################
######################################### Firm Tenure ##########################################################
################################################################################################################

Experience_Path=np.ones((Age_Periods))  #Number from Buchinsky et al
Experience_Path[1]=1.06
Experience_Path[2]=1.12
Path = np.concatenate([np.repeat(1,2),np.repeat(1.05,3),np.repeat(1.04,5),np.repeat(1.03,5),np.repeat(1.02,10),
                       np.repeat(1.005,15)])
for ii in range(3,Age_Periods):
    Experience_Path[ii]=Experience_Path[ii-1]*Path[ii]
Experience_Levels = 6
Exper=np.ones((Experience_Levels))
for ii in range(1,Experience_Levels):
    Exper[Experience_Levels-ii-1]=Experience_Path[ii*int(np.round(Age_Periods/(Experience_Levels),0))-1]


Exper_Gain=1-0.1**(1/(np.round(Age_Periods/(Experience_Levels-1),0)*Calib_Timing))
Exper_Loss = Exper_Gain/10
Exper_Gain_Matrix = np.zeros((Experience_Levels,Experience_Levels))
for ii in range(Experience_Levels):
    Exper_Gain_Matrix[ii,int(np.maximum(0,ii-1))]=Exper_Gain
    Exper_Gain_Matrix[ii,int(np.minimum(Experience_Levels-1,ii+1))]=Exper_Loss
    Exper_Gain_Matrix[ii,ii]=1-Exper_Gain_Matrix[ii,:].sum()+Exper_Gain_Matrix[ii,ii]

################################################################################################################
######################################### Sector Experience ####################################################
################################################################################################################

Sector_Path=np.ones((Age_Periods))
Sector_Path[0:6]=np.linspace(1,1.27,6) #Number from Buchinsky et al
Path_Sector = np.concatenate([np.repeat(1,6),np.repeat(1.02,5),np.repeat(1.015,5),np.repeat(1.01,5),np.repeat(1.01,10),
                              np.repeat(1.0075,9)])
for ii in range(6,Age_Periods):
    Sector_Path[ii]=Sector_Path[ii-1]*Path_Sector[ii]
    
Sector_XP_Levels = 3

Sector_XP_Gain=np.ones((Sector_XP_Levels))
for ii in range(1,Sector_XP_Levels):
    Sector_XP_Gain[Sector_XP_Levels-ii-1]=Sector_Path[ii*int(np.round(Age_Periods/(Sector_XP_Levels-1),0))-1]

Sector_XP_Gain_Rate = 1-0.1**(1/(np.round(Age_Periods/(Sector_XP_Levels-1),0)*Calib_Timing))
Sector_XP_Loss_Rate = Sector_XP_Gain_Rate/10
Sector_XP_Gain_Matrix = np.zeros((Sector_XP_Levels,Sector_XP_Levels))
for ii in range(Sector_XP_Levels):
    Sector_XP_Gain_Matrix[ii,int(np.maximum(0,ii-1))]=Sector_XP_Gain_Rate
    Sector_XP_Gain_Matrix[ii,int(np.minimum(Sector_XP_Levels-1,ii+1))]=Sector_XP_Loss_Rate
    Sector_XP_Gain_Matrix[ii,ii]=1-Sector_XP_Gain_Matrix[ii,:].sum()+Sector_XP_Gain_Matrix[ii,ii]

################################################################################################################
######################################### Production Shocks  ###################################################
################################################################################################################

Productivity_Params=5

Prod_Change_Matrix=np.zeros((No_Sectors,Productivity_Params,Productivity_Params))

def Prod_Matrix(Persistence):
    Prod_Matrix = np.zeros((Productivity_Params,Productivity_Params))
    for ii in range(Productivity_Params):
        Prod_Matrix[ii,:]=np.repeat((1-Persistence)/(Productivity_Params-1),Productivity_Params)
        Prod_Matrix[ii,ii]=Persistence
    JD=0.8
    Prod_Matrix[-1,:]=np.repeat((1-JD)/(Productivity_Params-1),Productivity_Params)
    Prod_Matrix[-1,-1]=JD
    return Prod_Matrix 
Prod_Change_Matrix[0] = Prod_Matrix(AR1)
Prod_Change_Matrix[1] = Prod_Matrix(AR1)
Prod_Change_Matrix[2] = Prod_Matrix(AR1)
Prod_Change_Matrix[3] = Prod_Matrix(AR1)
#Prod_Change_Matrix[0],Prod_Vals_Sector0 = TransitionFunc.Rouwenhorst(0.3,0.797,Productivity_Params)
#Prod_Change_Matrix[1],Prod_Vals_Sector1 = TransitionFunc.Rouwenhorst(0.3,0.658,Productivity_Params)
#Prod_Change_Matrix[2],Prod_Vals_Sector1b = TransitionFunc.Rouwenhorst(0.3,0.542,Productivity_Params)
#Prod_Change_Matrix[3],Prod_Vals_Sector2 = TransitionFunc.Rouwenhorst(0.3,0.564,Productivity_Params)

Prod_Vals_Base= np.linspace(1.3,0.001,Productivity_Params)
Prod_Vals_Sector0=TFP_L*Prod_Vals_Base
Prod_Vals_Sector1=TFP_M*Prod_Vals_Base
Prod_Vals_Sector1b=TFP_M_B*Prod_Vals_Base
Prod_Vals_Sector2=TFP_H*Prod_Vals_Base

Prod_Vals_All = np.concatenate((Prod_Vals_Sector0,Prod_Vals_Sector1,Prod_Vals_Sector1b,Prod_Vals_Sector2))
Prod_Vals = np.zeros((Productivity_Params*No_Sectors))
for ii in range(No_Sectors):
    Prod_Vals[ii*Productivity_Params:(ii+1)*(Productivity_Params)] = Prod_Vals_All[ii*Productivity_Params:(ii+1)*(Productivity_Params)]
    
Prod_Draw=np.zeros((Productivity_Params))
Prod_Draw[-1]=0.18
Prod_Draw[-2]=0.12 
Prod_Draw[0]=0.05
Prod_Draw[1]=0.08 
Prod_Draw[2]=1-Prod_Draw.sum()



################################################################################################################
######################################### Matching Fnc Parameters  #############################################
################################################################################################################



Fraction_Worker_Surplus = 0.35

#Matching Function Parameters
Vacancy_Cost = np.array([0.22,0.5,1,1])
Scale = np.array([0.7,0.6,0.6,0.7])
sigma = np.array([0.72,0.72,0.72,0.72])


#Val Functions
Production_Possibilities = Sector_XP_Levels*Productivity_Params*Experience_Levels+1*(Sector_XP_Levels)
Firm_Val_Func_High = np.zeros((Age_Periods,No_Sectors*(Production_Possibilities))) #+1 for unemployment state
Firm_Val_Func_Low = 0*Firm_Val_Func_High
Work_Val_Func_High = 0*Firm_Val_Func_High
Work_Val_Func_Low = 0*Firm_Val_Func_High
for ii in range(No_Sectors):
    Work_Val_Func_High[:,-1-(ii*Production_Possibilities)]=Skill_vals[0]*UnempBenefit
    Work_Val_Func_Low[:,-1-(ii*Production_Possibilities)]=Skill_vals[1]*UnempBenefit

#Swithching Cost Matrix Vectors:
Switch_Vector = np.repeat(Switch_All,Production_Possibilities)
SC_Mfg=0.82
Switcher_Matrix = np.repeat(np.repeat(1-np.eye(No_Sectors),Production_Possibilities,axis=1),Production_Possibilities,axis=0)
Switcher_Matrix[Production_Possibilities:Production_Possibilities*2,Production_Possibilities*2:Production_Possibilities*3] = SC_Mfg


Price_Sector1= 0.4
Price_Sector2 = 0.38
Price_Sector2b = 0.36
Price_Sector3 = 0.35
Price=np.repeat(np.array((Price_Sector1,Price_Sector2,Price_Sector2b,Price_Sector3)),Production_Possibilities)

#Setting the unemployment benefit
Unemp_High = Skill_vals[0]*UnempBenefit
Unemp_Low = Skill_vals[1]*UnempBenefit

Unemp_Val_Func_High=Unemp_High *(np.ones((Age_Periods,No_Sectors)))
Unemp_Val_Func_Low= Unemp_Low*(np.ones((Age_Periods,No_Sectors)))


Firm_Transition_Matrices_High =np.zeros((Age_Periods-1,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
Firm_Transition_Matrices_Low=np.zeros((Age_Periods-1,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
Worker_Transition_Matrices_High =np.zeros((Age_Periods-1,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
Worker_Transition_Matrices_Low=np.zeros((Age_Periods-1,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))

Firm_Continuation_Value_High = 0*Firm_Val_Func_High
Firm_Continuation_Value_Low = 0*Firm_Val_Func_High
Worker_Continuation_Value_High= 0*Firm_Val_Func_High
Worker_Continuation_Value_Low = 0*Firm_Val_Func_High
Unemp_Continuation_Value_High = 0*Firm_Val_Func_High
Unemp_Continuation_Value_Low = 0*Firm_Val_Func_High

#Vector checker:
Sector_Checker=np.eye(No_Sectors,No_Sectors)
Sector_Checker=np.repeat(Sector_Checker,Production_Possibilities,axis=1)


conv=10
iterate=0
iterator=0


#Initial Cohort Code:
  
High_Grid_Points = 48
Mid_Grid_Points = 48
Mid_Grid_Points_B = 48

c = 1.3
scale=2.5
loc=5.2
Cost_Mfg_Education_B = np.linspace(frechet_r.ppf(1/Mid_Grid_Points, c,loc,scale=scale),frechet_r.ppf(0.99, c,loc,scale=scale), Mid_Grid_Points)

c1 = 1.5
scale1=1.5
loc1=-0.15
Cost_Mfg_Education = np.linspace(frechet_r.ppf(1/Mid_Grid_Points, c1,loc1,scale=scale1),frechet_r.ppf(0.99,c1,loc1,scale=scale1), Mid_Grid_Points)
Cost_High_Education = np.linspace(6,18.,High_Grid_Points)

Grid_Points = int(High_Grid_Points*Mid_Grid_Points*Mid_Grid_Points_B)
Cost_Vector = np.zeros((4,int(High_Grid_Points*Mid_Grid_Points*Mid_Grid_Points_B/4),4))
Distr_Init_High = High_Ability/int(High_Grid_Points*Mid_Grid_Points*Mid_Grid_Points_B)+np.zeros((int(High_Grid_Points*Mid_Grid_Points*Mid_Grid_Points_B)))
Distr_Init_Low = (1-High_Ability)/int(High_Grid_Points*Mid_Grid_Points*Mid_Grid_Points_B)+np.zeros((int(High_Grid_Points*Mid_Grid_Points*Mid_Grid_Points_B)))

for ii in range(4):
    #Cost_Vector[ii,:,2]=np.repeat(Cost_High_Education,int(Mid_Grid_Points*3/High_Grid_Points))[int(Mid_Grid_Points)*ii:int(Mid_Grid_Points)*(ii+1)]#Cost_High_Education[int(Grid_Points/3)*ii:int(Grid_Points/3)*(ii+1)]
    #Cost_Vector[ii,:,1]=np.tile(Cost_Mfg_Education,3)[int(Mid_Grid_Points)*ii:int(Mid_Grid_Points)*(ii+1)]
    Cost_Vector[ii,:,3]=np.repeat(Cost_High_Education,int(Mid_Grid_Points*Mid_Grid_Points_B))[int(High_Grid_Points*Mid_Grid_Points*Mid_Grid_Points_B/4)*ii:int(High_Grid_Points*Mid_Grid_Points*Mid_Grid_Points_B/4)*(ii+1)]
    Cost_Vector[ii,:,2]=np.tile(np.repeat(Cost_Mfg_Education_B,int(Mid_Grid_Points_B*Mid_Grid_Points/High_Grid_Points)),High_Grid_Points)[int(High_Grid_Points*Mid_Grid_Points*Mid_Grid_Points_B/4)*ii:int(High_Grid_Points*Mid_Grid_Points*Mid_Grid_Points_B/4)*(ii+1)]
    Cost_Vector[ii,:,1]=np.tile(Cost_Mfg_Education,int(High_Grid_Points*Mid_Grid_Points_B*Mid_Grid_Points/4))[int(High_Grid_Points*Mid_Grid_Points*Mid_Grid_Points_B/4)*ii:int(High_Grid_Points*Mid_Grid_Points*Mid_Grid_Points_B/4)*(ii+1)]

Half = np.round(Productivity_Params/2)
Remaining = Productivity_Params-int(Half)

Prob_Draw_SHigh=np.zeros((Production_Possibilities))
High_Vals = np.concatenate((np.tile(np.array([0.8/Half]),int(Half)),np.tile(np.array([0.1/Remaining]),Productivity_Params-int(Half)),np.array([1-0.8-0.1])))
Prob_Draw_SHigh[-High_Vals.shape[0]:]=High_Vals


Prob_Draw_High=np.zeros((Production_Possibilities))
High_Vals = np.concatenate((np.tile(np.array([0.5/Half]),int(Half)),np.tile(np.array([0.4/Remaining]),Productivity_Params-int(Half)),np.array([1-0.4-0.5])))
Prob_Draw_High[-High_Vals.shape[0]:]=High_Vals

Prob_Draw_Mid=np.zeros((Production_Possibilities))
High_Vals = np.concatenate((np.tile(np.array([0.4/Half]),int(Half)),np.tile(np.array([0.5/Remaining]),Productivity_Params-int(Half)),np.array([1-0.5-0.4])))
Prob_Draw_Mid[-High_Vals.shape[0]:]=High_Vals

Prob_Draw_Low=np.zeros((Production_Possibilities))
High_Vals = np.concatenate((np.tile(np.array([0.1/Half]),int(Half)),np.tile(np.array([0.3/Remaining]),Productivity_Params-int(Half)),np.array([1-0.1-0.3])))
Prob_Draw_Low[-High_Vals.shape[0]:]=High_Vals

Prob_Draw=np.vstack((Prob_Draw_SHigh,Prob_Draw_Mid,Prob_Draw_Low))

Prob_Type_Best =np.vstack((Prob_Draw_SHigh,Prob_Draw_SHigh,Prob_Draw_SHigh,Prob_Draw_SHigh))
Prob_Type_Med =np.vstack((Prob_Draw_SHigh,Prob_Draw_SHigh,Prob_Draw_SHigh,Prob_Draw_Low))
Prob_Type_Med_B =np.vstack((Prob_Draw_High,Prob_Draw_SHigh,Prob_Draw_Low,Prob_Draw_Low))
Prob_Type_Lowest =np.vstack((Prob_Draw_SHigh,Prob_Draw_Low,Prob_Draw_Low,Prob_Draw_Low))
Prob_Types = np.stack((Prob_Type_Best,Prob_Type_Med,Prob_Type_Med_B,Prob_Type_Lowest))


def Initial_Cohort_Fixed(Work_Val_Func_High,Work_Val_Func_Low,Cost_Vector=Cost_Vector,Distr_Init_High=Distr_Init_High,
                         Distr_Init_Low=Distr_Init_Low):
    H_Servs = Work_Val_Func_High[0,-Production_Possibilities:]
    Mfg_B = Work_Val_Func_High[0,-2*Production_Possibilities:-Production_Possibilities]
    Mfg = Work_Val_Func_High[0,-3*Production_Possibilities:-2*Production_Possibilities]
    L_Servs = Work_Val_Func_High[0,-4*Production_Possibilities:-3*Production_Possibilities]
    
    H_Servs_L = Work_Val_Func_Low[0,-Production_Possibilities:]
    Mfg_L_B = Work_Val_Func_Low[0,-2*Production_Possibilities:-Production_Possibilities]
    Mfg_L = Work_Val_Func_Low[0,-3*Production_Possibilities:-2*Production_Possibilities]
    L_Servs_L = Work_Val_Func_Low[0,-4*Production_Possibilities:-3*Production_Possibilities]
    
    Job_Vals = np.vstack((L_Servs,Mfg,Mfg_B,H_Servs))
    Job_Vals_L = np.vstack((L_Servs_L,Mfg_L,Mfg_L_B,H_Servs_L))
    
    Combination_Vector= np.zeros((Prob_Draw.shape[0]*No_Sectors))
    Combination_Vector_L= np.zeros((Prob_Draw.shape[0]*No_Sectors))
    for ii in range(Combination_Vector.shape[0]):
        Combination_Vector[ii]=np.dot(Prob_Draw[int(ii/No_Sectors),],Job_Vals[ii % 4,])
        Combination_Vector_L[ii]=np.dot(Prob_Draw[int(ii/No_Sectors),],Job_Vals_L[ii % 4,])
   ##We are here     
    Poss_Choice_High=np.zeros((4))
    Poss_Choice_Mid=np.zeros((4))
    Poss_Choice_Mid_B=np.zeros((4))
    Poss_Choice_Low=np.zeros((4))
    
    Poss_Choice_High_L=np.zeros((4))
    Poss_Choice_Mid_L=np.zeros((4))
    Poss_Choice_Mid_L_B=np.zeros((4))
    Poss_Choice_Low_L=np.zeros((4))
    
    Poss_Choice_High = Combination_Vector[0:4].copy()
    Poss_Choice_High_L = Combination_Vector_L[0:4].copy()
    
    
    Poss_Choice_Mid_B = Combination_Vector[0:4].copy()
    Poss_Choice_Mid_B[-1]=Combination_Vector[-1].copy()
    Poss_Choice_Mid_L_B = Combination_Vector_L[0:4].copy()
    Poss_Choice_Mid_L_B[-1]=Combination_Vector_L[-1].copy()
    
    Poss_Choice_Mid = Combination_Vector[0:4].copy()
    Poss_Choice_Mid[-2:]=Combination_Vector[-2:].copy()
    Poss_Choice_Mid_L = Combination_Vector_L[0:4].copy()
    Poss_Choice_Mid_L[-2:]=Combination_Vector_L[-2:].copy()
    
    Poss_Choice_Low = Combination_Vector[-4:].copy()
    Poss_Choice_Low[0]=Combination_Vector[0].copy()
    Poss_Choice_Low_L = Combination_Vector_L[-4:].copy()
    Poss_Choice_Low_L[0]=Combination_Vector_L[0].copy()
    
    Poss_Choices = np.vstack((Poss_Choice_High,Poss_Choice_Mid_B,Poss_Choice_Mid,Poss_Choice_Low))
    Poss_Choices_L = np.vstack((Poss_Choice_High_L,Poss_Choice_Mid_L_B,Poss_Choice_Mid_L,Poss_Choice_Low_L))
    #Compare choices of high vs mid vs low by subtracting the relevant costs (create a cost vector)
    Choice_High = np.zeros((int(Grid_Points)))
    Choice_Low = np.zeros((int(Grid_Points)))
    Mid_Grid_Points=int(Grid_Points/4)
    Cost_Total_High=np.zeros((int(Grid_Points)))
    Cost_Total_Low=np.zeros((int(Grid_Points)))
    Inside_Grid = Mid_Grid_Points
    for ii in range(4):
        for kk in range(Inside_Grid):
            Choice_High[(ii*Inside_Grid)+kk]=np.argmax((Poss_Choices[ii]-Cost_Vector[ii,kk]),axis=0)
            Choice_Low[(ii*Inside_Grid)+kk]=np.argmax((Poss_Choices_L[ii]-Cost_Vector[ii,kk]),axis=0)
            Cost_Total_High[(ii*Inside_Grid)+kk] = Cost_Vector[ii,kk,int(Choice_High[(ii*Inside_Grid)+kk])]*Distr_Init_High[(ii*Inside_Grid)+kk]
            Cost_Total_Low[(ii*Inside_Grid)+kk] = Cost_Vector[ii,kk,int(Choice_Low[(ii*Inside_Grid)+kk])]*Distr_Init_Low[(ii*Inside_Grid)+kk]
    Init_Vector_High = np.zeros((Production_Possibilities*No_Sectors))
    Init_Vector_Low = np.zeros((Production_Possibilities*No_Sectors))
    for ii in range(4):
        for jj in range(Inside_Grid):
            Tempor_Probs_High=np.zeros((Production_Possibilities*No_Sectors))
            Tempor_Probs_Low=np.zeros((Production_Possibilities*No_Sectors))
            
            Tempor_Probs_High[int(Choice_High[(ii*Inside_Grid)+jj])*Production_Possibilities:
                (int(Choice_High[(ii*Inside_Grid)+jj])+1)*Production_Possibilities]=Distr_Init_High[
                        (ii*Inside_Grid)+jj]*Prob_Types[ii,int(Choice_High[(ii*Inside_Grid)+jj])]
            Init_Vector_High=Init_Vector_High+Tempor_Probs_High
            
            Tempor_Probs_Low[int(Choice_Low[(ii*Inside_Grid)+jj])*Production_Possibilities:
                (int(Choice_Low[(ii*Inside_Grid)+jj])+1)*Production_Possibilities]=Distr_Init_Low[
                        (ii*Inside_Grid)+jj]*Prob_Types[ii,int(Choice_Low[(ii*Inside_Grid)+jj])]
            Init_Vector_Low=Init_Vector_Low+Tempor_Probs_Low
            
    return Init_Vector_High,Init_Vector_Low,Cost_Total_High.sum(),Cost_Total_Low.sum()  


#Init_Vector_High_Selector = np.zeros((No_Sectors,Grid_Points))
#Init_Vector_Low_Selector = np.zeros((No_Sectors,Grid_Points))

#for ii in range(No_Sectors):
#    Init_Vector_High_Selector[ii,:]=Choice_High==ii
#    Init_Vector_Low_Selector[ii,:]=Choice_Low==ii

x=np.array([-0.00103875, -0.01439879]) 

Production_Possibilities = Sector_XP_Levels*Productivity_Params*Experience_Levels+1*(Sector_XP_Levels)
Sector_Prod=Experience_Levels*Productivity_Params
Price_Sector2 = np.exp(x[0])
Price_Foreign= Price_Multiplier_Foreign*np.exp(-0.16)

Price_Sector3 = np.exp(x[1])


Cost_Mfg = (Price_Sector2**(1-eta_mfg)+Price_Foreign**(1-eta_mfg))**(1/(1-eta_mfg))



Price_Sector1=(((((1/((Cost_Mfg/(1-alpha_out))**(1-alpha_out)))**(1/alpha_out))*alpha_out)**(1-eta_ins))-Price_Sector3**(1-eta_ins))**(
        1/(1-eta_ins)) #Cobb Douglas
Cost_Servs = (Price_Sector3**(1-eta_ins)+Price_Sector1**(1-eta_ins))**(1/(1-eta_ins))
Price_Final_Good = (Cost_Servs/alpha_out)**(alpha_out)*(Cost_Mfg/(1-alpha_out))**(1-alpha_out) #Cobb-Douglas
#Price_Sector1

Price=np.tile(np.repeat(np.array((Price_Sector1,Price_Foreign,Price_Sector2,Price_Sector3)),Production_Possibilities),(Age_Periods,1))
Price=Price/Price_Final_Good
Unemp_High=Skill_vals[0]*UnempBenefit/Price_Final_Good
Unemp_Low=UnempBenefit/Price_Final_Good

Prob_Match_Worker = np.array([0.7642352 , 0.63879985, 0.66406452, 0.82123503])#For 0.8
Prob_Match_Firm=np.array([0.52889336, 0.56117599, 0.46798714, 0.46412019])#For 0.8

Prob_Match_Firm=np.tile(Prob_Match_Firm,(Age_Periods-1,1))
Prob_Match_Worker=np.tile(Prob_Match_Worker,(Age_Periods-1,1))
#Val Functions
conv=10.
iterator=0
while conv>0.01 and iterator<20:
    iterator=iterator+1
    Firm_Val_Func_High = np.zeros((Age_Periods,No_Sectors*(Production_Possibilities))) #+1 for unemployment state
    Firm_Val_Func_Low = 0*Firm_Val_Func_High
    Work_Val_Func_High = 0*Firm_Val_Func_High
    Work_Val_Func_Low = 0*Firm_Val_Func_High
    Firm_Continuation_Value_High = 0*Firm_Val_Func_High
    Firm_Continuation_Value_Low = 0*Firm_Val_Func_High
    Worker_Continuation_Value_High= 0*Firm_Val_Func_High
    Worker_Continuation_Value_Low = 0*Firm_Val_Func_High
    Unemp_Continuation_Value_High = 0*Firm_Val_Func_High
    Unemp_Continuation_Value_Low = 0*Firm_Val_Func_High

    Surplus_High = np.zeros((Age_Periods,No_Sectors*(Production_Possibilities)))
    Surplus_Low =  0*Surplus_High
    Surplus_Check_High =0*Surplus_High
    Surplus_Check_Low=0*Surplus_High
    Terminal_Productivities = np.zeros((Production_Possibilities*No_Sectors))
    for ii in range(No_Sectors):
        Temp_Val = np.kron(Sector_XP_Gain,np.kron(Exper,Prod_Vals[ii*Productivity_Params:(ii+1)*Productivity_Params]))
        for jj in range(Sector_XP_Levels):
            Work_Val_Func_High[:,-1-(ii*Production_Possibilities+jj*(Sector_Prod+1))]=Skill_vals[0]*UnempBenefit
            Work_Val_Func_Low[:,-1-(ii*Production_Possibilities+jj*(Sector_Prod+1))]=Skill_vals[1]*UnempBenefit
            Temp_Val=np.insert(Temp_Val,(jj+1)*Sector_Prod+jj,0)
        
        Terminal_Productivities[ii*Production_Possibilities:(ii+1)*Production_Possibilities]=Temp_Val
    
    Choice_High = 0*Firm_Val_Func_High
    Choice_Low = 0*Firm_Val_Func_High
    Wages_High = 0*Firm_Val_Func_High
    Wages_Low=0*Firm_Val_Func_High
    
    Firm_Transition_Matrices_High = np.zeros((Age_Periods,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
    Firm_Transition_Matrices_Low = np.zeros((Age_Periods,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
    
    Worker_Transition_Matrices_High = np.zeros((Age_Periods,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
    Worker_Transition_Matrices_Low = np.zeros((Age_Periods,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
    
    
    Original_W_High_Transition_Matrices = np.zeros((Age_Periods,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
    Original_W_Low_Transition_Matrices = np.zeros((Age_Periods,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
    
    Original_F_High_Transition_Matrices = np.zeros((Age_Periods,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
    Original_F_Low_Transition_Matrices = np.zeros((Age_Periods,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))

     #Code Block Starts
    Terminal_Values_High_Prod = (Skill_vals[0]*Terminal_Productivities)
    Terminal_Values_Low_Prod = Skill_vals[1]*Terminal_Productivities
    
    Terminal_Values_High=Price*Terminal_Values_High_Prod
    Terminal_Values_Low=Price*Terminal_Values_Low_Prod

    
    #for jj in range(No_Sectors):
    #    Terminal_Values_High_Prod=np.concatenate((Terminal_Values_High_Prod[:(jj+1)*(Production_Possibilities-1)+jj],[0],
     #                                                            Terminal_Values_High_Prod[(jj+1)*(Production_Possibilities-1)+jj:]))
     #   Terminal_Values_Low_Prod=np.concatenate((Terminal_Values_Low_Prod[:(jj+1)*(Production_Possibilities-1)+jj],[0],
    #                                                             Terminal_Values_Low_Prod[(jj+1)*(Production_Possibilities-1)+jj:]))
    
    #Terminal_Values_High=Price*Terminal_Values_High_Prod
    #Terminal_Values_Low=Price*Terminal_Values_Low_Prod
    
    
    for ii in range(Age_Periods):     
        index_current = Age_Periods-ii-1
        index_before = index_current-1
        Max_Unemp_Option_Sector1_High = np.maximum.reduce([Unemp_High+Unemp_Continuation_Value_High[index_current,0:Production_Possibilities],
                np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities),
                np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
                np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)])
     
        Choice_TempSector1_High=np.argmax((Unemp_High+Unemp_Continuation_Value_High[index_current,0:Production_Possibilities],
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities),
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)),
            axis=0)    
                
        Max_Unemp_Option_Sector2_High =np.maximum.reduce(
            [np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1],Production_Possibilities),
            Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities:2*Production_Possibilities],
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities-1]-SC_Mfg*Switch_Cost1,Production_Possibilities),
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)])
                                                                   
        Choice_TempSector2_High=np.argmax(
            (np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1],Production_Possibilities),
             Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities:2*Production_Possibilities]+0.00000000001,
             np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities-1]-SC_Mfg*Switch_Cost1,Production_Possibilities),
             np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)),
             axis=0)                                                              
                                                                               
        Max_Unemp_Option_Sector2_High_B =np.maximum.reduce(
            [np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1],Production_Possibilities),
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities), 
            Unemp_High+Unemp_Continuation_Value_High[index_current,2*Production_Possibilities:3*Production_Possibilities],
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)])
                                                                   
        Choice_TempSector2_High_B=np.argmax(
            (np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1],Production_Possibilities),
             np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities),
             Unemp_High+Unemp_Continuation_Value_High[index_current,2*Production_Possibilities:3*Production_Possibilities]+0.00000000001,                 
             np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)),
             axis=0)                
                                                                                                                    
        Max_Unemp_Option_Sector3_High =np.maximum.reduce(
            [np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1],Production_Possibilities),
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities),
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
            Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities:4*Production_Possibilities]])


        Choice_TempSector3_High=np.argmax(
            (np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1],Production_Possibilities),
             np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities),
             np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
             Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities:4*Production_Possibilities]+0.00000000001),axis=0)
                                                                                             

        Choice_High[index_current,:] = np.concatenate((Choice_TempSector1_High,Choice_TempSector2_High,Choice_TempSector2_High_B,Choice_TempSector3_High))                                                                          
        Max_Unemp_Option_High = np.concatenate((Max_Unemp_Option_Sector1_High,Max_Unemp_Option_Sector2_High,Max_Unemp_Option_Sector2_High_B,Max_Unemp_Option_Sector3_High))
     
        Max_Unemp_Option_Sector1_Low = np.maximum.reduce([Unemp_Low+Unemp_Continuation_Value_Low[index_current,0:Production_Possibilities],
                np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities),
                np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
                np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)])
     
        Choice_TempSector1_Low=np.argmax((Unemp_Low+Unemp_Continuation_Value_Low[index_current,0:Production_Possibilities],
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities),
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)),
            axis=0)    
                
        Max_Unemp_Option_Sector2_Low =np.maximum.reduce(
            [np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1],Production_Possibilities),
            Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities:2*Production_Possibilities],
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities-1]-SC_Mfg*Switch_Cost1,Production_Possibilities),
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)])
                                                                   
        Choice_TempSector2_Low=np.argmax(
            (np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1],Production_Possibilities),
             Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities:2*Production_Possibilities]+0.00000000001,
             np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities-1]-SC_Mfg*Switch_Cost1,Production_Possibilities),
             np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)),
             axis=0)                                                              
                                                                               
        Max_Unemp_Option_Sector2_Low_B =np.maximum.reduce(
            [np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1],Production_Possibilities),
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities), 
            Unemp_Low+Unemp_Continuation_Value_Low[index_current,2*Production_Possibilities:3*Production_Possibilities],
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)])
                                                                   
        Choice_TempSector2_Low_B=np.argmax(
            (np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1],Production_Possibilities),
             np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities),
             Unemp_Low+Unemp_Continuation_Value_Low[index_current,2*Production_Possibilities:3*Production_Possibilities]+0.00000000001,                 
             np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)),
             axis=0)                
                                                                                                                    
        Max_Unemp_Option_Sector3_Low =np.maximum.reduce(
            [np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1],Production_Possibilities),
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities),
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
            Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities:4*Production_Possibilities]])


        Choice_TempSector3_Low=np.argmax(
            (np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1],Production_Possibilities),
             np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities),
             np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
             Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities:4*Production_Possibilities]+0.00000000001),axis=0)
                                                                  
        Choice_Low[index_current,:] = np.concatenate((Choice_TempSector1_Low,Choice_TempSector2_Low,Choice_TempSector2_Low_B,Choice_TempSector3_Low))                                                                          
        Max_Unemp_Option_Low = np.concatenate((Max_Unemp_Option_Sector1_Low,Max_Unemp_Option_Sector2_Low,Max_Unemp_Option_Sector2_Low_B,Max_Unemp_Option_Sector3_Low))
        
        Surplus_High[index_current,:] = Terminal_Values_High[index_current,:] + Firm_Continuation_Value_High[index_current,:] + Worker_Continuation_Value_High[index_current,:] -Max_Unemp_Option_High
        Surplus_Low[index_current,:] = Terminal_Values_Low[index_current,:] + Firm_Continuation_Value_Low[index_current,:] + Worker_Continuation_Value_Low[index_current,:] -Max_Unemp_Option_Low
        
        Surplus_Check_High[index_current,:]=np.maximum(0,100000*Surplus_High[index_current,:])
        Surplus_Check_Low[index_current,:]=np.maximum(0,100000*Surplus_Low[index_current,:])
        Surplus_Check_High[index_current,:]=np.minimum(1,Surplus_Check_High[index_current,:])
        Surplus_Check_Low[index_current,:]=np.minimum(1,Surplus_Check_Low[index_current,:])       
                
        for jj in range(Sector_XP_Levels*No_Sectors):
            Surplus_Check_High[index_current,(jj+1)*(Sector_Prod+1)-1]=1
            Surplus_Check_Low[index_current,(jj+1)*(Sector_Prod+1)-1]=1
        
        Wages_High[index_current,:]=Surplus_Check_High[index_current,:]*((Max_Unemp_Option_High-Worker_Continuation_Value_High[index_current,:])+Fraction_Worker_Surplus*Surplus_High[index_current,:])
        Wages_Low[index_current,:]=Surplus_Check_Low[index_current,:]*((Max_Unemp_Option_Low-Worker_Continuation_Value_Low[index_current,:])+Fraction_Worker_Surplus*Surplus_Low[index_current,:])
            
        for jj in range(Sector_XP_Levels*No_Sectors):
            Wages_High[index_current,(jj+1)*(Sector_Prod+1)-1] = Unemp_High
            Wages_Low[index_current,(jj+1)*(Sector_Prod+1)-1] = Unemp_Low
                
        
        
    
        Firm_Val_Func_High[index_current,:] = Surplus_Check_High[index_current,:]*(Terminal_Values_High[index_current,:]-Wages_High[index_current,:]+(
                Firm_Continuation_Value_High[index_current,:]))
        Firm_Val_Func_Low[index_current,:] = Surplus_Check_Low[index_current,:]*(Terminal_Values_Low[index_current,:]-Wages_Low[index_current,:]+(
                Firm_Continuation_Value_Low[index_current,:]))
 ############ ##Non-Pecuniary
        Wages_High[index_current,:Production_Possibilities]=Wages_High[index_current,:Production_Possibilities]+Non_Pec
        Wages_Low[index_current,:Production_Possibilities]=Wages_Low[index_current,:Production_Possibilities]+Non_Pec
            
        for jj in range(Sector_XP_Levels*No_Sectors):
            Wages_High[index_current,(jj+1)*(Sector_Prod+1)-1] = Unemp_High
            Wages_Low[index_current,(jj+1)*(Sector_Prod+1)-1] = Unemp_Low
########################               
        Work_Val_Func_High[index_current,:] = Wages_High[index_current,:]+(
                Worker_Continuation_Value_High[index_current,:])
        Work_Val_Func_Low[index_current,:] = Wages_Low[index_current,:]+(
                Worker_Continuation_Value_Low[index_current,:])
        
        for jj in range(Sector_XP_Levels*No_Sectors):
            Firm_Val_Func_High[index_current,(jj+1)*(Sector_Prod+1)-1]=0.  #Enforcing free-entry
            Firm_Val_Func_Low[index_current,(jj+1)*(Sector_Prod+1)-1] =0  #Enforcing free-entry  
            
        #Transitions
        #Tranistion Matrix for Job Values
        
        Job_Transition_Matrix=np.zeros((Firm_Val_Func_Low.shape[1]-Sector_XP_Levels*No_Sectors,Firm_Val_Func_Low.shape[1]-Sector_XP_Levels*No_Sectors))
        Job_Transition_Matrix_Worker = np.zeros((Firm_Val_Func_Low.shape[1]-Sector_XP_Levels*No_Sectors,Firm_Val_Func_Low.shape[1]-Sector_XP_Levels*No_Sectors))
    
        
        
        for jj in range(No_Sectors):
            Initial_Val = np.kron(Sector_XP_Gain_Matrix,np.kron(Exper_Gain_Matrix,Prod_Change_Matrix[jj])) #Control Different Matrix types for each sector
            index_start = (jj)*(Production_Possibilities-Sector_XP_Levels)
            index_end = (jj+1)*(Production_Possibilities-Sector_XP_Levels)            
            Job_Transition_Matrix[index_start:index_end,
                                  index_start:index_end]=Initial_Val
            Job_Transition_Matrix_Worker[index_start:index_end,
                                 index_start:index_end]=Initial_Val
        
        #Insert Unemployment States for each sector productvity
        Col = np.zeros((Job_Transition_Matrix.shape[0]))
        Col=Col.reshape(Job_Transition_Matrix.shape[0],1)
        for jj in (range(No_Sectors*Sector_XP_Levels)):
            Job_Transition_Matrix = np.hstack((Job_Transition_Matrix[:,:((jj+1)*Sector_Prod)+jj], Col,
                                                                     Job_Transition_Matrix[:,((jj+1)*Sector_Prod)+jj:]))
            Job_Transition_Matrix_Worker = np.hstack((Job_Transition_Matrix_Worker[:,:((jj+1)*Sector_Prod)+jj], Col,
                                                                     Job_Transition_Matrix_Worker[:,((jj+1)*Sector_Prod)+jj:]))
        
        Row = np.zeros((Job_Transition_Matrix.shape[1])) 
        for jj in (range(No_Sectors*Sector_XP_Levels)):
            Job_Transition_Matrix = np.vstack((Job_Transition_Matrix[:((jj+1)*Sector_Prod)+jj,:], Row,
                                                                 Job_Transition_Matrix[((jj+1)*Sector_Prod)+jj:,:]))
            Job_Transition_Matrix_Worker = np.vstack((Job_Transition_Matrix_Worker[:((jj+1)*Sector_Prod)+jj,:], Row,
                                                                 Job_Transition_Matrix_Worker[((jj+1)*Sector_Prod)+jj:,:]))
        
        for jj in range(No_Sectors):
            for ss in range(Sector_XP_Levels):
                index_start = (jj*Production_Possibilities)+(ss)*(Sector_Prod+1)
                index_end = (jj*Production_Possibilities)+(ss+1)*(Sector_Prod+1)         
                for kk in range(Productivity_Params):
                    Job_Transition_Matrix[index_end-1,index_start+(Experience_Levels-1)*Productivity_Params+kk] = Prod_Draw[kk]*Prob_Match_Firm[index_before,jj]
                    Job_Transition_Matrix[index_end-1,index_end-1] = 1-Prob_Match_Firm[index_before,jj] #Last state from unemployment
                    Job_Transition_Matrix_Worker[index_end-1,index_start+(Experience_Levels-1)*Productivity_Params+kk] = Prod_Draw[kk]*Prob_Match_Worker[index_before,jj]
                    Job_Transition_Matrix_Worker[index_end-1,index_end-1] = 1-Prob_Match_Worker[index_before,jj]

                
        Job_Transition_Matrix_Adjusted_High=np.zeros((Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
        Job_Transition_Matrix_Adjusted_Low=np.zeros((Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
        
        Job_Transition_Matrix_Worker_Adjusted_High=np.zeros((Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
        Job_Transition_Matrix_Worker_Adjusted_Low=np.zeros((Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
        
        Job_Transition_Matrix_Adjusted_High=Surplus_Check_High[index_current,:]*Job_Transition_Matrix
        Job_Transition_Matrix_Adjusted_Low=Surplus_Check_Low[index_current,:]*Job_Transition_Matrix 
        
        Job_Transition_Matrix_Worker_Adjusted_High=Surplus_Check_High[index_current,:]*Job_Transition_Matrix_Worker
        Job_Transition_Matrix_Worker_Adjusted_Low=Surplus_Check_Low[index_current,:]*Job_Transition_Matrix_Worker
        
        
        for jj in range(No_Sectors):
            for ss in range(Sector_XP_Levels):
                index_start = (jj*Production_Possibilities)+(ss)*(Sector_Prod+1)
                index_end =  (jj*Production_Possibilities)+(ss+1)*(Sector_Prod+1)   
                Job_Transition_Matrix_Adjusted_High[index_start:index_end,index_end-1]=Job_Transition_Matrix_Adjusted_High[
                                    index_start:index_end,index_end-1]+((1-np.tile(Surplus_Check_High[index_current],
                                    (No_Sectors*Production_Possibilities,1)))*Job_Transition_Matrix).sum(1)[index_start:index_end] #Add all the unemployment outcomes
              
                Job_Transition_Matrix_Adjusted_Low[index_start:index_end,index_end-1]=Job_Transition_Matrix_Adjusted_Low[
                                    index_start:index_end,index_end-1]+((1-np.tile(Surplus_Check_Low[index_current],
                                    (No_Sectors*Production_Possibilities,1)))*Job_Transition_Matrix).sum(1)[index_start:index_end]
            
                TemporHigh=Job_Transition_Matrix_Worker_Adjusted_High[
                                    index_start:index_end,index_end-1]+((1-np.tile(Surplus_Check_High[index_current],
                                    (No_Sectors*Production_Possibilities,1)))*Job_Transition_Matrix_Worker
                                    ).sum(1)[index_start:index_end] 
            
                Job_Transition_Matrix_Worker_Adjusted_High[index_start:index_end,index_end-1]=0
                Sector_Choice=int(Choice_High[index_current,index_end-1])
                if Sector_Choice==jj:
                    Job_Transition_Matrix_Worker_Adjusted_High[index_start:index_end,(Sector_Choice*Sector_XP_Levels+1+
                                                                ss)*(Sector_Prod+1)-1]=TemporHigh
                else:
                    Job_Transition_Matrix_Worker_Adjusted_High[index_start:index_end,(Sector_Choice+1
                                                                )*Production_Possibilities-1]=TemporHigh
    
    #Making sure the unemployed switch to unemployment in other sector        
    #                Job_Transition_Matrix_Worker_Adjusted_High[index_end-1,:]=Sector_Checker[Sector_Choice]*(
    #                                   Job_Transition_Matrix_Worker_Adjusted_High[index_end-1,:])
    #                Job_Transition_Matrix_Worker_Adjusted_High[index_end-1,(Sector_Choice+1
    #                                )*Production_Possibilities-1]=(1-Job_Transition_Matrix_Worker_Adjusted_High[index_end-1,:].sum()+
    #                                Job_Transition_Matrix_Worker_Adjusted_High[index_end-1,(Sector_Choice+1
    #                                )*Production_Possibilities-1])
      
    
    
            
                TemporLow=Job_Transition_Matrix_Worker_Adjusted_Low[
                        index_start:index_end,index_end-1]+((1-np.tile(Surplus_Check_Low[index_current],
                        (No_Sectors*Production_Possibilities,1)))*Job_Transition_Matrix_Worker
                        ).sum(1)[index_start:index_end]
                
                Job_Transition_Matrix_Worker_Adjusted_Low[index_start:index_end,index_end-1]=0
                Sector_Choice_Low=int(Choice_Low[index_current,index_end-1])
                
                if Sector_Choice_Low==jj:
                    Job_Transition_Matrix_Worker_Adjusted_Low[index_start:index_end,(Sector_Choice*Sector_XP_Levels+1+
                                                            ss)*(Sector_Prod+1)-1]=TemporLow
                else:   
                    Job_Transition_Matrix_Worker_Adjusted_Low[index_start:index_end,(Sector_Choice_Low+1
                                                            )*Production_Possibilities-1]=TemporLow

        
                
        Original_W_High_Transition_Matrices[index_before,:,:] = Job_Transition_Matrix_Worker_Adjusted_High
        Original_W_Low_Transition_Matrices[index_before,:,:] = Job_Transition_Matrix_Worker_Adjusted_Low
        
        
        Original_F_High_Transition_Matrices[index_before,:,:] = Job_Transition_Matrix_Adjusted_High
        Original_F_Low_Transition_Matrices[index_before,:,:] = Job_Transition_Matrix_Adjusted_Low
        
        Job_Transition_Matrix_Worker_Adjusted_High1 = np.linalg.matrix_power(Job_Transition_Matrix_Worker_Adjusted_High,4)
        Job_Transition_Matrix_Worker_Adjusted_Low1 = np.linalg.matrix_power(Job_Transition_Matrix_Worker_Adjusted_Low,4)
        

        Job_Transition_Matrix_Adjusted_High1 = np.linalg.matrix_power(Job_Transition_Matrix_Adjusted_High,4)
        Job_Transition_Matrix_Adjusted_Low1 = np.linalg.matrix_power(Job_Transition_Matrix_Adjusted_Low,4)
        
        Firm_Transition_Matrices_High[index_before,:,:]=Job_Transition_Matrix_Adjusted_High1
        Firm_Transition_Matrices_Low[index_before,:,:]=Job_Transition_Matrix_Adjusted_Low1
        
        
        Worker_Transition_Matrices_High[index_before,:,:]=Job_Transition_Matrix_Worker_Adjusted_High1
        Worker_Transition_Matrices_Low[index_before,:,:]=Job_Transition_Matrix_Worker_Adjusted_Low1           
        
                     
        Worker_Continuation_Value_High[index_before,:]=Beta*(np.dot(Job_Transition_Matrix_Worker_Adjusted_High,
                            Work_Val_Func_High[index_current,:])-np.dot(
                                    Switcher_Matrix*Job_Transition_Matrix_Worker_Adjusted_High,Switch_Vector))
        Worker_Continuation_Value_Low[index_before,:]=Beta*(np.dot(Job_Transition_Matrix_Worker_Adjusted_Low,
                           Work_Val_Func_Low[index_current,:])-np.dot(
                                   Switcher_Matrix*Job_Transition_Matrix_Worker_Adjusted_Low,Switch_Vector))
        
        for jj in range(No_Sectors):
            for ss in range(Sector_XP_Levels):
                Unemp_Continuation_Value_High[index_before,jj*Production_Possibilities+ss*(Sector_Prod+1):jj*Production_Possibilities+(ss+1)*(Sector_Prod+1)
                    ]=np.repeat(Worker_Continuation_Value_High[index_before,(jj)*Production_Possibilities+(ss+1)*(Sector_Prod+1)-1],(Sector_Prod+1))
                Unemp_Continuation_Value_Low[index_before,jj*Production_Possibilities+ss*(Sector_Prod+1):jj*Production_Possibilities+(ss+1)*(Sector_Prod+1)
                    ]=np.repeat(Worker_Continuation_Value_Low[index_before,(jj)*Production_Possibilities+(ss+1)*(Sector_Prod+1)-1],(Sector_Prod+1))        
        
        Firm_Continuation_Value_High[index_before,:]=Beta*(
                np.dot(Job_Transition_Matrix_Adjusted_High,Firm_Val_Func_High[index_current,:]))
        Firm_Continuation_Value_Low[index_before,:]=Beta*(
                np.dot(Job_Transition_Matrix_Adjusted_Low,Firm_Val_Func_Low[index_current,:]))      
    
    Disribution_Wokers_Low = np.zeros((Age_Periods,Firm_Val_Func_Low.shape[1]))
    Disribution_Wokers_High= np.zeros((Age_Periods,Firm_Val_Func_Low.shape[1]))   
    Init_High,Init_Low,Ch,Cl = Initial_Cohort_Fixed(Work_Val_Func_High,Work_Val_Func_Low)
    Disribution_Wokers_High[0]= Init_High
    Disribution_Wokers_Low[0]=Init_Low
            
    Firm_Match_Expectation_High = np.zeros((Age_Periods-1,No_Sectors,Sector_XP_Levels)) #Since first period workers are all searchers
    Firm_Match_Expectation_Low = np.zeros((Age_Periods-1,No_Sectors,Sector_XP_Levels))
        #Diving by original Prob of matching to get expected value conditional on matching. 
        #The last element does not matter since mutliplied by zero as V (value of vacancy has to be zero)
    for ii in range(1,Age_Periods):
        Disribution_Wokers_Low[ii,:]= np.dot(Disribution_Wokers_Low[ii-1,:],Worker_Transition_Matrices_Low[ii-1,:,:]) 
        Disribution_Wokers_High[ii,:]= np.dot(Disribution_Wokers_High[ii-1,:],Worker_Transition_Matrices_High[ii-1,:,:])
        for jj in range(No_Sectors):
            for ss in range(Sector_XP_Levels):
                index_start=(jj*Sector_XP_Levels+ss)*(Sector_Prod+1)
                index_end=((jj)*Sector_XP_Levels+ss+1)*(Sector_Prod+1)
                Firm_Match_Expectation_High[ii-1,jj,ss] = np.dot(
                        Original_F_High_Transition_Matrices[ii-1,index_end-1,index_start:index_end-1]/((Prob_Match_Firm[ii-1,jj])),
                        Firm_Val_Func_High[ii,index_start:index_end-1])
                Firm_Match_Expectation_Low[ii-1,jj,ss] = np.dot(
                        Original_F_Low_Transition_Matrices[ii-1,index_end-1,index_start:index_end-1]/((Prob_Match_Firm[ii-1,jj])),
                        Firm_Val_Func_Low[ii,index_start:index_end-1])
    
    Total_Uh = np.zeros((No_Sectors,Sector_XP_Levels))
    Total_Ul = 0*Total_Uh
    for jj in range(No_Sectors):
        for ss in range(Sector_XP_Levels):
            Total_Uh[jj,ss] = np.sum(Disribution_Wokers_High[:-1,(jj*Sector_XP_Levels+ss+1)*(Sector_Prod+1)-1]) #Ignore last year since they can't find a job anymore
            Total_Ul[jj,ss] = np.sum(Disribution_Wokers_Low[:-1,(jj*Sector_XP_Levels+ss+1)*(Sector_Prod+1)-1])
    Total_U = Total_Uh.sum(1)+Total_Ul.sum(1)
    
    #Prod_Draw_Vector = np.zeros((Firm_Val_Func_High_Next.shape[1]))
    #for ii in range(Productivity_Params):
    #        Prod_Draw_Vector[Experience_Levels*ii+1] = Prod_Draw[ii]
    Expected_Match_Value_Firm= np.zeros((No_Sectors))
    for jj in range(No_Sectors):
        for ss in range(Sector_XP_Levels):
            Expected_Match_Value_Firm[jj] = np.dot(Firm_Match_Expectation_Low[:,jj,ss],
                                     Disribution_Wokers_Low[:-1,(jj*Sector_XP_Levels+ss+1)*(Sector_Prod+1)-1]/Total_U[jj])+np.dot(
                Firm_Match_Expectation_High[:,jj,ss],
                Disribution_Wokers_High[:-1,(jj*Sector_XP_Levels+ss+1)*(Sector_Prod+1)-1]/Total_U[jj])+Expected_Match_Value_Firm[jj]

    Expected_Match_Value_Firm=np.maximum(np.nan_to_num(Expected_Match_Value_Firm),0.01*np.ones((4)))
    dampen=0.7
    q_match = np.minimum(1,Vacancy_Cost/Expected_Match_Value_Firm)
    Total_U_Temp=np.maximum(Total_U, 0.000001*np.ones((No_Sectors)))
    Vacancy_Amount = Total_U_Temp/((q_match/Scale)**(1/sigma))
    Prob_Match_Firm_Prev=Prob_Match_Firm[0,:]
    Prob_Match_Worker_Prev=Prob_Match_Worker[0,:]   
    Prob_Match_Firm =(1-dampen)*np.minimum(1., Scale*(Total_U_Temp/Vacancy_Amount)**sigma)+dampen*Prob_Match_Firm_Prev
    Prob_Match_Worker = (1-dampen)*np.minimum(1.,Scale*(Vacancy_Amount/Total_U_Temp)**(1-sigma))+dampen*Prob_Match_Worker_Prev
    conv=np.max([np.abs(Prob_Match_Firm_Prev-np.minimum(1., Scale*(Total_U_Temp/Vacancy_Amount)**sigma)),np.abs(Prob_Match_Worker_Prev-np.minimum(1.,Scale*(Vacancy_Amount/Total_U_Temp)**(1-sigma)))])
    Prob_Match_Firm=np.tile(Prob_Match_Firm,(Age_Periods-1,1))
    Prob_Match_Worker=np.tile(Prob_Match_Worker,(Age_Periods-1,1))
    print(conv)   

Total_L=(Terminal_Values_High_Prod[0*Production_Possibilities:1*Production_Possibilities]*Disribution_Wokers_High[:,
        0*Production_Possibilities:1*Production_Possibilities]).sum()+((
        Terminal_Values_Low_Prod[0*Production_Possibilities:1*Production_Possibilities]*Disribution_Wokers_Low[:,
        0*Production_Possibilities:1*Production_Possibilities]).sum())
Total_M=(Terminal_Values_High_Prod[1*Production_Possibilities:2*Production_Possibilities]*Disribution_Wokers_High[:,
        1*Production_Possibilities:2*Production_Possibilities]).sum()+((
            Terminal_Values_Low_Prod[1*Production_Possibilities:2*Production_Possibilities]*Disribution_Wokers_Low[:,
        1*Production_Possibilities:2*Production_Possibilities]).sum())

Total_M_B=(Terminal_Values_High_Prod[2*Production_Possibilities:3*Production_Possibilities]*Disribution_Wokers_High[:,
        2*Production_Possibilities:3*Production_Possibilities]).sum()+((
            Terminal_Values_Low_Prod[2*Production_Possibilities:3*Production_Possibilities]*Disribution_Wokers_Low[:,
        2*Production_Possibilities:3*Production_Possibilities]).sum())

Total_H=(Terminal_Values_High_Prod[3*Production_Possibilities:4*Production_Possibilities]*Disribution_Wokers_High[:,
        3*Production_Possibilities:4*Production_Possibilities]).sum()+((
            Terminal_Values_Low_Prod[3*Production_Possibilities:4*Production_Possibilities]*Disribution_Wokers_Low[:,
        3*Production_Possibilities:4*Production_Possibilities]).sum())

Total_Earnings = (Wages_High*Disribution_Wokers_High).sum()+(Wages_Low*Disribution_Wokers_Low).sum()
Total_Firm_Earnings = ((Terminal_Values_High-Wages_High)*Disribution_Wokers_High).sum()+((Terminal_Values_Low-
                      Wages_Low)*Disribution_Wokers_Low).sum()
                      
Wage_Guesses=np.array([-0.01466646,-0.01457459,-0.01458011,-0.01448786,-0.01440165,-0.01430304
,-0.01430682,-0.01421345,-0.01412034,-0.01412515,-0.01400432,-0.01391098
,-0.01372019,-0.01372445,-0.01363116,-0.01353787,-0.01344454,-0.01344835
,-0.01335412,-0.01325999,-0.01315006,-0.01305538,-0.01296141,-0.01286673
,-0.0128696,-0.01267737,-0.01257517,-0.01247505,-0.01237452,-0.01227358
,-0.0118913,-0.0118859,-0.01158746,-0.01148331,-0.01147595,-0.01137025
,-0.01116637,-0.01105867,-0.01104516,-0.01083185,-0.01071533,-0.01069604
,-0.01048185,-0.01036941,-0.01025703,-0.00186403,-0.00154371,-0.00122543
,-0.00090823,-0.00061841,-0.00040234,-0.00019761,0.00010545,0.00031104
,0.00061514,0.0010014,0.0012032,0.00140438,0.00170234,0.00199955
,0.00219875,0.00249459,0.00288585,0.0030808,0.0034704,0.003753
,0.0041403,0.00462862,0.00491994,0.00511226,0.00530385,0.00559015
,0.00587727,0.00616385,0.00664518,0.00668861,0.00677615,0.00686205
,0.00675119,0.00673536,0.00661868,0.00659533,0.00656623,0.00662757
,0.00658454,0.00624668,0.00581327,0.00548031,0.00527148,0.00506205]) 
       
        
#Wage_Guesses=np.round(Wage_Guesses,4)

Price_Sector2_Temp=np.concatenate((np.repeat(1.24*np.exp(-0.16),10),np.repeat(1.23*np.exp(-0.16),10),np.repeat(1.225*np.exp(-0.16),10),np.repeat(1.20*np.exp(-0.16),15)))            

Price_Transition_Period = 0
Price_Multiplier_Foreign_New=1.20

def SolverFuncFixed(Wage_Guesses):
    print(Wage_Guesses)
    #Solving Code
     #append final steady state wage
     
    Production_Possibilities = Sector_XP_Levels*Productivity_Params*Experience_Levels+1*(Sector_XP_Levels)
    Sector_Prod=Experience_Levels*Productivity_Params
    Price_Sector2 = np.exp(np.append(Wage_Guesses[Transition_Periods:],Final_Vector_Guess[0]))
    #Price_Foreign= np.concatenate((np.repeat(1.24*np.exp(-0.16),10),np.repeat(1.23*np.exp(-0.16),10),np.repeat(1.225*np.exp(-0.16),10),np.repeat(1.20*np.exp(-0.16),16)))#np.repeat(Price_Multiplier_Foreign_New*np.exp(-0.16),Price_Sector2.shape[0])
    Price_Foreign=np.repeat(Price_Multiplier_Foreign_New*np.exp(-0.16),Price_Sector2.shape[0])
    
    Price_Sector3 = np.exp(np.append(Wage_Guesses[:Transition_Periods],Final_Vector_Guess[1]))


    Cost_Mfg = (Price_Sector2**(1-eta_mfg)+Price_Foreign**(1-eta_mfg))**(1/(1-eta_mfg))
    
    
    Price_Sector1=(((((1/((Cost_Mfg/(1-alpha_out))**(1-alpha_out)))**(1/alpha_out))*alpha_out)**(1-eta_ins))-Price_Sector3**(1-eta_ins))**(
            1/(1-eta_ins)) #Cobb Douglas
    Cost_Servs = (Price_Sector3**(1-eta_ins)+Price_Sector1**(1-eta_ins))**(1/(1-eta_ins))
    Price_Final_Good = (Cost_Servs/alpha_out)**(alpha_out)*(Cost_Mfg/(1-alpha_out))**(1-alpha_out) #Cobb-Douglas
    #Price_Sector1
    
    Unemp_High=Skill_vals[0]*UnempBenefit
    Unemp_Low=UnempBenefit

    
    
    Stage_Transition = (Age_Periods-1)+Age_Periods+Transition_Periods
    Transition_Prices = np.concatenate((Price_Sector1,Price_Foreign,Price_Sector2,Price_Sector3),axis=0)
    Transition_Prices=np.reshape(Transition_Prices,(int(Transition_Prices.shape[0]/No_Sectors),int(No_Sectors)),order='F')
    Transition_Prices=np.concatenate((np.tile(Transition_Prices[0,:],(Age_Periods-1,1)),
                                      Transition_Prices,np.tile(Transition_Prices[-1,:],(Stage_Transition-Transition_Periods,1))))
        
    
    Transition_Prices=np.repeat(Transition_Prices,Production_Possibilities,axis=1)    
    
    Needed_Distributions = Age_Periods+Transition_Periods
    #Prob_Match_Worker = np.array([0.47972544, 0.39670502, 0.50540898]) #For 0.85
    #Prob_Match_Firm=np.array([0.26129, 0.64358, 0.22943])  #For 0.85
    
    Prob_Match_Worker = np.array([0.87232614, 0.68876657, 0.68062051, 0.80133969])#For 0.8
    Prob_Match_Firm=np.array([0.39787407, 0.42441032, 0.43427489, 0.49451436])#For 0.8
    
    
    
    Prob_Match_Worker_Transition = np.tile(Prob_Match_Worker,(Size_of_Probs,1))
    Prob_Match_Firm_Transition = np.tile(Prob_Match_Firm,(Size_of_Probs,1))
    
    Prob_Match_Worker_Beg=np.array([0.37381631, 0.47937891, 0.47101612])
    Prob_Match_Firm_Beg=np.array([0.4780786 , 0.38350111, 0.26378343])
    
    #Prob_Match_Worker_Transition = np.tile(Prob_Match_Worker_Beg,(Size_of_Probs,1))
    #Prob_Match_Firm_Transition = np.tile(Prob_Match_Firm_Beg,(Size_of_Probs,1))
    
    
    #Prob_Match_Worker = np.array([0.37385819, 0.47939761, 0.4710314 ])
    #Prob_Match_Firm=np.array([0.47816515, 0.38357854, 0.26387577])
    #Prob_Match_Firm=np.array([0.06162904, 0.15232613, 0.1620824 ])

    
    #Transition linspacing code
    #Prob_Match_Worker_Transition[Age_Periods-1-1:Age_Periods-1-1+Needed_Distributions,:] =np.linspace(Prob_Match_Worker_Beg,Prob_Match_Worker,Needed_Distributions)
    #Prob_Match_Firm_Transition[Age_Periods-1-1:Age_Periods-1-1+Needed_Distributions,:] = np.linspace(Prob_Match_Firm_Beg,Prob_Match_Firm,Needed_Distributions)
    
    
    #################################
    
    
    
    Worker_Transition_Matrices_High_Transition = [[0]*1]*Stage_Transition
    Worker_Transition_Matrices_Low_Transition = [[0]*1]*Stage_Transition
    #Orig_Worker_Transition_Matrices_High_Transition = np.zeros((Stage_Transition,Worker_Transition_Matrices_High.shape[0],
    #                            Worker_Transition_Matrices_High.shape[1],Worker_Transition_Matrices_High.shape[2]))
    #Orig_Worker_Transition_Matrices_Low_Transition = np.zeros((Stage_Transition,Worker_Transition_Matrices_High.shape[0],
    #                            Worker_Transition_Matrices_High.shape[1],Worker_Transition_Matrices_High.shape[2]))
    #Firm_Transition_Matrices_High_Transition = 0*Worker_Transition_Matrices_Low_Transition
    #Firm_Transition_Matrices_Low_Transition = 0*Worker_Transition_Matrices_Low_Transition
    Orig_Firm_Transition_Matrices_High_Transition = [[0]*1]*Stage_Transition
    Orig_Firm_Transition_Matrices_Low_Transition = [[0]*1]*Stage_Transition
    Wages_High_Transition = np.zeros((Stage_Transition,Age_Periods,Production_Possibilities*No_Sectors))
    Wages_Low_Transition = 0*Wages_High_Transition
    Surplus_High_Transition = np.zeros((Stage_Transition,Age_Periods,Production_Possibilities*No_Sectors))
    Surplus_Low_Transition = 0*Surplus_High_Transition
    Firm_Val_Func_High_Transition = 0*Surplus_High_Transition
    Firm_Val_Func_Low_Transition =0*Surplus_High_Transition
    Work_Val_Func_High_Transition = np.zeros((Stage_Transition,Age_Periods,Production_Possibilities*No_Sectors))
    Work_Val_Func_Low_Transition = np.zeros((Stage_Transition,Age_Periods,Production_Possibilities*No_Sectors))
    Price_Vector = np.zeros((Stage_Transition,Age_Periods,Production_Possibilities*No_Sectors))
    # Computing all 
    Init_High = np.zeros((Stage_Transition,Production_Possibilities*No_Sectors))
    Init_Low = np.zeros((Stage_Transition,Production_Possibilities*No_Sectors))
    
    arg_instances=list(range(Stage_Transition))
    convtotal=10.
    iterator=0
    while convtotal>0.02 and iterator<10:
    #while iterator < 1:
        iterator=iterator+1
        if __name__ == "__main__":
                number_processes = 3
                prod_x=partial(TransitionFunc.TransitionValFunc_FullCSR,Transition_Prices=Transition_Prices,
                               Prob_Match_Worker_Transition=Prob_Match_Worker_Transition,
                               Prob_Match_Firm_Transition=Prob_Match_Firm_Transition,
                         Age_Periods=Age_Periods,No_Sectors=No_Sectors,Production_Possibilities=Production_Possibilities,
                         Unemp_High=Unemp_High,Unemp_Low=Unemp_Low,Skill_vals=Skill_vals,
                         Prod_Vals=Prod_Vals,Exper=Exper,Swith_Cost0=Swith_Cost0,Switch_Cost=Switch_Cost,Switch_Cost1=Switch_Cost1,Switch_Cost2=Switch_Cost2,
                         Prod_Change_Matrix=Prod_Change_Matrix,Exper_Gain_Matrix=Exper_Gain_Matrix,
                         Sector_XP_Gain=Sector_XP_Gain,Sector_XP_Gain_Matrix=Sector_XP_Gain_Matrix,
                         Productivity_Params=Productivity_Params,Experience_Levels=Experience_Levels,
                         Sector_XP_Levels=Sector_XP_Levels,Prod_Draw=Prod_Draw,
                         Beta=Beta,Switcher_Matrix=Switcher_Matrix,Switch_Vector=Switch_Vector,
                         Fraction_Worker_Surplus=Fraction_Worker_Surplus,Cost_Vector=Cost_Vector,Distr_Init_High=Distr_Init_High,
                         Distr_Init_Low=Distr_Init_Low,Prob_Types=Prob_Types,Prob_Draw=Prob_Draw,Grid_Points=Grid_Points,Non_Pec=Non_Pec,SC_Mfg=SC_Mfg)
                pool = multiprocessing.Pool(number_processes)

                results =[]
                results = pool.map_async(prod_x, arg_instances)
                ABC = results.get()
                pool.close()
                pool.join()
        for ss in range(len(arg_instances)):
            (Worker_Transition_Matrices_High_Transition[Stage_Transition-ss-1],Worker_Transition_Matrices_Low_Transition[Stage_Transition-ss-1],
             Orig_Firm_Transition_Matrices_High_Transition[Stage_Transition-ss-1],Orig_Firm_Transition_Matrices_Low_Transition[Stage_Transition-ss-1],
             Wages_High_Transition[Stage_Transition-ss-1,:],Wages_Low_Transition[Stage_Transition-ss-1,:],Surplus_High_Transition[Stage_Transition-ss-1,:],
             Surplus_Low_Transition[Stage_Transition-ss-1,:],Firm_Val_Func_High_Transition[Stage_Transition-ss-1,:],
             Firm_Val_Func_Low_Transition[Stage_Transition-ss-1,:],
             Work_Val_Func_High_Transition[Stage_Transition-ss-1,:],
             Work_Val_Func_Low_Transition[Stage_Transition-ss-1,:],
             Price_Vector[Stage_Transition-ss-1,:],Init_High[Stage_Transition-ss-1,:],Init_Low[Stage_Transition-ss-1,:])=ABC[ss]
        
               
                              
        #Getting the distribution
        
        Transition_Distribution_High =  np.zeros((Age_Periods,Stage_Transition,Production_Possibilities*No_Sectors))
        Transition_Distribution_Low =  np.zeros((Age_Periods,Stage_Transition,Production_Possibilities*No_Sectors))
        Needed_Distributions = Age_Periods+Transition_Periods
        Final_Distribution_High =  np.zeros((Age_Periods,Needed_Distributions,Production_Possibilities*No_Sectors))
        Final_Distribution_Low =  np.zeros((Age_Periods,Needed_Distributions,Production_Possibilities*No_Sectors))
        #Final_Firm_Transition_Mat_High = np.zeros((Needed_Distributions,Age_Periods-1,Production_Possibilities*No_Sectors,Production_Possibilities*No_Sectors))
        #Final_Firm_Transition_Mat_Low = np.zeros((Needed_Distributions,Age_Periods-1,Production_Possibilities*No_Sectors,Production_Possibilities*No_Sectors))
        Final_Orig_Firm_Transition_Mat_High = [[0]*(Age_Periods-1)]*Needed_Distributions #np.zeros((Needed_Distributions,Age_Periods-1,Production_Possibilities*No_Sectors,Production_Possibilities*No_Sectors))
        Final_Orig_Firm_Transition_Mat_Low = [[0]*(Age_Periods-1)]*Needed_Distributions #np.zeros((Needed_Distributions,Age_Periods-1,Production_Possibilities*No_Sectors,Production_Possibilities*No_Sectors))
        Final_Firm_Val_Func_High_Transition = np.zeros((Needed_Distributions,Age_Periods,Production_Possibilities*No_Sectors))
        Final_Firm_Val_Func_Low_Transition = np.zeros((Needed_Distributions,Age_Periods,Production_Possibilities*No_Sectors))
        for ss in range(Stage_Transition):
            Transition_Distribution_High[:,ss,:]=Disribution_Wokers_High #First stage - birth cohort
            Transition_Distribution_Low[:,ss,:]=Disribution_Wokers_Low
                
        
        for nn in range(Needed_Distributions):
            for ss in range(Stage_Transition):

                #Init_High,Init_Low = Initial_Cohort_Naive(Price_Vector[ss,:],4)
                
                Transition_Distribution_High[0,ss,:]=Init_High[ss,:]
                Transition_Distribution_Low[0,ss,:]=Init_Low[ss,:]
                for ii in reversed(range(1,Transition_Distribution_High.shape[0])):
                    Transition_Distribution_High[ii,ss,:] = Transition_Distribution_High[ii-1,ss,:]*Worker_Transition_Matrices_High_Transition[ss][ii-1]
                    Transition_Distribution_Low[ii,ss,:] = Transition_Distribution_Low[ii-1,ss,:]*Worker_Transition_Matrices_Low_Transition[ss][ii-1]
                    
            for ll in range(Age_Periods-1):
                Final_Distribution_High[ll,Needed_Distributions-1-nn,:]=Transition_Distribution_High[ll,Needed_Distributions-1-nn+ll,:]
                Final_Distribution_Low[ll,Needed_Distributions-1-nn,:]=Transition_Distribution_Low[ll,Needed_Distributions-1-nn+ll,:]
                #Final_Firm_Transition_Mat_High[Needed_Distributions-1-nn,ll,:,:]=Firm_Transition_Matrices_High_Transition[Needed_Distributions-1-nn+ll+1,ll,:]
                #Final_Firm_Transition_Mat_Low[Needed_Distributions-1-nn,ll,:,:]=Firm_Transition_Matrices_Low_Transition[Needed_Distributions-1-nn+ll+1,ll,:]
                #Final_Orig_Firm_Transition_Mat_High[Needed_Distributions-1-nn][ll]=Orig_Firm_Transition_Matrices_High_Transition[Needed_Distributions-1-nn+ll+1][ll]
                #Final_Orig_Firm_Transition_Mat_Low[Needed_Distributions-1-nn][ll]=Orig_Firm_Transition_Matrices_Low_Transition[Needed_Distributions-1-nn+ll+1][ll]
                Final_Firm_Val_Func_High_Transition[Needed_Distributions-1-nn,ll,:]=Firm_Val_Func_High_Transition[Needed_Distributions-1-nn+ll,ll,:]
                Final_Firm_Val_Func_Low_Transition[Needed_Distributions-1-nn,ll,:]=Firm_Val_Func_Low_Transition[Needed_Distributions-1-nn+ll,ll,:]
            
            Final_Distribution_High[39,Needed_Distributions-1-nn,:]=Transition_Distribution_High[39,Needed_Distributions-1-nn+39,:]
            Final_Distribution_Low[39,Needed_Distributions-1-nn,:]=Transition_Distribution_Low[39,Needed_Distributions-1-nn+39,:]
            Final_Firm_Val_Func_High_Transition[Needed_Distributions-1-nn,39,:]=Firm_Val_Func_High_Transition[Needed_Distributions-1-nn+39,39,:]
            Final_Firm_Val_Func_Low_Transition[Needed_Distributions-1-nn,39,:]=Firm_Val_Func_Low_Transition[Needed_Distributions-1-nn+39,39,:]
        
               #Diving by original Prob of matching to get expected value conditional on matching. 
                #The last element does not matter since mutliplied by zero as V (value of vacancy has to be zero)
        Total_Uh_Transition = np.zeros((Needed_Distributions,No_Sectors,Sector_XP_Levels))
        Total_Ul_Transition = np.zeros((Needed_Distributions,No_Sectors,Sector_XP_Levels))
        Total_U = np.zeros((Needed_Distributions,No_Sectors))
        Firm_Match_Expectation_High=np.zeros((Needed_Distributions,Age_Periods-1,No_Sectors,Sector_XP_Levels))
        Firm_Match_Expectation_Low=np.zeros((Needed_Distributions,Age_Periods-1,No_Sectors,Sector_XP_Levels))
        conv=np.zeros((Needed_Distributions))
        for nn in range(Needed_Distributions):
            for ii in range(1,Age_Periods):
                for jj in range(No_Sectors):
                    for ss in range(Sector_XP_Levels):
                        Firm_H=Orig_Firm_Transition_Matrices_High_Transition[Needed_Distributions-1-nn+ii+1-1][ii-1].toarray()
                        Firm_L=Orig_Firm_Transition_Matrices_Low_Transition[Needed_Distributions-1-nn+ii+1-1][ii-1].toarray()
                        index_start=(jj*Sector_XP_Levels+ss)*(Sector_Prod+1)
                        index_end=((jj)*Sector_XP_Levels+ss+1)*(Sector_Prod+1)
                        Firm_Match_Expectation_High[Needed_Distributions-1-nn,ii-1,jj,ss] = np.dot(
                                Firm_H[index_end-1,index_start:index_end-1]/Prob_Match_Firm_Transition[Age_Periods-1+nn-1,jj],
                                Final_Firm_Val_Func_High_Transition[Needed_Distributions-1-nn,ii,index_start:index_end-1])
                        Firm_Match_Expectation_Low[Needed_Distributions-1-nn,ii-1,jj,ss] = np.dot(
                                Firm_L[index_end-1,index_start:index_end-1]/Prob_Match_Firm_Transition[Age_Periods-1+nn-1,jj],
                                Final_Firm_Val_Func_Low_Transition[Needed_Distributions-1-nn,ii,index_start:index_end-1])


      
        Expected_Match_Value_Firm= np.zeros((Needed_Distributions,No_Sectors))   

        for jj in range(No_Sectors):  
            for ss in range(Sector_XP_Levels):
                 Total_Uh_Transition[Needed_Distributions-1,jj,ss] = np.sum(Disribution_Wokers_High[:-1,(jj*Sector_XP_Levels+ss+1)*(Sector_Prod+1)-1]) #Ignore last year since they can't find a job anymore
                 Total_Ul_Transition[Needed_Distributions-1,jj,ss] = np.sum(Disribution_Wokers_Low[:-1,(jj*Sector_XP_Levels+ss+1)*(Sector_Prod+1)-1])
        Total_U[Needed_Distributions-1,:] = Total_Uh_Transition[Needed_Distributions-1,:].sum(1)+Total_Ul_Transition[Needed_Distributions-1,:].sum(1)
         
        for jj in range(No_Sectors):
            for ss in range(Sector_XP_Levels):
                Expected_Match_Value_Firm[Needed_Distributions-1,jj] = np.dot(Firm_Match_Expectation_Low[Needed_Distributions-1,:,jj,ss],
                        Disribution_Wokers_Low[:-1,(jj*Sector_XP_Levels+ss+1)*(Sector_Prod+1)-1]/Total_U[Needed_Distributions-1,jj])+np.dot(
                    Firm_Match_Expectation_High[Needed_Distributions-1,:,jj,ss],
                    Disribution_Wokers_High[:-1,(jj*Sector_XP_Levels+ss+1)*(Sector_Prod+1)-1]/Total_U[Needed_Distributions-1,jj])+Expected_Match_Value_Firm[Needed_Distributions-1,jj]
            
        
        for nn in range(1,Needed_Distributions):
            for jj in range(No_Sectors):
                for ss in range(Sector_XP_Levels):
                    Total_Uh_Transition[Needed_Distributions-1-nn,jj,ss] = np.sum(Final_Distribution_High[:-1,Needed_Distributions-1-nn+1,(jj*Sector_XP_Levels+ss+1)*(Sector_Prod+1)-1]) #Ignore last year since they can't find a job anymore
                    Total_Ul_Transition[Needed_Distributions-1-nn,jj,ss] = np.sum(Final_Distribution_Low[:-1,Needed_Distributions-1-nn+1,(jj*Sector_XP_Levels+ss+1)*(Sector_Prod+1)-1])
            Total_U[Needed_Distributions-1-nn,:] = Total_Uh_Transition[Needed_Distributions-1-nn,:].sum(1)+Total_Ul_Transition[Needed_Distributions-1-nn,:].sum(1)
            
            
            for jj in range(No_Sectors):
                for ss in range(Sector_XP_Levels):
                    Expected_Match_Value_Firm[Needed_Distributions-1-nn,jj] = np.dot(Firm_Match_Expectation_Low[Needed_Distributions-1-nn,:,jj,ss],
                            Final_Distribution_Low[:-1,Needed_Distributions-1-nn+1,(jj*Sector_XP_Levels+ss+1)*(Sector_Prod+1)-1]/Total_U[Needed_Distributions-1-nn,jj])+np.dot(
                        Firm_Match_Expectation_High[Needed_Distributions-1-nn,:,jj,ss],
                        Final_Distribution_High[:-1,Needed_Distributions-1-nn+1,(jj*Sector_XP_Levels+ss+1)*(Sector_Prod+1)-1]/Total_U[Needed_Distributions-1-nn,jj])+Expected_Match_Value_Firm[Needed_Distributions-1-nn,jj]    
       
        
        for nn in range(Needed_Distributions): 
            dampen=0.4
            q_match = np.minimum(1,Vacancy_Cost/Expected_Match_Value_Firm[Needed_Distributions-1-nn])
            Vacancy_Amount = Total_U[Needed_Distributions-1-nn,:]/((q_match/Scale)**(1/sigma))
            Prob_Match_Firm_Prev=Prob_Match_Firm_Transition[Age_Periods-1+nn-1,:]
            Prob_Match_Worker_Prev=Prob_Match_Worker_Transition[Age_Periods-1+nn-1,:]
            Prob_Firm_Temp =(1-dampen)*np.minimum(1., Scale*(Total_U[Needed_Distributions-1-nn,:]/Vacancy_Amount)**sigma)+dampen*Prob_Match_Firm_Prev
            Prob_Work_Temp = (1-dampen)*np.minimum(1.,Scale*(Vacancy_Amount/Total_U[Needed_Distributions-1-nn,:])**(1-sigma))+dampen*Prob_Match_Worker_Prev
            #conv[nn]=np.max([np.abs(Prob_Match_Firm_Prev-Prob_Firm_Temp),np.abs(Prob_Match_Worker_Prev- Prob_Work_Temp)])
            conv[nn]=np.max([np.abs(Prob_Match_Firm_Prev-np.minimum(1., Scale*(Total_U[Needed_Distributions-1-nn,:]/Vacancy_Amount)**sigma)),np.abs(Prob_Match_Worker_Prev- np.minimum(1.,Scale*(Vacancy_Amount/Total_U[Needed_Distributions-1-nn,:])**(1-sigma)))])
            #Prob_Work_Temp=np.round(Prob_Work_Temp,5)
            #Prob_Firm_Temp=np.round(Prob_Firm_Temp,5)
            Prob_Match_Worker_Transition[Age_Periods-1+nn-1,:] =Prob_Work_Temp
            Prob_Match_Firm_Transition[Age_Periods-1+nn-1,:] = Prob_Firm_Temp
            #print(Prob_Match_Firm,Prob_Match_Worker)
        convtotal = (np.abs(conv)).max()
        print(convtotal)

    Total_L_Transition = np.zeros((Needed_Distributions,1))
    Total_M_Transition= np.zeros((Needed_Distributions,1))
    Total_M_B_Transition= np.zeros((Needed_Distributions,1))
    Total_H_Transition = np.zeros((Needed_Distributions,1))

    for nn in range(Needed_Distributions):
        Total_L_Transition[nn]=(Terminal_Values_High_Prod[0*Production_Possibilities:1*Production_Possibilities]*Final_Distribution_High[:,Needed_Distributions-1-nn,
                0*Production_Possibilities:1*Production_Possibilities]).sum()+((
                Terminal_Values_Low_Prod[0*Production_Possibilities:1*Production_Possibilities]*Final_Distribution_Low[:,Needed_Distributions-1-nn,
                0*Production_Possibilities:1*Production_Possibilities]).sum())  
        Total_M_Transition[nn]=(Terminal_Values_High_Prod[1*Production_Possibilities:2*Production_Possibilities]*Final_Distribution_High[:,Needed_Distributions-1-nn,
                1*Production_Possibilities:2*Production_Possibilities]).sum()+((
                    Terminal_Values_Low_Prod[1*Production_Possibilities:2*Production_Possibilities]*Final_Distribution_Low[:,Needed_Distributions-1-nn,
                1*Production_Possibilities:2*Production_Possibilities]).sum())
                        
        Total_M_B_Transition[nn]=(Terminal_Values_High_Prod[2*Production_Possibilities:3*Production_Possibilities]*Final_Distribution_High[:,Needed_Distributions-1-nn,
                2*Production_Possibilities:3*Production_Possibilities]).sum()+((
                    Terminal_Values_Low_Prod[2*Production_Possibilities:3*Production_Possibilities]*Final_Distribution_Low[:,Needed_Distributions-1-nn,
                2*Production_Possibilities:3*Production_Possibilities]).sum())
           
        Total_H_Transition[nn]=(Terminal_Values_High_Prod[3*Production_Possibilities:4*Production_Possibilities]*Final_Distribution_High[:,Needed_Distributions-1-nn,
                3*Production_Possibilities:4*Production_Possibilities]).sum()+((
                    Terminal_Values_Low_Prod[3*Production_Possibilities:4*Production_Possibilities]*Final_Distribution_Low[:,Needed_Distributions-1-nn,
                3*Production_Possibilities:4*Production_Possibilities]).sum())
   
    #Import_Temp = np.concatenate((Import,np.repeat(Import[-1],Needed_Distributions-Price_Sector3.shape[0])))


    
   
    Price_Sector3_Temp = np.concatenate((Price_Sector3,np.repeat(Price_Sector3[-1],Needed_Distributions-Price_Sector3.shape[0])))
    Price_Sector2_Temp = np.concatenate((Price_Sector2,np.repeat(Price_Sector2[-1],Needed_Distributions-Price_Sector3.shape[0])))
    Price_Sector1_Temp = np.concatenate((Price_Sector1,np.repeat(Price_Sector1[-1],Needed_Distributions-Price_Sector3.shape[0])))
    Price_Foreign_Temp= np.concatenate((Price_Foreign,np.repeat(Price_Foreign[-1],Needed_Distributions-Price_Sector3.shape[0])))

    #Cost_Mfg_Temp=np.concatenate((Cost_Mfg,np.repeat(Cost_Mfg[-1],Needed_Distributions-Price_Sector3.shape[0])))
    #Cost_Servs_Temp=np.concatenate((Cost_Servs,np.repeat(Cost_Servs[-1],Needed_Distributions-Price_Sector3.shape[0])))
    
    Total_Expenditure_Temp = Price_Sector1_Temp*Total_L_Transition.flatten()+Price_Sector2_Temp*Total_M_B_Transition.flatten()+Price_Sector3_Temp*Total_H_Transition.flatten()+Price_Foreign_Temp*Total_M_Transition.flatten()

    
    


    Demand_M1=((1-alpha_out)*Total_Expenditure_Temp) * (Price_Sector2_Temp)**(-eta_mfg)/(
                    Price_Sector2_Temp**(1-eta_mfg)+(Price_Foreign_Temp)**(1-eta_mfg))
    
    
    #0.05
    
    Demand_M=((1-alpha_out)*Total_Expenditure_Temp) * (Price_Foreign_Temp)**(-eta_mfg)/(
                    Price_Sector2_Temp**(1-eta_mfg)+(Price_Foreign_Temp)**(1-eta_mfg))
    
    Frac_to_Mfg=0.5
    Export_Mfg= (Frac_to_Mfg)*(Demand_M-Total_M_Transition.flatten())*Price_Foreign_Temp/Price_Sector2_Temp
    Export_Srvs = (1-Frac_to_Mfg)*(Demand_M-Total_M_Transition.flatten())*Price_Foreign_Temp/Price_Sector1_Temp
    Demand_M_B = Demand_M1+Export_Mfg
      
    #Demand_L = Total_Output*(Price_Sector1/Cost_Servs)**(-eta_ins)*(Cost_Servs/Price_Final_Good)**(-eta_out)
    Demand_L =  ((alpha_out)*Total_Expenditure_Temp) * (Price_Sector1_Temp)**(-eta_ins)/(
                    Price_Sector1_Temp**(1-eta_ins)+(Price_Sector3_Temp)**(1-eta_ins))+Export_Srvs
      
    #Demand_H = Total_Output*(Price_Sector3/Cost_Servs)**(-eta_ins)*(Cost_Servs/Price_Final_Good)**(-eta_out)
    Demand_H=((alpha_out)*Total_Expenditure_Temp) * (Price_Sector3_Temp)**(-eta_ins)/(
                    Price_Sector1_Temp**(1-eta_ins)+(Price_Sector3_Temp)**(1-eta_ins))
    




    
    #Residuals
    Resid_Vector = np.zeros((2,Demand_H.shape[0]))
    
    #Resid_Vector[1] = Demand_M-Total_M
    #Resid_Vector[0,:] = (Demand_M-Total_M_Transition.flatten() - Import)/Demand_M
    #Resid_Vector[0,:] = (Demand_L-Total_L_Transition.flatten())/Demand_L
    Resid_Vector[0,:] = (Demand_M_B-Total_M_B_Transition.flatten())/Demand_M_B
    Resid_Vector[1,:] = (Demand_H-Total_H_Transition.flatten())/Demand_H
    print(Resid_Vector)
    Evaluated =np.append(Resid_Vector[0,:Transition_Periods],Resid_Vector[1,:Transition_Periods])
    return Evaluated                      
