import numpy as np
from scipy.sparse import csr_matrix

def TransitionValFunc_Alt(ss,Transition_Prices,Prob_Match_Worker_Transition,Prob_Match_Firm_Transition,
                         Age_Periods,No_Sectors,Production_Possibilities,Unemp_High,Unemp_Low,Skill_vals,
                         UnempBenefit,Prod_Vals,Exper,Switch_Cost,Switch_Cost2,Prod_Change_Matrix,
                         Exper_Gain_Matrix,Productivity_Params,Experience_Levels,Prod_Draw,Beta,
                         Switcher_Matrix,Switch_Vector,Fraction_Worker_Surplus):
    Price = Transition_Prices[ss:ss+Age_Periods,:]
    Prob_Match_Worker=Prob_Match_Worker_Transition[ss:ss+Age_Periods-1,:]
    Prob_Match_Firm=Prob_Match_Firm_Transition[ss:ss+Age_Periods-1,:]
    Firm_Val_Func_High = np.zeros((Age_Periods,No_Sectors*(Production_Possibilities))) #+1 for unemployment state
    Firm_Val_Func_Low = 0*Firm_Val_Func_High
    Work_Val_Func_High = 0*Firm_Val_Func_High
    Work_Val_Func_Low = 0*Firm_Val_Func_High
    Surplus_High = np.zeros((Age_Periods,No_Sectors*(Production_Possibilities)))
    Surplus_Low =  0*Surplus_High
    Surplus_Check_High =0*Surplus_High
    Surplus_Check_Low=0*Surplus_High
    Unemp_Val_Func_High=Unemp_High *(np.ones((Age_Periods,No_Sectors)))
    Unemp_Val_Func_Low= Unemp_Low*(np.ones((Age_Periods,No_Sectors)))
    Firm_Continuation_Value_High = 0*Firm_Val_Func_High
    Firm_Continuation_Value_Low = 0*Firm_Val_Func_High
    Worker_Continuation_Value_High= 0*Firm_Val_Func_High
    Worker_Continuation_Value_Low = 0*Firm_Val_Func_High
    Unemp_Continuation_Value_High = 0*Unemp_Val_Func_High
    Unemp_Continuation_Value_Low = 0*Unemp_Val_Func_High
    for ii in range(No_Sectors):
        Work_Val_Func_High[:,-1-(ii*Production_Possibilities)]=Skill_vals[0]*UnempBenefit
        Work_Val_Func_Low[:,-1-(ii*Production_Possibilities)]=Skill_vals[1]*UnempBenefit
    
    
    Choice_High = 0*Firm_Val_Func_High
    Choice_Low = 0*Firm_Val_Func_High
    Wages_High = 0*Firm_Val_Func_High
    Wages_Low=0*Firm_Val_Func_High
    
    Firm_Transition_Matrices_High = np.zeros((Age_Periods,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
    Firm_Transition_Matrices_Low = np.zeros((Age_Periods,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
    
    Worker_Transition_Matrices_High = np.zeros((Age_Periods,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
    Worker_Transition_Matrices_Low = np.zeros((Age_Periods,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
    
  
            
    
    #Code Block Starts
    
    Combinations= np.kron(Prod_Vals,Exper)
    Terminal_Values_High_Prod = (Skill_vals[0]*Combinations)
    Terminal_Values_Low_Prod = Skill_vals[1]*Combinations
    
    for jj in range(No_Sectors):
        Terminal_Values_High_Prod=np.concatenate((Terminal_Values_High_Prod[:(jj+1)*(Production_Possibilities-1)+jj],[0],
                                                                 Terminal_Values_High_Prod[(jj+1)*(Production_Possibilities-1)+jj:]))
        Terminal_Values_Low_Prod=np.concatenate((Terminal_Values_Low_Prod[:(jj+1)*(Production_Possibilities-1)+jj],[0],
                                                                 Terminal_Values_Low_Prod[(jj+1)*(Production_Possibilities-1)+jj:]))
    
    Terminal_Values_High=Price*Terminal_Values_High_Prod
    Terminal_Values_Low=Price*Terminal_Values_Low_Prod   
   
    

    
    for ii in range(Age_Periods):     
        index_current = Age_Periods-ii-1
        index_before = index_current-1
        Max_Unemp_Option_Sector1_High = np.tile(max([Unemp_High+Unemp_Continuation_Value_High[index_current,0],
                                                             Unemp_High+Unemp_Continuation_Value_High[index_current,1]-Switch_Cost,
                                                             Unemp_High+Unemp_Continuation_Value_High[index_current,2]-Switch_Cost2]),(
                                                                     Production_Possibilities,1)).T
        
        Choice_TempSector1_High=np.tile(np.argmax(([Unemp_High+Unemp_Continuation_Value_High[index_current,0],
                                                             Unemp_High+Unemp_Continuation_Value_High[index_current,1]-Switch_Cost,
                                                             Unemp_High+Unemp_Continuation_Value_High[index_current,2]-Switch_Cost2]),axis=0),
                                                                             (Production_Possibilities,1) ).T
                                                                              
        Max_Unemp_Option_Sector2_High =np.tile(max([Unemp_High+Unemp_Continuation_Value_High[index_current,0],
                                                             Unemp_High+Unemp_Continuation_Value_High[index_current,1],
                                                              Unemp_High+Unemp_Continuation_Value_High[index_current,2]-Switch_Cost2]),(
                                                                     Production_Possibilities,1)).T
        
        Choice_TempSector2_High=np.tile(np.argmax((Unemp_High+Unemp_Continuation_Value_High[index_current,0],
                                                              Unemp_High+Unemp_Continuation_Value_High[index_current,1]+0.00000000001,
                                                              Unemp_High+Unemp_Continuation_Value_High[index_current,2]-Switch_Cost2),axis=0),(
                                                                     Production_Possibilities,1)).T                                                                    
                               
        Max_Unemp_Option_Sector3_High =np.tile(max([Unemp_High+Unemp_Continuation_Value_High[index_current,0],
                                                              Unemp_High+Unemp_Continuation_Value_High[index_current,1]-Switch_Cost,
                                                            Unemp_High+Unemp_Continuation_Value_High[index_current,2]]),(
                                                                     Production_Possibilities,1)).T
        
        Choice_TempSector3_High=np.tile(np.argmax((Unemp_High+Unemp_Continuation_Value_High[index_current,0],
                                                              Unemp_High+Unemp_Continuation_Value_High[index_current,1]-Switch_Cost,
                                                            Unemp_High+Unemp_Continuation_Value_High[index_current,2]+0.00000000001),axis=0),(
                                                                     Production_Possibilities,1)).T   
                                                                  
        Choice_High[index_current,:] = np.concatenate((Choice_TempSector1_High,Choice_TempSector2_High,Choice_TempSector3_High),axis=1)                                                                          
        Max_Unemp_Option_High = np.concatenate((Max_Unemp_Option_Sector1_High,Max_Unemp_Option_Sector2_High,Max_Unemp_Option_Sector3_High),axis=1)
                                 
        
        Max_Unemp_Option_Sector1_Low = np.tile(max([Unemp_Low+Unemp_Continuation_Value_Low[index_current,0],
                                                             Unemp_Low+Unemp_Continuation_Value_Low[index_current,1]-Switch_Cost,
                                                             Unemp_Low+Unemp_Continuation_Value_Low[index_current,2]-Switch_Cost2]),(
                                                                     Production_Possibilities,1)).T
        
        Choice_TempSector1_Low=np.tile(np.argmax(([Unemp_Low+Unemp_Continuation_Value_Low[index_current,0],
                                                             Unemp_Low+Unemp_Continuation_Value_Low[index_current,1]-Switch_Cost,
                                                             Unemp_Low+Unemp_Continuation_Value_Low[index_current,2]-Switch_Cost2]),axis=0),
                                                                             (Production_Possibilities,1) ).T
                                                                              
        Max_Unemp_Option_Sector2_Low =np.tile(max([Unemp_Low+Unemp_Continuation_Value_Low[index_current,0],
                                                             Unemp_Low+Unemp_Continuation_Value_Low[index_current,1],
                                                              Unemp_Low+Unemp_Continuation_Value_Low[index_current,2]-Switch_Cost2]),(
                                                                     Production_Possibilities,1)).T
        
        Choice_TempSector2_Low=np.tile(np.argmax((Unemp_Low+Unemp_Continuation_Value_Low[index_current,0],
                                                              Unemp_Low+Unemp_Continuation_Value_Low[index_current,1]+0.00000000001,
                                                              Unemp_Low+Unemp_Continuation_Value_Low[index_current,2]-Switch_Cost2),axis=0),(
                                                                     Production_Possibilities,1)).T                                                                    
                               
        Max_Unemp_Option_Sector3_Low =np.tile(max([Unemp_Low+Unemp_Continuation_Value_Low[index_current,0],
                                                              Unemp_Low+Unemp_Continuation_Value_Low[index_current,1]-Switch_Cost,
                                                            Unemp_Low+Unemp_Continuation_Value_Low[index_current,2]]),(
                                                                     Production_Possibilities,1)).T
        
        Choice_TempSector3_Low=np.tile(np.argmax((Unemp_Low+Unemp_Continuation_Value_Low[index_current,0],
                                                              Unemp_Low+Unemp_Continuation_Value_Low[index_current,1]-Switch_Cost,
                                                            Unemp_Low+Unemp_Continuation_Value_Low[index_current,2]+0.00000000001),axis=0),(
                                                                     Production_Possibilities,1)).T   

                                                                        
        Choice_Low[index_current,:] = np.concatenate((Choice_TempSector1_Low,Choice_TempSector2_Low,Choice_TempSector3_Low),axis=1)                                                                          
        Max_Unemp_Option_Low = np.concatenate((Max_Unemp_Option_Sector1_Low,Max_Unemp_Option_Sector2_Low,Max_Unemp_Option_Sector3_Low),axis=1)   

        Surplus_High[index_current,:] = Terminal_Values_High[index_current,:] + Firm_Continuation_Value_High[index_current,:] + Worker_Continuation_Value_High[index_current,:] -Max_Unemp_Option_High
        Surplus_Low[index_current,:] = Terminal_Values_Low[index_current,:] + Firm_Continuation_Value_Low[index_current,:] + Worker_Continuation_Value_Low[index_current,:] -Max_Unemp_Option_Low
  
        
        Surplus_Check_High[index_current,:]=np.maximum(0,100000*Surplus_High[index_current,:])
        Surplus_Check_Low[index_current,:]=np.maximum(0,100000*Surplus_Low[index_current,:])
        Surplus_Check_High[index_current,:]=np.minimum(1,Surplus_Check_High[index_current,:])
        Surplus_Check_Low[index_current,:]=np.minimum(1,Surplus_Check_Low[index_current,:])       
        Surplus_Check_High[index_current,3*Production_Possibilities-1]=1
        Surplus_Check_Low[index_current,3*Production_Possibilities-1]=1
        Surplus_Check_High[index_current,2*Production_Possibilities-1]=1
        Surplus_Check_Low[index_current,2*Production_Possibilities-1]=1
        Surplus_Check_High[index_current,1*Production_Possibilities-1]=1
        Surplus_Check_Low[index_current,1*Production_Possibilities-1]=1


        
        
        Wages_High[index_current,:]=Surplus_Check_High[index_current,:]*((Max_Unemp_Option_High-Worker_Continuation_Value_High[index_current,:])+Fraction_Worker_Surplus*Surplus_High[index_current,:])
        Wages_Low[index_current,:]=Surplus_Check_Low[index_current,:]*((Max_Unemp_Option_Low-Worker_Continuation_Value_Low[index_current,:])+Fraction_Worker_Surplus*Surplus_Low[index_current,:])
            
        for jj in range(No_Sectors):
            Wages_High[index_current,(jj+1)*(Production_Possibilities)-1] = Unemp_High
            Wages_Low[index_current,(jj+1)*(Production_Possibilities)-1] = Unemp_Low
        
        
        
    
        Firm_Val_Func_High[index_current,:] = Surplus_Check_High[index_current,:]*(Terminal_Values_High[index_current,:]-Wages_High[index_current,:]+(
                Firm_Continuation_Value_High[index_current,:]))
        Firm_Val_Func_Low[index_current,:] = Surplus_Check_Low[index_current,:]*(Terminal_Values_Low[index_current,:]-Wages_Low[index_current,:]+(
                Firm_Continuation_Value_Low[index_current,:]))
        
        Work_Val_Func_High[index_current,:] = Wages_High[index_current,:]+(
                Worker_Continuation_Value_High[index_current,:])
        Work_Val_Func_Low[index_current,:] = Wages_Low[index_current,:]+(
                Worker_Continuation_Value_Low[index_current,:])
    
     
        for jj in range(No_Sectors):
            Firm_Val_Func_High[index_current,(jj+1)*Production_Possibilities-1]=0.  #Enforcing free-entry
            Firm_Val_Func_Low[index_current,(jj+1)*Production_Possibilities-1] =0  #Enforcing free-entry  
        
        #Transitions
        #Tranistion Matrix for Job Values
      

        Job_Transition_Matrix=np.zeros((Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
        Job_Transition_Matrix_Worker = np.zeros((Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
        Initial_Val = np.kron(Prod_Change_Matrix,Exper_Gain_Matrix)       
        
        
        for jj in range(No_Sectors):
            index_start = (jj)*(Production_Possibilities)
            index_end = (jj+1)*Production_Possibilities            
            Job_Transition_Matrix[index_start:index_end-1,
                                  index_start:index_end-1]=Initial_Val
            Job_Transition_Matrix_Worker[index_start:index_end-1,
                                 index_start:index_end-1]=Initial_Val
            for kk in range(Productivity_Params):
                Job_Transition_Matrix[index_end-1,index_start+(Experience_Levels*kk)+1] = Prod_Draw[kk]*Prob_Match_Firm[index_before,jj]
                Job_Transition_Matrix[index_end-1,index_end-1] = 1-Prob_Match_Firm[index_before,jj] #Last state from unemployment
                Job_Transition_Matrix_Worker[index_end-1,index_start+(Experience_Levels*kk)+1] = Prod_Draw[kk]*Prob_Match_Worker[index_before,jj]
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
            index_start = (jj)*(Production_Possibilities)
            index_end = (jj+1)*Production_Possibilities
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
            Job_Transition_Matrix_Worker_Adjusted_High[index_start:index_end,(Sector_Choice+1
                                                            )*Production_Possibilities-1]=TemporHigh
        
            TemporLow=Job_Transition_Matrix_Worker_Adjusted_Low[
                    index_start:index_end,index_end-1]+((1-np.tile(Surplus_Check_Low[index_current],
                    (No_Sectors*Production_Possibilities,1)))*Job_Transition_Matrix_Worker
                    ).sum(1)[index_start:index_end]
         
            Job_Transition_Matrix_Worker_Adjusted_Low[index_start:index_end,index_end-1]=0
            Sector_Choice_Low=int(Choice_Low[index_current,index_end-1])
            Job_Transition_Matrix_Worker_Adjusted_Low[index_start:index_end,(Sector_Choice_Low+1
                                                        )*Production_Possibilities-1]=TemporLow
            
            Firm_Transition_Matrices_High[index_before,:,:]=Job_Transition_Matrix_Adjusted_High
            Firm_Transition_Matrices_Low[index_before,:,:]=Job_Transition_Matrix_Adjusted_Low
            
            
            Worker_Transition_Matrices_High[index_before,:,:]=Job_Transition_Matrix_Worker_Adjusted_High
            Worker_Transition_Matrices_Low[index_before,:,:]=Job_Transition_Matrix_Worker_Adjusted_Low            
            
                         
            Worker_Continuation_Value_High[index_before,:]=Beta*(np.dot(Job_Transition_Matrix_Worker_Adjusted_High,
                                Work_Val_Func_High[index_current,:])-np.dot(
                                        Switcher_Matrix*Job_Transition_Matrix_Worker_Adjusted_High,Switch_Vector))
            Worker_Continuation_Value_Low[index_before,:]=Beta*(np.dot(Job_Transition_Matrix_Worker_Adjusted_Low,
                               Work_Val_Func_Low[index_current,:])-np.dot(
                                       Switcher_Matrix*Job_Transition_Matrix_Worker_Adjusted_Low,Switch_Vector))
            
        for jj in range(No_Sectors):
            Unemp_Continuation_Value_High[index_before,jj] =Worker_Continuation_Value_High[index_before,(jj+1)*Production_Possibilities-1]
            Unemp_Continuation_Value_Low[index_before,jj]=Worker_Continuation_Value_Low[index_before,(jj+1)*Production_Possibilities-1]
            
        Firm_Continuation_Value_High[index_before,:]=Beta*(
                    np.dot(Job_Transition_Matrix_Adjusted_High,Firm_Val_Func_High[index_current,:]))
        Firm_Continuation_Value_Low[index_before,:]=Beta*(
                    np.dot(Job_Transition_Matrix_Adjusted_Low,Firm_Val_Func_Low[index_current,:]))
     
    return  (Worker_Transition_Matrices_High[:-1],Worker_Transition_Matrices_Low[:-1],Firm_Transition_Matrices_High[:-1],
             Firm_Transition_Matrices_Low[:-1],Wages_High,Wages_Low,Surplus_High,Surplus_Low,Firm_Val_Func_High,
             Firm_Val_Func_Low,Work_Val_Func_High,Work_Val_Func_Low,Price)
    
    
def TransitionValFunc_Analysis(ss,Transition_Prices,Prob_Match_Worker_Transition,Prob_Match_Firm_Transition,
                         Age_Periods,No_Sectors,Production_Possibilities,Unemp_High,Unemp_Low,Skill_vals,
                         UnempBenefit,Prod_Vals,Exper,Switch_Cost,Switch_Cost2,Prod_Change_Matrix,
                         Exper_Gain_Matrix,Productivity_Params,Experience_Levels,Prod_Draw,Beta,
                         Switcher_Matrix,Switch_Vector,Fraction_Worker_Surplus):
    Price = Transition_Prices[ss:ss+Age_Periods,:]
    Prob_Match_Worker=Prob_Match_Worker_Transition[ss:ss+Age_Periods-1,:]
    Prob_Match_Firm=Prob_Match_Firm_Transition[ss:ss+Age_Periods-1,:]
    Firm_Val_Func_High = np.zeros((Age_Periods,No_Sectors*(Production_Possibilities))) #+1 for unemployment state
    Firm_Val_Func_Low = 0*Firm_Val_Func_High
    Work_Val_Func_High = 0*Firm_Val_Func_High
    Work_Val_Func_Low = 0*Firm_Val_Func_High
    Surplus_High = np.zeros((Age_Periods,No_Sectors*(Production_Possibilities)))
    Surplus_Low =  0*Surplus_High
    Surplus_Check_High =0*Surplus_High
    Surplus_Check_Low=0*Surplus_High
    Unemp_Val_Func_High=Unemp_High *(np.ones((Age_Periods,No_Sectors)))
    Unemp_Val_Func_Low= Unemp_Low*(np.ones((Age_Periods,No_Sectors)))
    Firm_Continuation_Value_High = 0*Firm_Val_Func_High
    Firm_Continuation_Value_Low = 0*Firm_Val_Func_High
    Worker_Continuation_Value_High= 0*Firm_Val_Func_High
    Worker_Continuation_Value_Low = 0*Firm_Val_Func_High
    Unemp_Continuation_Value_High = 0*Unemp_Val_Func_High
    Unemp_Continuation_Value_Low = 0*Unemp_Val_Func_High
    for ii in range(No_Sectors):
        Work_Val_Func_High[:,-1-(ii*Production_Possibilities)]=Skill_vals[0]*UnempBenefit
        Work_Val_Func_Low[:,-1-(ii*Production_Possibilities)]=Skill_vals[1]*UnempBenefit
    
    
    Choice_High = 0*Firm_Val_Func_High
    Choice_Low = 0*Firm_Val_Func_High
    Wages_High = 0*Firm_Val_Func_High
    Wages_Low=0*Firm_Val_Func_High
    
    Firm_Transition_Matrices_High = np.zeros((Age_Periods,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
    Firm_Transition_Matrices_Low = np.zeros((Age_Periods,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
    
    Worker_Transition_Matrices_High = np.zeros((Age_Periods,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
    Worker_Transition_Matrices_Low = np.zeros((Age_Periods,Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
    
  
            
    
    #Code Block Starts
    
    Combinations= np.kron(Prod_Vals,Exper)
    Terminal_Values_High_Prod = (Skill_vals[0]*Combinations)
    Terminal_Values_Low_Prod = Skill_vals[1]*Combinations
    
    for jj in range(No_Sectors):
        Terminal_Values_High_Prod=np.concatenate((Terminal_Values_High_Prod[:(jj+1)*(Production_Possibilities-1)+jj],[0],
                                                                 Terminal_Values_High_Prod[(jj+1)*(Production_Possibilities-1)+jj:]))
        Terminal_Values_Low_Prod=np.concatenate((Terminal_Values_Low_Prod[:(jj+1)*(Production_Possibilities-1)+jj],[0],
                                                                 Terminal_Values_Low_Prod[(jj+1)*(Production_Possibilities-1)+jj:]))
    
    Terminal_Values_High=Price*Terminal_Values_High_Prod
    Terminal_Values_Low=Price*Terminal_Values_Low_Prod   
   
    

    
    for ii in range(Age_Periods):     
        index_current = Age_Periods-ii-1
        index_before = index_current-1
        Max_Unemp_Option_Sector1_High = np.tile(max([Unemp_High+Unemp_Continuation_Value_High[index_current,0],
                                                             Unemp_High+Unemp_Continuation_Value_High[index_current,1]-Switch_Cost,
                                                             Unemp_High+Unemp_Continuation_Value_High[index_current,2]-Switch_Cost2]),(
                                                                     Production_Possibilities,1)).T
        
        Choice_TempSector1_High=np.tile(np.argmax(([Unemp_High+Unemp_Continuation_Value_High[index_current,0],
                                                             Unemp_High+Unemp_Continuation_Value_High[index_current,1]-Switch_Cost,
                                                             Unemp_High+Unemp_Continuation_Value_High[index_current,2]-Switch_Cost2]),axis=0),
                                                                             (Production_Possibilities,1) ).T
                                                                              
        Max_Unemp_Option_Sector2_High =np.tile(max([Unemp_High+Unemp_Continuation_Value_High[index_current,0],
                                                             Unemp_High+Unemp_Continuation_Value_High[index_current,1],
                                                              Unemp_High+Unemp_Continuation_Value_High[index_current,2]-Switch_Cost2]),(
                                                                     Production_Possibilities,1)).T
        
        Choice_TempSector2_High=np.tile(np.argmax((Unemp_High+Unemp_Continuation_Value_High[index_current,0],
                                                              Unemp_High+Unemp_Continuation_Value_High[index_current,1]+0.00000000001,
                                                              Unemp_High+Unemp_Continuation_Value_High[index_current,2]-Switch_Cost2),axis=0),(
                                                                     Production_Possibilities,1)).T                                                                    
                               
        Max_Unemp_Option_Sector3_High =np.tile(max([Unemp_High+Unemp_Continuation_Value_High[index_current,0],
                                                              Unemp_High+Unemp_Continuation_Value_High[index_current,1]-Switch_Cost,
                                                            Unemp_High+Unemp_Continuation_Value_High[index_current,2]]),(
                                                                     Production_Possibilities,1)).T
        
        Choice_TempSector3_High=np.tile(np.argmax((Unemp_High+Unemp_Continuation_Value_High[index_current,0],
                                                              Unemp_High+Unemp_Continuation_Value_High[index_current,1]-Switch_Cost,
                                                            Unemp_High+Unemp_Continuation_Value_High[index_current,2]+0.00000000001),axis=0),(
                                                                     Production_Possibilities,1)).T   
                                                                  
        Choice_High[index_current,:] = np.concatenate((Choice_TempSector1_High,Choice_TempSector2_High,Choice_TempSector3_High),axis=1)                                                                          
        Max_Unemp_Option_High = np.concatenate((Max_Unemp_Option_Sector1_High,Max_Unemp_Option_Sector2_High,Max_Unemp_Option_Sector3_High),axis=1)
                                 
        
        Max_Unemp_Option_Sector1_Low = np.tile(max([Unemp_Low+Unemp_Continuation_Value_Low[index_current,0],
                                                             Unemp_Low+Unemp_Continuation_Value_Low[index_current,1]-Switch_Cost,
                                                             Unemp_Low+Unemp_Continuation_Value_Low[index_current,2]-Switch_Cost2]),(
                                                                     Production_Possibilities,1)).T
        
        Choice_TempSector1_Low=np.tile(np.argmax(([Unemp_Low+Unemp_Continuation_Value_Low[index_current,0],
                                                             Unemp_Low+Unemp_Continuation_Value_Low[index_current,1]-Switch_Cost,
                                                             Unemp_Low+Unemp_Continuation_Value_Low[index_current,2]-Switch_Cost2]),axis=0),
                                                                             (Production_Possibilities,1) ).T
                                                                              
        Max_Unemp_Option_Sector2_Low =np.tile(max([Unemp_Low+Unemp_Continuation_Value_Low[index_current,0],
                                                             Unemp_Low+Unemp_Continuation_Value_Low[index_current,1],
                                                              Unemp_Low+Unemp_Continuation_Value_Low[index_current,2]-Switch_Cost2]),(
                                                                     Production_Possibilities,1)).T
        
        Choice_TempSector2_Low=np.tile(np.argmax((Unemp_Low+Unemp_Continuation_Value_Low[index_current,0],
                                                              Unemp_Low+Unemp_Continuation_Value_Low[index_current,1]+0.00000000001,
                                                              Unemp_Low+Unemp_Continuation_Value_Low[index_current,2]-Switch_Cost2),axis=0),(
                                                                     Production_Possibilities,1)).T                                                                    
                               
        Max_Unemp_Option_Sector3_Low =np.tile(max([Unemp_Low+Unemp_Continuation_Value_Low[index_current,0],
                                                              Unemp_Low+Unemp_Continuation_Value_Low[index_current,1]-Switch_Cost,
                                                            Unemp_Low+Unemp_Continuation_Value_Low[index_current,2]]),(
                                                                     Production_Possibilities,1)).T
        
        Choice_TempSector3_Low=np.tile(np.argmax((Unemp_Low+Unemp_Continuation_Value_Low[index_current,0],
                                                              Unemp_Low+Unemp_Continuation_Value_Low[index_current,1]-Switch_Cost,
                                                            Unemp_Low+Unemp_Continuation_Value_Low[index_current,2]+0.00000000001),axis=0),(
                                                                     Production_Possibilities,1)).T   

                                                                        
        Choice_Low[index_current,:] = np.concatenate((Choice_TempSector1_Low,Choice_TempSector2_Low,Choice_TempSector3_Low),axis=1)                                                                          
        Max_Unemp_Option_Low = np.concatenate((Max_Unemp_Option_Sector1_Low,Max_Unemp_Option_Sector2_Low,Max_Unemp_Option_Sector3_Low),axis=1)   

        Surplus_High[index_current,:] = Terminal_Values_High[index_current,:] + Firm_Continuation_Value_High[index_current,:] + Worker_Continuation_Value_High[index_current,:] -Max_Unemp_Option_High
        Surplus_Low[index_current,:] = Terminal_Values_Low[index_current,:] + Firm_Continuation_Value_Low[index_current,:] + Worker_Continuation_Value_Low[index_current,:] -Max_Unemp_Option_Low
  
        
        Surplus_Check_High[index_current,:]=np.maximum(0,100000*Surplus_High[index_current,:])
        Surplus_Check_Low[index_current,:]=np.maximum(0,100000*Surplus_Low[index_current,:])
        Surplus_Check_High[index_current,:]=np.minimum(1,Surplus_Check_High[index_current,:])
        Surplus_Check_Low[index_current,:]=np.minimum(1,Surplus_Check_Low[index_current,:])       
        Surplus_Check_High[index_current,3*Production_Possibilities-1]=1
        Surplus_Check_Low[index_current,3*Production_Possibilities-1]=1
        Surplus_Check_High[index_current,2*Production_Possibilities-1]=1
        Surplus_Check_Low[index_current,2*Production_Possibilities-1]=1
        Surplus_Check_High[index_current,1*Production_Possibilities-1]=1
        Surplus_Check_Low[index_current,1*Production_Possibilities-1]=1


        
        
        Wages_High[index_current,:]=Surplus_Check_High[index_current,:]*((Max_Unemp_Option_High-Worker_Continuation_Value_High[index_current,:])+Fraction_Worker_Surplus*Surplus_High[index_current,:])
        Wages_Low[index_current,:]=Surplus_Check_Low[index_current,:]*((Max_Unemp_Option_Low-Worker_Continuation_Value_Low[index_current,:])+Fraction_Worker_Surplus*Surplus_Low[index_current,:])
            
        for jj in range(No_Sectors):
            Wages_High[index_current,(jj+1)*(Production_Possibilities)-1] = Unemp_High
            Wages_Low[index_current,(jj+1)*(Production_Possibilities)-1] = Unemp_Low
        
        
        
    
        Firm_Val_Func_High[index_current,:] = Surplus_Check_High[index_current,:]*(Terminal_Values_High[index_current,:]-Wages_High[index_current,:]+(
                Firm_Continuation_Value_High[index_current,:]))
        Firm_Val_Func_Low[index_current,:] = Surplus_Check_Low[index_current,:]*(Terminal_Values_Low[index_current,:]-Wages_Low[index_current,:]+(
                Firm_Continuation_Value_Low[index_current,:]))
        
        Work_Val_Func_High[index_current,:] = Wages_High[index_current,:]+(
                Worker_Continuation_Value_High[index_current,:])
        Work_Val_Func_Low[index_current,:] = Wages_Low[index_current,:]+(
                Worker_Continuation_Value_Low[index_current,:])
    
     
        for jj in range(No_Sectors):
            Firm_Val_Func_High[index_current,(jj+1)*Production_Possibilities-1]=0.  #Enforcing free-entry
            Firm_Val_Func_Low[index_current,(jj+1)*Production_Possibilities-1] =0  #Enforcing free-entry  
        
        #Transitions
        #Tranistion Matrix for Job Values
      

        Job_Transition_Matrix=np.zeros((Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
        Job_Transition_Matrix_Worker = np.zeros((Firm_Val_Func_Low.shape[1],Firm_Val_Func_Low.shape[1]))
        Initial_Val = np.kron(Prod_Change_Matrix,Exper_Gain_Matrix)       
        
        
        for jj in range(No_Sectors):
            index_start = (jj)*(Production_Possibilities)
            index_end = (jj+1)*Production_Possibilities            
            Job_Transition_Matrix[index_start:index_end-1,
                                  index_start:index_end-1]=Initial_Val
            Job_Transition_Matrix_Worker[index_start:index_end-1,
                                 index_start:index_end-1]=Initial_Val
            for kk in range(Productivity_Params):
                Job_Transition_Matrix[index_end-1,index_start+(Experience_Levels*kk)+1] = Prod_Draw[kk]*Prob_Match_Firm[index_before,jj]
                Job_Transition_Matrix[index_end-1,index_end-1] = 1-Prob_Match_Firm[index_before,jj] #Last state from unemployment
                Job_Transition_Matrix_Worker[index_end-1,index_start+(Experience_Levels*kk)+1] = Prod_Draw[kk]*Prob_Match_Worker[index_before,jj]
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
            index_start = (jj)*(Production_Possibilities)
            index_end = (jj+1)*Production_Possibilities
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
            Job_Transition_Matrix_Worker_Adjusted_High[index_start:index_end,(Sector_Choice+1
                                                            )*Production_Possibilities-1]=TemporHigh
        
            TemporLow=Job_Transition_Matrix_Worker_Adjusted_Low[
                    index_start:index_end,index_end-1]+((1-np.tile(Surplus_Check_Low[index_current],
                    (No_Sectors*Production_Possibilities,1)))*Job_Transition_Matrix_Worker
                    ).sum(1)[index_start:index_end]
         
            Job_Transition_Matrix_Worker_Adjusted_Low[index_start:index_end,index_end-1]=0
            Sector_Choice_Low=int(Choice_Low[index_current,index_end-1])
            Job_Transition_Matrix_Worker_Adjusted_Low[index_start:index_end,(Sector_Choice_Low+1
                                                        )*Production_Possibilities-1]=TemporLow
            
            Firm_Transition_Matrices_High[index_before,:,:]=Job_Transition_Matrix_Adjusted_High
            Firm_Transition_Matrices_Low[index_before,:,:]=Job_Transition_Matrix_Adjusted_Low
            
            
            Worker_Transition_Matrices_High[index_before,:,:]=Job_Transition_Matrix_Worker_Adjusted_High
            Worker_Transition_Matrices_Low[index_before,:,:]=Job_Transition_Matrix_Worker_Adjusted_Low            
            
                         
            Worker_Continuation_Value_High[index_before,:]=Beta*(np.dot(Job_Transition_Matrix_Worker_Adjusted_High,
                                Work_Val_Func_High[index_current,:])-np.dot(
                                        Switcher_Matrix*Job_Transition_Matrix_Worker_Adjusted_High,Switch_Vector))
            Worker_Continuation_Value_Low[index_before,:]=Beta*(np.dot(Job_Transition_Matrix_Worker_Adjusted_Low,
                               Work_Val_Func_Low[index_current,:])-np.dot(
                                       Switcher_Matrix*Job_Transition_Matrix_Worker_Adjusted_Low,Switch_Vector))
            
        for jj in range(No_Sectors):
            Unemp_Continuation_Value_High[index_before,jj] =Worker_Continuation_Value_High[index_before,(jj+1)*Production_Possibilities-1]
            Unemp_Continuation_Value_Low[index_before,jj]=Worker_Continuation_Value_Low[index_before,(jj+1)*Production_Possibilities-1]
            
        Firm_Continuation_Value_High[index_before,:]=Beta*(
                    np.dot(Job_Transition_Matrix_Adjusted_High,Firm_Val_Func_High[index_current,:]))
        Firm_Continuation_Value_Low[index_before,:]=Beta*(
                    np.dot(Job_Transition_Matrix_Adjusted_Low,Firm_Val_Func_Low[index_current,:]))
     
    return  (Worker_Transition_Matrices_High[:-1],Worker_Transition_Matrices_Low[:-1],Firm_Transition_Matrices_High[:-1],
             Firm_Transition_Matrices_Low[:-1],Wages_High,Wages_Low,Surplus_High,Surplus_Low,Firm_Val_Func_High,
             Firm_Val_Func_Low,Work_Val_Func_High,Work_Val_Func_Low,Price)
    

def Initial_Cohort_Fixed(Work_Val_Func_High,Work_Val_Func_Low,Cost_Vector,Distr_Init_High,
                         Distr_Init_Low,Production_Possibilities,No_Sectors,Grid_Points,Prob_Types,Prob_Draw):
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
    
def TransitionValFunc_Trade(ss,Transition_Prices,Prob_Match_Worker_Transition,Prob_Match_Firm_Transition,
                         Age_Periods,No_Sectors,Production_Possibilities,Unemp_High,Unemp_Low,Skill_vals,
                         Prod_Vals,Exper,Switch_Cost,Switch_Cost1,Switch_Cost2,Prod_Change_Matrix,
                         Exper_Gain_Matrix,Sector_XP_Gain,Sector_XP_Gain_Matrix,
                         Productivity_Params,Experience_Levels,Sector_XP_Levels,Prod_Draw,Beta,
                         Switcher_Matrix,Switch_Vector,Fraction_Worker_Surplus,Cost_Vector,Distr_Init_High,Distr_Init_Low,Prob_Types,Prob_Draw,
                         Grid_Points):
    
    Price = Transition_Prices[ss:ss+Age_Periods,:]
    Prob_Match_Worker=Prob_Match_Worker_Transition[ss:ss+Age_Periods-1,:]
    Prob_Match_Firm=Prob_Match_Firm_Transition[ss:ss+Age_Periods-1,:]

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
    Sector_Prod=Experience_Levels*Productivity_Params
    Surplus_High = np.zeros((Age_Periods,No_Sectors*(Production_Possibilities)))
    Surplus_Low =  0*Surplus_High
    Surplus_Check_High =0*Surplus_High
    Surplus_Check_Low=0*Surplus_High
    Terminal_Productivities = np.zeros((Production_Possibilities*No_Sectors))
    for ii in range(No_Sectors):
        Temp_Val = np.kron(Sector_XP_Gain,np.kron(Exper,Prod_Vals[ii*Productivity_Params:(ii+1)*Productivity_Params]))
        for jj in range(Sector_XP_Levels):
            Work_Val_Func_High[:,-1-(ii*Production_Possibilities+jj*(Sector_Prod+1))]=Unemp_High
            Work_Val_Func_Low[:,-1-(ii*Production_Possibilities+jj*(Sector_Prod+1))]=Unemp_Low
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
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)])
                                                                   
        Choice_TempSector2_High=np.argmax(
            (np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1],Production_Possibilities),
             Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities:2*Production_Possibilities]+0.00000000001,
             np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
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
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)])
                                                                   
        Choice_TempSector2_Low=np.argmax(
            (np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1],Production_Possibilities),
             Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities:2*Production_Possibilities]+0.00000000001,
             np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
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
     
        
    Init_High,Init_Low,Ch,Cl  = Initial_Cohort_Fixed(Work_Val_Func_High,Work_Val_Func_Low,Cost_Vector,Distr_Init_High,
                         Distr_Init_Low,Production_Possibilities,No_Sectors,Grid_Points,Prob_Types,Prob_Draw)    
        
        
    return  (Worker_Transition_Matrices_High,Worker_Transition_Matrices_Low,
             Original_F_High_Transition_Matrices,
             Original_F_Low_Transition_Matrices,
             Wages_High,Wages_Low,Surplus_High,Surplus_Low,Firm_Val_Func_High,
             Firm_Val_Func_Low,Work_Val_Func_High,Work_Val_Func_Low,Price,Init_High,Init_Low)
    
def TransitionValFunc_TradeNPC(ss,Transition_Prices,Prob_Match_Worker_Transition,Prob_Match_Firm_Transition,
                         Age_Periods,No_Sectors,Production_Possibilities,Unemp_High,Unemp_Low,Skill_vals,
                         Prod_Vals,Exper,Switch_Cost,Switch_Cost1,Switch_Cost2,Prod_Change_Matrix,
                         Exper_Gain_Matrix,Sector_XP_Gain,Sector_XP_Gain_Matrix,
                         Productivity_Params,Experience_Levels,Sector_XP_Levels,Prod_Draw,Beta,
                         Switcher_Matrix,Switch_Vector,Fraction_Worker_Surplus,Cost_Vector,Distr_Init_High,Distr_Init_Low,Prob_Types,Prob_Draw,
                         Grid_Points,Non_Pec):
    
    Price = Transition_Prices[ss:ss+Age_Periods,:]
    Prob_Match_Worker=Prob_Match_Worker_Transition[ss:ss+Age_Periods-1,:]
    Prob_Match_Firm=Prob_Match_Firm_Transition[ss:ss+Age_Periods-1,:]

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
    Sector_Prod=Experience_Levels*Productivity_Params
    Surplus_High = np.zeros((Age_Periods,No_Sectors*(Production_Possibilities)))
    Surplus_Low =  0*Surplus_High
    Surplus_Check_High =0*Surplus_High
    Surplus_Check_Low=0*Surplus_High
    Terminal_Productivities = np.zeros((Production_Possibilities*No_Sectors))
    for ii in range(No_Sectors):
        Temp_Val = np.kron(Sector_XP_Gain,np.kron(Exper,Prod_Vals[ii*Productivity_Params:(ii+1)*Productivity_Params]))
        for jj in range(Sector_XP_Levels):
            Work_Val_Func_High[:,-1-(ii*Production_Possibilities+jj*(Sector_Prod+1))]=Unemp_High
            Work_Val_Func_Low[:,-1-(ii*Production_Possibilities+jj*(Sector_Prod+1))]=Unemp_Low
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
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)])
                                                                   
        Choice_TempSector2_High=np.argmax(
            (np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1],Production_Possibilities),
             Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities:2*Production_Possibilities]+0.00000000001,
             np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
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
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)])
                                                                   
        Choice_TempSector2_Low=np.argmax(
            (np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1],Production_Possibilities),
             Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities:2*Production_Possibilities]+0.00000000001,
             np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
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
  ##Non-Pecuniary
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
     
        
    Init_High,Init_Low,Ch,Cl  = Initial_Cohort_Fixed(Work_Val_Func_High,Work_Val_Func_Low,Cost_Vector,Distr_Init_High,
                         Distr_Init_Low,Production_Possibilities,No_Sectors,Grid_Points,Prob_Types,Prob_Draw)    
        
        
    return  (Worker_Transition_Matrices_High,Worker_Transition_Matrices_Low,
             Original_F_High_Transition_Matrices,
             Original_F_Low_Transition_Matrices,
             Wages_High,Wages_Low,Surplus_High,Surplus_Low,Firm_Val_Func_High,
             Firm_Val_Func_Low,Work_Val_Func_High,Work_Val_Func_Low,Price,Init_High,Init_Low)


def TransitionValFunc_TradeSCF(ss,Transition_Prices,Prob_Match_Worker_Transition,Prob_Match_Firm_Transition,
                         Age_Periods,No_Sectors,Production_Possibilities,Unemp_High,Unemp_Low,Skill_vals,
                         Prod_Vals,Exper,Switch_Cost,Switch_Cost1,Switch_Cost2,Prod_Change_Matrix,
                         Exper_Gain_Matrix,Sector_XP_Gain,Sector_XP_Gain_Matrix,
                         Productivity_Params,Experience_Levels,Sector_XP_Levels,Prod_Draw,Beta,
                         Switcher_Matrix,Switch_Vector,Fraction_Worker_Surplus,Cost_Vector,Distr_Init_High,Distr_Init_Low,Prob_Types,Prob_Draw,
                         Grid_Points,Non_Pec,SC_Mfg):
    
    Price = Transition_Prices[ss:ss+Age_Periods,:]
    Prob_Match_Worker=Prob_Match_Worker_Transition[ss:ss+Age_Periods-1,:]
    Prob_Match_Firm=Prob_Match_Firm_Transition[ss:ss+Age_Periods-1,:]

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
    Sector_Prod=Experience_Levels*Productivity_Params
    Surplus_High = np.zeros((Age_Periods,No_Sectors*(Production_Possibilities)))
    Surplus_Low =  0*Surplus_High
    Surplus_Check_High =0*Surplus_High
    Surplus_Check_Low=0*Surplus_High
    Terminal_Productivities = np.zeros((Production_Possibilities*No_Sectors))
    for ii in range(No_Sectors):
        Temp_Val = np.kron(Sector_XP_Gain,np.kron(Exper,Prod_Vals[ii*Productivity_Params:(ii+1)*Productivity_Params]))
        for jj in range(Sector_XP_Levels):
            Work_Val_Func_High[:,-1-(ii*Production_Possibilities+jj*(Sector_Prod+1))]=Unemp_High
            Work_Val_Func_Low[:,-1-(ii*Production_Possibilities+jj*(Sector_Prod+1))]=Unemp_Low
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
  ##Non-Pecuniary
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
     
        
    Init_High,Init_Low,Ch,Cl  = Initial_Cohort_Fixed(Work_Val_Func_High,Work_Val_Func_Low,Cost_Vector,Distr_Init_High,
                         Distr_Init_Low,Production_Possibilities,No_Sectors,Grid_Points,Prob_Types,Prob_Draw)    
        
        
    return  (Worker_Transition_Matrices_High,Worker_Transition_Matrices_Low,
             Original_F_High_Transition_Matrices,
             Original_F_Low_Transition_Matrices,
             Wages_High,Wages_Low,Surplus_High,Surplus_Low,Firm_Val_Func_High,
             Firm_Val_Func_Low,Work_Val_Func_High,Work_Val_Func_Low,Price,Init_High,Init_Low)
    
def TransitionValFunc_General(ss,Transition_Prices,Prob_Match_Worker_Transition,Prob_Match_Firm_Transition,
                         Age_Periods,No_Sectors,Production_Possibilities,Unemp_High,Unemp_Low,Skill_vals,
                         Prod_Vals,Exper,Switch_Cost0,Switch_Cost,Switch_Cost1,Switch_Cost2,Prod_Change_Matrix,
                         Exper_Gain_Matrix,Sector_XP_Gain,Sector_XP_Gain_Matrix,
                         Productivity_Params,Experience_Levels,Sector_XP_Levels,Prod_Draw,Beta,
                         Switcher_Matrix,Switch_Vector,Fraction_Worker_Surplus,Cost_Vector,Distr_Init_High,Distr_Init_Low,Prob_Types,Prob_Draw,
                         Grid_Points,Non_Pec,SC_Mfg):
    
    Price = Transition_Prices[ss:ss+Age_Periods,:]
    Prob_Match_Worker=Prob_Match_Worker_Transition[ss:ss+Age_Periods-1,:]
    Prob_Match_Firm=Prob_Match_Firm_Transition[ss:ss+Age_Periods-1,:]

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
    Sector_Prod=Experience_Levels*Productivity_Params
    Surplus_High = np.zeros((Age_Periods,No_Sectors*(Production_Possibilities)))
    Surplus_Low =  0*Surplus_High
    Surplus_Check_High =0*Surplus_High
    Surplus_Check_Low=0*Surplus_High
    Terminal_Productivities = np.zeros((Production_Possibilities*No_Sectors))
    for ii in range(No_Sectors):
        Temp_Val = np.kron(Sector_XP_Gain,np.kron(Exper,Prod_Vals[ii*Productivity_Params:(ii+1)*Productivity_Params]))
        for jj in range(Sector_XP_Levels):
            Work_Val_Func_High[:,-1-(ii*Production_Possibilities+jj*(Sector_Prod+1))]=Unemp_High
            Work_Val_Func_Low[:,-1-(ii*Production_Possibilities+jj*(Sector_Prod+1))]=Unemp_Low
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
            [np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
            Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities:2*Production_Possibilities],
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities-1]-SC_Mfg*Switch_Cost1,Production_Possibilities),
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)])
                                                                   
        Choice_TempSector2_High=np.argmax(
            (np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
             Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities:2*Production_Possibilities]+0.00000000001,
             np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities-1]-SC_Mfg*Switch_Cost1,Production_Possibilities),
             np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)),
             axis=0)                                                              
                                                                               
        Max_Unemp_Option_Sector2_High_B =np.maximum.reduce(
            [np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities), 
            Unemp_High+Unemp_Continuation_Value_High[index_current,2*Production_Possibilities:3*Production_Possibilities],
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)])
                                                                   
        Choice_TempSector2_High_B=np.argmax(
            (np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
             np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities),
             Unemp_High+Unemp_Continuation_Value_High[index_current,2*Production_Possibilities:3*Production_Possibilities]+0.00000000001,                 
             np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)),
             axis=0)                
                                                                                                                    
        Max_Unemp_Option_Sector3_High =np.maximum.reduce(
            [np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities),
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
            Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities:4*Production_Possibilities]])


        Choice_TempSector3_High=np.argmax(
            (np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
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
            [np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
            Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities:2*Production_Possibilities],
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities-1]-SC_Mfg*Switch_Cost1,Production_Possibilities),
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)])
                                                                   
        Choice_TempSector2_Low=np.argmax(
            (np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
             Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities:2*Production_Possibilities]+0.00000000001,
             np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities-1]-SC_Mfg*Switch_Cost1,Production_Possibilities),
             np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)),
             axis=0)                                                              
                                                                               
        Max_Unemp_Option_Sector2_Low_B =np.maximum.reduce(
            [np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities), 
            Unemp_Low+Unemp_Continuation_Value_Low[index_current,2*Production_Possibilities:3*Production_Possibilities],
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)])
                                                                   
        Choice_TempSector2_Low_B=np.argmax(
            (np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
             np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities),
             Unemp_Low+Unemp_Continuation_Value_Low[index_current,2*Production_Possibilities:3*Production_Possibilities]+0.00000000001,                 
             np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)),
             axis=0)                
                                                                                                                    
        Max_Unemp_Option_Sector3_Low =np.maximum.reduce(
            [np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities),
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
            Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities:4*Production_Possibilities]])


        Choice_TempSector3_Low=np.argmax(
            (np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
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
  ##Non-Pecuniary
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
     
        
    Init_High,Init_Low,Ch,Cl  = Initial_Cohort_Fixed(Work_Val_Func_High,Work_Val_Func_Low,Cost_Vector,Distr_Init_High,
                         Distr_Init_Low,Production_Possibilities,No_Sectors,Grid_Points,Prob_Types,Prob_Draw)    
        
        
    return  (Worker_Transition_Matrices_High,Worker_Transition_Matrices_Low,
             Original_F_High_Transition_Matrices,
             Original_F_Low_Transition_Matrices,
             Wages_High,Wages_Low,Surplus_High,Surplus_Low,Firm_Val_Func_High,
             Firm_Val_Func_Low,Work_Val_Func_High,Work_Val_Func_Low,Price,Init_High,Init_Low)
    
    
def TransitionValFunc_FullCSR(ss,Transition_Prices,Prob_Match_Worker_Transition,Prob_Match_Firm_Transition,
                         Age_Periods,No_Sectors,Production_Possibilities,Unemp_High,Unemp_Low,Skill_vals,
                         Prod_Vals,Exper,Swith_Cost0,Switch_Cost,Switch_Cost1,Switch_Cost2,Prod_Change_Matrix,
                         Exper_Gain_Matrix,Sector_XP_Gain,Sector_XP_Gain_Matrix,
                         Productivity_Params,Experience_Levels,Sector_XP_Levels,Prod_Draw,Beta,
                         Switcher_Matrix,Switch_Vector,Fraction_Worker_Surplus,Cost_Vector,Distr_Init_High,Distr_Init_Low,Prob_Types,Prob_Draw,
                         Grid_Points,Non_Pec,SC_Mfg):
    
    Price = Transition_Prices[ss:ss+Age_Periods,:]
    Prob_Match_Worker=Prob_Match_Worker_Transition[ss:ss+Age_Periods-1,:]
    Prob_Match_Firm=Prob_Match_Firm_Transition[ss:ss+Age_Periods-1,:]

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
    Sector_Prod=Experience_Levels*Productivity_Params
    Surplus_High = np.zeros((Age_Periods,No_Sectors*(Production_Possibilities)))
    Surplus_Low =  0*Surplus_High
    Surplus_Check_High =0*Surplus_High
    Surplus_Check_Low=0*Surplus_High
    Terminal_Productivities = np.zeros((Production_Possibilities*No_Sectors))
    for ii in range(No_Sectors):
        Temp_Val = np.kron(Sector_XP_Gain,np.kron(Exper,Prod_Vals[ii*Productivity_Params:(ii+1)*Productivity_Params]))
        for jj in range(Sector_XP_Levels):
            Work_Val_Func_High[:,-1-(ii*Production_Possibilities+jj*(Sector_Prod+1))]=Unemp_High
            Work_Val_Func_Low[:,-1-(ii*Production_Possibilities+jj*(Sector_Prod+1))]=Unemp_Low
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
            [np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
            Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities:2*Production_Possibilities],
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities-1]-SC_Mfg*Switch_Cost1,Production_Possibilities),
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)])
                                                                   
        Choice_TempSector2_High=np.argmax(
            (np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
             Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities:2*Production_Possibilities]+0.00000000001,
             np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities-1]-SC_Mfg*Switch_Cost1,Production_Possibilities),
             np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)),
             axis=0)                                                              
                                                                               
        Max_Unemp_Option_Sector2_High_B =np.maximum.reduce(
            [np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities), 
            Unemp_High+Unemp_Continuation_Value_High[index_current,2*Production_Possibilities:3*Production_Possibilities],
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)])
                                                                   
        Choice_TempSector2_High_B=np.argmax(
            (np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
             np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities),
             Unemp_High+Unemp_Continuation_Value_High[index_current,2*Production_Possibilities:3*Production_Possibilities]+0.00000000001,                 
             np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)),
             axis=0)                
                                                                                                                    
        Max_Unemp_Option_Sector3_High =np.maximum.reduce(
            [np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities),
            np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
            Unemp_High+Unemp_Continuation_Value_High[index_current,3*Production_Possibilities:4*Production_Possibilities]])


        Choice_TempSector3_High=np.argmax(
            (np.repeat(Unemp_High+Unemp_Continuation_Value_High[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
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
            [np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
            Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities:2*Production_Possibilities],
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities-1]-SC_Mfg*Switch_Cost1,Production_Possibilities),
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)])
                                                                   
        Choice_TempSector2_Low=np.argmax(
            (np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
             Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities:2*Production_Possibilities]+0.00000000001,
             np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities-1]-SC_Mfg*Switch_Cost1,Production_Possibilities),
             np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)),
             axis=0)                                                              
                                                                               
        Max_Unemp_Option_Sector2_Low_B =np.maximum.reduce(
            [np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities), 
            Unemp_Low+Unemp_Continuation_Value_Low[index_current,2*Production_Possibilities:3*Production_Possibilities],
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)])
                                                                   
        Choice_TempSector2_Low_B=np.argmax(
            (np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
             np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities),
             Unemp_Low+Unemp_Continuation_Value_Low[index_current,2*Production_Possibilities:3*Production_Possibilities]+0.00000000001,                 
             np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,4*Production_Possibilities-1]-Switch_Cost2,Production_Possibilities)),
             axis=0)                
                                                                                                                    
        Max_Unemp_Option_Sector3_Low =np.maximum.reduce(
            [np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,2*Production_Possibilities-1]-Switch_Cost,Production_Possibilities),
            np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities-1]-Switch_Cost1,Production_Possibilities),
            Unemp_Low+Unemp_Continuation_Value_Low[index_current,3*Production_Possibilities:4*Production_Possibilities]])


        Choice_TempSector3_Low=np.argmax(
            (np.repeat(Unemp_Low+Unemp_Continuation_Value_Low[index_current,Production_Possibilities-1]-Swith_Cost0,Production_Possibilities),
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
  ##Non-Pecuniary
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
     
        
    Init_High,Init_Low,Ch,Cl  = Initial_Cohort_Fixed(Work_Val_Func_High,Work_Val_Func_Low,Cost_Vector,Distr_Init_High,
                         Distr_Init_Low,Production_Possibilities,No_Sectors,Grid_Points,Prob_Types,Prob_Draw)   
    
    W_H=[]
    W_L=[]
    OF_H=[]
    OF_L=[]

    for ii in range(Age_Periods):
        W_H=np.append(W_H,csr_matrix(Worker_Transition_Matrices_High[ii]))
        W_L=np.append(W_L,csr_matrix(Worker_Transition_Matrices_Low[ii]))
        OF_H=np.append(OF_H,csr_matrix(Original_F_High_Transition_Matrices[ii]))
        OF_L=np.append(OF_L,csr_matrix(Original_F_Low_Transition_Matrices[ii]))
        
        
    return  (W_H, W_L,
             OF_H,
             OF_L,
             Wages_High,Wages_Low,Surplus_High,Surplus_Low,Firm_Val_Func_High,
             Firm_Val_Func_Low,Work_Val_Func_High,Work_Val_Func_Low,Price,Init_High,Init_Low)
