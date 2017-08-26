 

def CP_detector(time_series,N_train,N_test,range_w_n,using_stopping_criterion_flag,total_anomaly_cases_for_CP,print_results_flag):

   import numpy as np # importing required modules

   length_time_series = len(time_series)
   range_t_train = range(0,N_train)
   
   if (N_test == 0):
      range_t_test = range(0,length_time_series - len(range_t_train))
   else:
      if (N_test + N_train <= length_time_series):  
         range_t_test = range(0,N_test)
      else:
         raise NameError("Value of N_test is too large! End of time series reached. Choose a smaller value of N_test.")
          
       
   print("TRAINING THE MODEL BASED ON Z SCORES ------------------ \n")
   
   final_z_ensemble, mean_current_z_ensemble = gathering_different_CP_scores(time_series, range_t_train, range_w_n, using_stopping_criterion_flag)

   print("MODEL TRAINED! NOW TESTING TIME SERIES ---------------- \n")


   n_combinations, n_time = np.shape(final_z_ensemble[:,1:])

   mean_z_time = [] # This is a list that will contain the average z scores computed
                 # over all t for all values of w and n
   z_list_sample_without_nans = [] # List to store non-nan z scores

   for i_comb in range(0,n_combinations): 
      z_list_values_flatten = final_z_ensemble[i_comb,1:]
      z_list_values_flatten = np.asarray(z_list_values_flatten)  
      z_list_sample_without_nans = [ x for x in z_list_values_flatten if not np.isnan(x) ]     
      mean_z_time.append(np.mean(z_list_sample_without_nans))

   thresh_low = np.min(mean_z_time)
   thresh_upper = np.max(mean_z_time)


   CP_list = [] # This is a list that will contain the detected change points
   anomaly_counter = 0 # Anomaly counter is started as zero, it should count up
                    # to 'total_anomaly_cases_for_CP' to flag a CP

   for t_i in range_t_test:

      z_sample_list = [] # This is a list that will contain z scores computed for
                         # all w and n considered at a given time instant
                          
      print("Testing point t = %d \n" % (N_train + t_i))
         
      for i_comb in range(0,n_combinations): # i_comb stands for the i-th combination of w and n
         current_combination = final_z_ensemble[i_comb,0] # first column of the
         # final_z_ensemble array contains the combination of w and n values 
         current_w = current_combination[0]
         current_n = current_combination[1]
         z = computing_z_score(time_series,N_train + t_i,int(current_w),int(current_n))
         z_sample_list.append(z)

      
      if (sum(np.isnan(z_sample_list)) < len(z_sample_list)): # if the list of current z scores for the 
                                   # n and w values is not empty
          
         if ( (np.nanmean(z_sample_list) < thresh_low) or (np.nanmean(z_sample_list) > thresh_upper)):  
            anomaly_counter = anomaly_counter+1
         else:
            anomaly_counter = 0
         if (anomaly_counter == total_anomaly_cases_for_CP):    
            CP = N_train + t_i - total_anomaly_cases_for_CP
            print("Change point found! t = %d \n" % (CP))  
            CP_list.append(np.array([CP, current_w, current_n,np.nanmean(z_sample_list),thresh_low,thresh_upper]))                                
            print(np.nanmean(z_sample_list))
      
      
   if (len(CP_list) > 0):  
      number_CPs, size_array = np.shape(CP_list)
   else:
      number_CPs = 0
      
   if (print_results_flag == 1):  
          
      print("Summary Test Results: \n") 
      print("Number of change points detected: %d \n" % (number_CPs)) 
         
      if (number_CPs > 0):
         for i_CP in range(0,number_CPs):
            print("CP found at: %d, with range w = [2,%d] and range n = [2,%d], zTestValue = %f, Thresh test values: zThreshLow = %f and zThreshHigh = %f  \n" % (CP_list[i_CP][0],CP_list[i_CP][1],CP_list[i_CP][2],CP_list[i_CP][3],CP_list[i_CP][4],CP_list[i_CP][5])) 
       
                               
   return CP_list, final_z_ensemble, mean_current_z_ensemble 

         
         



def gathering_different_CP_scores(time_series,range_t, range_w_n, using_stopping_criterion_flag):
   
   import numpy as np # importing required modules
   
   # make it not count case with w=1 
   first_case_flag = 1
   mean_current_z_ensemble = []
   z_array_time = []
   z_ensemble = []
   model_stopped_flag = 0

   if (range_t == 0):
      range_t = range(0,np.size(time_series))

   if (range_w_n == 0):
      range_w_n = range(2,np.size(range_t)-1)
      print(range_w_n)

   current_w_n_values_to_test = np.array([[1,1]])
   iteration_counter = 0
      
   for i_w_n in range_w_n:
                   
      past_w_n_values_to_test = current_w_n_values_to_test+1
      #current_w_n_values_to_test = concatenate((array([[i_w_n,1]]),past_w_n_values_to_test),axis = 0)
      current_w_n_values_to_test = np.concatenate((np.array([[1,i_w_n]]),np.array([[i_w_n,1]]),past_w_n_values_to_test),axis = 0)
   
      # for computing the z-score, the first case corresponding to array([[1,i_w_n]])
      # we will skip, since it stands for using w=1
      for ii_w_n in range(1,current_w_n_values_to_test.shape[0]):      
      
         z_array_time = [] 
         z_array_time.append(list(current_w_n_values_to_test[ii_w_n,:]))
      
         for t_i in range(0,np.size(range_t)-1):
        
            t = range_t[t_i]  
            z_array_time.append(computing_z_score(time_series,t,current_w_n_values_to_test[ii_w_n,0],current_w_n_values_to_test[ii_w_n,1]))
  
         if (first_case_flag == 1):
            z_ensemble = z_array_time
            first_case_flag = 0
        
         elif (first_case_flag == 0):   
            z_ensemble = np.vstack((z_ensemble,z_array_time))
       
      z_values_to_compute_mean = z_ensemble[:,1:] # as the first element of z_ensemble represents the w and n information     
      current_z_list_values = []
      current_z_list_values = z_values_to_compute_mean.flatten()
      # Replacing nan's with zero to compute stopping criterion
      nan_mask = np.isnan(list(current_z_list_values)) # list positions with nan's
      current_z_list_values[nan_mask] = 0
      
#      current_z_list_vaues[nan_mask] = 0 # nans replaced with zero                                
      mean_current_z_ensemble.append(np.nanmean(current_z_list_values))     
      current_length_mean_sequence = len(mean_current_z_ensemble)-1


      # We use i_w_n > 2 because i_w_n starts at 2
      if ((i_w_n > 2) and (current_length_mean_sequence > 2) and (using_stopping_criterion_flag == 1)):

      
         if (mean_current_z_ensemble[current_length_mean_sequence] < mean_current_z_ensemble[current_length_mean_sequence-1]):
          
            final_z_ensemble = z_ensemble_past_iteration                   
            model_stopped_flag = 1 # stopping criterion has been fulfilled
            print("Model creation stopped at w = %d and n = %d\n" % (final_z_ensemble[i_w_n,0][0],final_z_ensemble[i_w_n,0][1]))
            break
        
         else:
             
            z_ensemble_past_iteration = z_ensemble 
            
            print(" %d percent complete of the max. number of iterations that may be required. But stop criterion may be achieved at next iteration \n" % (100*iteration_counter/np.size(range_w_n)))
            
    
      else:
          
         z_ensemble_past_iteration = z_ensemble 
         
         if (using_stopping_criterion_flag == 1):
            print(" %d percent complete of the max. number of iterations that may be required. But stop criterion may be achieved at next iteration \n" % (100*iteration_counter/np.size(range_w_n)))
         else:
            print(" %d percent complete of the required number of iterations. \n" % (100*iteration_counter/np.size(range_w_n)))
            
      iteration_counter = iteration_counter + 1  
  
                       
   if (model_stopped_flag == 0): 
   # which means that all values of z are needed to be used for bulding the model
   
      final_z_ensemble = z_ensemble_past_iteration    

         
   final_z_ensemble_only_z_values = final_z_ensemble[:,1:]
        

   return final_z_ensemble, mean_current_z_ensemble
    




   
   
def computing_z_score(time_series,t,w,n):

   import numpy as np
   
   # Compute trajectory matrix  
   
   T = np.size(time_series) # "T" stands for the number of points in the time series
   W1 = w+n-1 # By definition, the trajectory matrix is defined over W=w+n-1 elements from x(t-1) to x[t-(w+n-1)] 
   W2 = w+n-2
   B = np.zeros((w,n))# Initialize matrix for the "past" of instant "t"
   F = np.zeros((w,n)) # Initialize matrix for the "future" of instant "t"      
           
   if ((t >= W1) and ((t+W2) < T)):
    
      for i in range(1,n+1):
         # generating vector containing representative patterns of the "past"
         b = np.transpose(time_series[np.array(range(t-i,t-w-i,-1))]) 
         # generating vector containing representative patterns of the "future"
         f = np.transpose(time_series[np.array(range(t+i-1,t+w+i-1))]) 
        
         B[:,i-1] = b # storing these vectors in the B matrix
         F[:,i-1] = f # storing these vectors in the B matrix 

      B = B[ ::-1,::-1] # now B matrix need to be put in the reverse order
   
   else:
      B = np.float('NaN')
      F = np.float('NaN')
      #print('Too close to the sides of the time series to create trajectory matrices. More time points needed.')

   
   # Checking if we had enough time points to compute the trajectory matrix
   if ( (np.any(np.isnan(B))) or (np.any(np.isnan(F))) ):
      
      z = float('NaN') # Could not compute z score, more points needed  

   else:
      U_f, s_f, V_f = np.linalg.linalg.svd(F) # SVD for points in the "future" of instant "t"
      U_b, s_b, V_b = np.linalg.linalg.svd(B) # SVD for points in the "past" of instant "t"
      
      # We compute the change point score "z" only by using the first singular
      # vector of the future and past matrices
      
      z = 1 - (np.dot(U_f[:,0],U_b[:,0])/(np.linalg.norm(U_f[:,0])*np.linalg.norm(U_b[:,0])))**2
     
              
   return z
   
   

   


   
   
   
   