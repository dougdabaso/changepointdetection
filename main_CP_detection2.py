#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 18:33:54 2017

@author: douglas
"""


import matplotlib.pyplot as plt
import operator as op
import csv
from numpy.linalg.linalg import svd
from numpy.linalg.linalg import norm
from numpy import *


length_time_series = 300
CP = 150 # TRUE CHANGE POINT
gain = 0.08
abrupt_change_trame = gain*hstack([zeros(CP),ones(length_time_series-CP)])
#abrupt_change_trame = gain*hstack([zeros(CP),ones(CP),zeros(CP)])
additive_noise = random.normal(loc=0,scale=0.05,size=length_time_series)
time_series = abrupt_change_trame + additive_noise
#time_series = additive_noise
plt.plot(time_series)


from CP_detection_functions_most_recent import *

N_train = 100 # Number of points to train the model
N_test = 0 # Number of points to test
range_w_n = range(2,30) # Total range of w and n values to consider to build the model
using_stopping_criterion_flag = 1 # Flag to consider or not stopping criterion (based on unimodality assumption)
total_anomaly_cases_for_CP = 5 # Number of anomaly points above which a change can be assumed (i.e. rule out outliers)
print_results_flag = 1 # Flag to print the results  

CP_list, final_z_ensemble, mean_current_z_ensemble = CP_detector(time_series,N_train,N_test,range_w_n,using_stopping_criterion_flag,total_anomaly_cases_for_CP,print_results_flag)   
  

 computing_z_score(time_series,0,2,2)






   
   