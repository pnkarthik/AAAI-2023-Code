#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:58:43 2022


"""

"""
This code provides a comparison of the total cost under FedElim when C=10,100
and the total cost under FedElim0 (when C=0).
FedElim0 is a version of FedElim which communicates in every round.

The code controls the different scenarios with a variable called exp_idx
exp_idx = 0 means C=0 (FedElim0, every time instant communication)
exp_idx = 1 means C=0 (FedElim, exponentially sparse communication)
exp_idx = 2 means C=10
exp_idx = 3 means C=100
"""

import numpy as np
from progressbar import ProgressBar
import math
import statistics as stat
import pandas as pd
import random as rd

def Log2(x):
    if x == 0:
        return False;
 
    return (math.log10(x) /
            math.log10(2));

def isPowerOfTwo(n):
    return (math.ceil(Log2(n)) ==
            math.floor(Log2(n)));

def conditionForn(n, exp_idx):
    if (exp_idx == 0):
        return True
    else:
        return isPowerOfTwo(n)

M = 3   # number of clients
K = 4   # number of arms at each client
Mu = [[0.9, 0.1, 0.1, 0.5], 
     [0.1, 0.9, 0.1, 0.5], 
     [0.1, 0.1, 0.9, 0.5]]  # problem instance; entry in (m,k) position cprresponds to the mean of arm k of client m
S = [0,1,2,3]   # the actual local and global best arms in the problem instance. The indexing of the arms is from 0. 
sigma = 1 # common variance parameter of the arms
C_vals = [0, 0, 10, 100]   # values of comm cost

delta_vals = [np.exp(-1*i) for i in np.arange(1.0, 6.0, 2.0)]  # error probability

total_cost = np.zeros((len(C_vals),len(delta_vals)))  # vector of total costs for each delta
err_values_total_cost = np.zeros((len(C_vals), len(delta_vals))) # vector of error bars in total cost computations

pbar = ProgressBar()

for exp_idx in pbar(range(len(C_vals))): # 0 indicates C=0 (FedElim0), 1 indicates C=0 (FedElim), 2 indicates C=10 (FedElim), and 3 indicates C=100 (FedElim)
    C = C_vals[exp_idx]
    total_cost_per_exp = []
    err_total_cost_per_exp = []
    for delta in delta_vals:
        arm_pulls_vector = []
        comm_cost_vector = []
        total_cost_vector = []
        error_prob_vector = []
        num_trials = 10**2  # number of independent trials over which averaging will be done.
        run_num = 0
        
        # keep running the algorithm until run_num <= num_trials.
        while run_num <= num_trials:
            
            rd.seed(run_num)
            run_num += 1    # number of times FLSEA runs
            
            # Initial values to be passed to FLSEA
            n = 0
            arm_pulls = 0   # total number of arm pulls required to declare the local and global best arms with prob >= 1-delta
            communication_cost = 0  # total communication cost for a given delta
            empirical_local_means = np.zeros((M,K))
            empirical_global_means = np.zeros((1,K))
            A = [i for i in range(K)]   # set of arms
            S_lm = [A,A,A]     # for all m, S_lm = [K] to begin with 
            S_g = np.array([i for i in range(K)]) # S_g = [K] to begin with
            local_best_arms = np.zeros(M, dtype=int)   # records the local best arms output by FLSEA
            run = 1     # run = true
            
            # FLSEA runs as long as run = true
            while (run == 1):
                
                n += 1
                
                # local arm calculations at the clients
                for m in range(M):
                    
                    # S_m = union(S_lm, S_g)
                    S_m = np.array(list(set(S_lm[m]) | set(S_g)))     # arms that client m must pull
                    
                    if (len(S_m) > 1):
                        
                        # pull each arm in S_m once and update its empirical mean
                        for k in range(len(S_m)):
                            
                            mu_km = Mu[m][S_m[k]]   # mean of arm k of client m
                            X = np.random.normal(mu_km, sigma)  # Gaussian sample from the arm
                            empirical_local_means[m][S_m[k]] = X/n + (((n-1)/n) * empirical_local_means[m][S_m[k]])
                            arm_pulls += 1
                        
                    if (len(S_lm[m]) > 1):
                        
                        # eliminate the local inactive arms of client m
                        mu_hat_star_m = max([empirical_local_means[m][S_lm[m][k]] for k in range(len(S_lm[m]))])
                        alpha_l = np.sqrt(2 * np.log(8 * K * M * n**2 / delta) / n)
                        S_lm_inactive = []
                        for k in range(len(S_lm[m])):
                            if (mu_hat_star_m - empirical_local_means[m][S_lm[m][k]] > (2 * alpha_l)):
                                S_lm_inactive.append(S_lm[m][k])
                        S_lm[m] = np.setdiff1d(S_lm[m], S_lm_inactive) # eliminate inactive arms from S_lm 
                                
                    # local best arm of client m identified
                    if (len (S_lm[m]) == 1):
                        local_best_arm_of_client_m = S_lm[m][0]
                        local_best_arms[m] = local_best_arm_of_client_m
                        S_lm[m] = np.setdiff1d(S_lm[m], local_best_arm_of_client_m)
                    
                
                if (len(S_g) > 1 and conditionForn(n, exp_idx)):
                    
                    communication_cost += C * M * len(S_g)
                    # communication between the clients and server; global inactive arms elimination
                    for k in range(len(S_g)):
                        empirical_global_means[0][S_g[k]] = stat.mean([empirical_local_means[m1][S_g[k]] for m1 in range(M)])
                        
                    mu_star = max([empirical_global_means[0][S_g[k]] for k in range(len(S_g))])
                    alpha_g = np.sqrt(2 * np.log(8 * K * n**2 / delta) / (M * n))
                    
                    # eliminate global inactive arms
                    S_g_inactive = []
                    for k in range(len(S_g)):
                        if (mu_star - empirical_global_means[0][S_g[k]] > (2 * alpha_g)):
                            S_g_inactive.append(S_g[k])
                    S_g = np.setdiff1d(S_g, S_g_inactive) # eliminate inactive arms from S_g
                    
                    
                    
                # global best arm identified
                if (len(S_g) == 1):
                    global_best_arm = S_g[0]
                    S_g = np.setdiff1d(S_g, global_best_arm)
                
                if (len(S_g) == 0):
                    no_of_arms = 0
                    for m in range(M):
                        if (len(S_lm[m]) == 0):
                            no_of_arms += 1
                    if (no_of_arms == M):
                        run = 0
            
            comm_cost_vector.append(communication_cost)   # for each run_num      
            arm_pulls_vector.append(arm_pulls)
            S_hat = np.append(local_best_arms, global_best_arm)
            if ((S_hat == S).all()):
                error_prob_vector.append(0)
            else:
                error_prob_vector.append(1)
            
        total_cost_per_exp.append(int(stat.mean(np.log(np.add(comm_cost_vector, arm_pulls_vector))))) # for each value of delta
        err_total_cost_per_exp.append(np.sqrt(stat.variance(np.log(np.add(comm_cost_vector, arm_pulls_vector))))) # for each value of delta
        
    total_cost[exp_idx] =  total_cost_per_exp   # for each exp_idx
    err_values_total_cost[exp_idx] = err_total_cost_per_exp    # for each exp_idx

df = pd.DataFrame({'delta': delta_vals,
                   'total-cost-C=0-FedElim0': total_cost[0],
                   'total-cost-C=0-FedElim': total_cost[1],
                   'total-cost-C=10-FedElim': total_cost[2],
                   'total-cost-C=100-FedElim': total_cost[3],
                   'error-bar-total-cost-C=0-FedElim0': err_values_total_cost[0],
                   'error-bar-total-cost-C=0-FedElim': err_values_total_cost[1],
                   'error-bar-total-cost-C=10-FedElim': err_values_total_cost[2],
                   'error-bar-total-cost-C=100-FedElim': err_values_total_cost[3]}, index=None)
df.to_csv('total-cost-varying-C.csv', index=None)

