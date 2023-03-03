#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 19:49:59 2022

"""

"""
This code compares COMM COST for 3 algorithms for the MovieLens dataset:
    1. Periodic communication with H=1 as the period (communication in every time step)
    2. Periodic communication with H=5
    3. Periodic communication with H=10
    4. FedElim algorithm.
"""

import numpy as np
import math
from progressbar import ProgressBar
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
    if (exp_idx == 3):
        return isPowerOfTwo(n)
    else:
        return (n % H_vals[exp_idx] == 0)
    


df = pd.read_csv("movielens-with-genres-and-countries.csv", usecols=['rating', 'genre', 'country'])
# countries = ['USA', 'France', 'UK'] # country = client
countries = df['country'].to_numpy(dtype = str)
countries = np.unique(countries)
M = len(countries)
# genres = ['Action', 'Sci-Fi', 'Romance', 'Comedy'] # genre = arm.
genres = df['genre'].to_numpy(dtype = str)
genres = np.unique(genres)
K = len(genres) 

Mu = np.zeros((M, K))   # problem instance
S = []   # set of the local best arms and the global best arm
m_vals_to_consider = [] # set of client indices to consider after deleting rows which have non-unique best arm.
new_countries = [] # new country names after eliminating the ones which have non-unique best arm.
Mu_copy = []
array_ratings_matrix = [] # matrix of array ratings corresponding to each client and genre pair.
pbar = ProgressBar()

for m in pbar(range(M)):
    array_ratings = []
    for k in range(K):
        genre = genres[k]
        country = countries[m]
        df1 = df[(df['genre'].to_numpy(dtype=str) == genre) \
                  & (df['country'].to_numpy(dtype=str) == country)]
        rewards = df1['rating'].to_numpy(dtype = float)
        
        if (len(rewards) == 0):
            Mu[m][k] = 0
            array_ratings.append(np.zeros(10))
        else:
            Mu[m][k] = np.mean(rewards)
            array_ratings.append(rewards)
    
    # check if m corresponds to a client with non-unique best arm
    mean_vectors_of_client_m_arms = Mu[m]     
    km_star = np.where(mean_vectors_of_client_m_arms == max(mean_vectors_of_client_m_arms))
    
    if (len(km_star[0]) == 1): # unique best arm; this 'm' should be considered.
        m_vals_to_consider.append(m)
        new_countries.append(country)
        
        # copy the row corresponding to 'm' to Mu_copy
        if (len(Mu_copy) == 0):
            Mu_copy = mean_vectors_of_client_m_arms
        else:
            Mu_copy = np.vstack([Mu_copy, np.transpose(mean_vectors_of_client_m_arms)])
        
        # preserve the array_ratings corresponding to 'm'
        if (len(array_ratings_matrix) == 0):
            array_ratings_matrix = array_ratings
        else:
            array_ratings_matrix = np.vstack([array_ratings_matrix, np.transpose(array_ratings)])

# M is now the number of 'm' indices retained after elimination of those indices having non-unique best arm.
# Mu_copy = np.delete(Mu_copy, [9, 23, 39, 44], axis=0)
M, N = np.shape(Mu_copy)
Mu = Mu_copy

# update array_ratings_matrix
# array_ratings_matrix = np.delete(array_ratings_matrix, [9, 23, 39, 44], axis=0)


# update local best arms
for m in range(M):
    km_star = np.argwhere(Mu[m] == np.max(Mu[m]))
    S.append(km_star[0][0])

    
# determine Delta_km
Delta_km = np.zeros((M,K))
for m in range(M):
    client_m_delta_km_values = []
    for k in range(K):
        if (k == S[m]):
            continue
        else:
            Delta_km[m][k] = Mu[m][int(S[m])] - Mu[m][k]
            client_m_delta_km_values.append(Delta_km[m][k])
            
    Delta_km[m][S[m]] = min(client_m_delta_km_values)

# eliminate all rows in Delta_km which have Delta_km values less than 0.1
# also delete the corresponding rows from Mu
# also delete the corresponding rows from array_ratings_matrix
# also delete the best arm values in S at the deleted indices

indices_to_delete = np.unique(np.where(Delta_km < 0.1)[0])
indices_to_delete = np.setdiff1d(indices_to_delete, [54]) # excluding USA (index 54) as it makes the problem more interesting.
Mu = np.delete(Mu, indices_to_delete, axis=0)
Delta_km = np.delete(Delta_km, indices_to_delete, axis=0)
array_ratings_matrix = np.delete(array_ratings_matrix, indices_to_delete, axis=0)
indices_retained = np.setdiff1d(range(M), indices_to_delete)
S = [S[m] for m in indices_retained]

# new M value
M, N = np.shape(array_ratings_matrix)

# append the global best arm to S
global_means = np.zeros(K)
for k in range(K):
    mu_k = np.mean([Mu[m1][k] for m1 in range(M)])
    global_means[k] = mu_k

k_star = np.where(global_means == np.amax(global_means))
S = np.append(S, k_star[0][0])
    
# Determine the Delta_k values
Delta_k = np.zeros(K)
delta_k_values = []
for k in range(K):
    if (k == S[M]):
        continue
    else:
        Delta_k[k] = global_means[S[M]] - global_means[k]
        delta_k_values.append(Delta_k[k])
        
Delta_k[S[M]] = min(delta_k_values)

C = 10
delta_vals = [np.exp(-1*i) for i in np.arange(1.0, 6.0, 2.0)]  # error probability
H_vals = [1, 5, 10]     # H vals for periodic communication
comm_cost = np.zeros((4,len(delta_vals)))  # vector of communication costs for each delta
err_values_comm_cost = np.zeros((4, len(delta_vals)))  # vector of error bars in comm cost computations

pbar = ProgressBar()

for exp_idx in pbar(range(0,4)): # 0 indicates periodic comm with H=1, 1 indicates H=5, 2 indicates H=10, and 3 indicates FLSEA
    comm_cost_per_exp = []  
    err_comm_cost_per_exp = []
    for delta in delta_vals:
        comm_cost_vector = []
        total_cost_vector = []
        error_prob_indicator = []   # array of 1s and 0s whose mean gives empirical error probability 
        num_trials = 10**2  # number of independent runs over which averaging is done.
        run_num = 1
        
        # keep running the algorithm until the empirical error probability is >= 1-delta
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
            
            # S_lm=A for all clients to begin with.
            S_lm = []
            for i in range(M):
                S_lm.append(A)
                
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
                            
                            movie_ratings = array_ratings_matrix[m][S_m[k]]
                            rand_movie_indx = rd.randint(0, len(movie_ratings)-1)
                            X = movie_ratings[rand_movie_indx] # random sample from all the movie ratings corresponding to armID S_m[k] and clientID m
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
                
        comm_cost_per_exp.append(stat.mean(np.log(comm_cost_vector))) # for each value of delta
        err_comm_cost_per_exp.append(np.sqrt(stat.variance(np.log(comm_cost_vector))))  # for each value of delta
        
    comm_cost[exp_idx] = comm_cost_per_exp # for each exp_idx
    err_values_comm_cost[exp_idx] = err_comm_cost_per_exp   # for each exp_idx

df = pd.DataFrame({'delta': delta_vals,
                   'comm-cost-H-is-1': comm_cost[0],
                   'comm-cost-H-is-5': comm_cost[1],
                   'comm-cost-H-is-10': comm_cost[2],
                   'comm-cost-FedElim': comm_cost[3],
                   'error-comm-cost-H-is-1': err_values_comm_cost[0],
                   'error-comm-cost-H-is-5': err_values_comm_cost[1],
                   'error-comm-cost-H-is-10': err_values_comm_cost[2],
                   'error-comm-cost-FedElim': err_values_comm_cost[3]}, index=None)
df.to_csv('FedElim-and-periodic-comm-cost-comparison-movielens.csv', index=None)
    