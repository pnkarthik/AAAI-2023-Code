#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:46:19 2022

@author: karthikpn
"""

"""
This code computes the Delta_km and Delta_k values for the MovieLens dataset.
"""

import pandas as pd
import numpy as np
from progressbar import ProgressBar
# import matplotlib.pyplot as plt

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
indices_to_delete = np.setdiff1d(indices_to_delete, [54])
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

# further elimination of all rows from

# for m in range(M):
#     for k in range(K):
#         fig, axs = plt.subplots()
#         axs.hist(array_ratings_matrix[m][k])

# total_no_of_data_points = 0
# all_movies = []
# df_copy = pd.read_csv("movielens-with-genres-and-countries.csv", usecols=['rating', 'genre', 'country', 'movieID'])
# for m in range(M):
#     country = new_countries[m]
#     df1 = df_copy[df_copy['country'].to_numpy(dtype=str) == country]
#     movies = np.unique(df1['movieID'].to_numpy(dtype=str))
#     if (len(all_movies) == 0):
#         all_movies = movies
#     else:
#         all_movies = np.append(all_movies, movies)
#     total_no_of_data_points += len(df1)

# country = 'USA'
# df1 = df_copy[df_copy['country'].to_numpy(dtype=str) == country]
# movies = np.unique(df1['movieID'].to_numpy(dtype=str))
# all_movies = np.append(all_movies, movies)
# total_no_of_data_points += len(df1)
# all_movies = np.unique(all_movies)
