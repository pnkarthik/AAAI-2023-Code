#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 20:15:36 2022

@author: karthikpn
"""

"""
This code groups the movielens dataset according to country (client) and genres (arms).
The observation from an arm is the "rating" of a movie of a certain genre shot in a certain country 
"""



import pandas as pd
import numpy as np

df_genres = pd.read_table("movie-genres.dat") # contains movieID and genre
df_countries = pd.read_table("movie-countries.dat") # contains movieID and countries.
df = pd.read_table("user-ratedmovies.dat", usecols=['userID','movieID', 'rating']) # contains movieID, userID, rating among other items

genres = df_genres['genre'].to_numpy(dtype = str)
movie_ids_genres = df_genres['movieID'].to_numpy(dtype = str)
countries = df_countries['country'].to_numpy(dtype = str)
movie_ids_countries = df_countries['movieID'].to_numpy(dtype = str)

movie_ids = df['movieID'].to_numpy(dtype = str)
ratings = df['rating'].to_numpy(dtype = float)

# append genre to df

new_user_ids = []
new_movie_ids = []
new_ratings = []
new_genres = []
new_countries = []
for i in range(len(df)):
    movieID = movie_ids[i]
    rating = ratings[i]
    indx_genre = np.where(movie_ids_genres == movieID)
    indx_country = np.where(movie_ids_countries == movieID)
    
    if (len(indx_genre[0]) >= 1):
        for j in range(len(indx_genre[0])):
            new_movie_ids.append(movieID)
            new_ratings.append(rating)
            genre = genres[indx_genre[0][j]]
            country = countries[indx_country[0][0]]
            new_genres.append(genre)
            new_countries.append(country)
    else:
        continue
    
df1 = pd.DataFrame({'movieID': new_movie_ids,
                    'rating': new_ratings,
                    'genre': new_genres, 
                    'country': new_countries}, index=None)    
df1.to_csv('movielens-with-genres-and-countries.csv')

# some values that Vincent asked for
df1 = pd.read_csv('movielens-with-genres-and-countries.csv')
genres = ['Action', 'Sci-Fi', 'Romance', 'Comedy'] # genre = arm.
total_movies_for_genres = 0
for g in genres:
    df2 = df1[df1['genre'].to_numpy(dtype=str) == g]
    movies = df2['movieID'].to_numpy(dtype = str)
    movies = np.unique(movies)
    total_movies_for_genres += len(movies)




