import pandas as pd
import numpy as np

"""
    ------------------------------------
            USER DATA
    ------------------------------------
"""
user_cols = ["UserID", "Gender", "Age", "Occupation", "Zip-Code"]
user = pd.read_csv("./ml-1m/users.dat", delimiter="::",
                   engine='python', names=user_cols).set_index("UserID")
print(user.head())

"""
    ------------------------------------
            MOVIE DATA
    ------------------------------------
"""
movie_cols = ["MovieID", "Title", "Genre"]
movie = pd.read_csv("./ml-1m/movies.dat", delimiter="::",
                    engine='python', names=movie_cols).set_index("MovieID")
print(movie.head())

"""
    ------------------------------------
            RATINGS DATA
    ------------------------------------
"""
ratings_cols = ["UserID", "MovieID", "Rating", "Timestamp"]
raating = pd.read_csv("./ml-1m/ratings.dat", delimiter="::",
                      engine='python', names=movie_cols)
print(raating.head())
