import pandas as pd
import numpy as np
import pickle

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
            RATINGS DATA
    ------------------------------------
"""
ratings_cols = ["UserID", "MovieID", "Rating", "Timestamp"]
rating = pd.read_csv("./ml-1m/ratings.dat", delimiter="::",
                     engine='python', names=ratings_cols)
