# module for common data processing

import pandas as pd
import pickle


def movie_index():
    '''
        Function to return movie details
    '''

    try:
        with open('./recommenders/training_data/movie_index', 'rb') as infile:
            movie_index = pickle.load(infile)
            return movie_index
    except EnvironmentError:
        try:
            movie_index = pd.read_csv('./recommenders/csv/movies.csv', index_col = 'movieId')
            return movie_index
        except EnvironmentError:
            print('Error reading csv file')
            print(EnvironmentError)
            return None