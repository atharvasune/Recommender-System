
import pandas as pd
import numpy as np
import pickle

"""
    ------------------------------------
            MOVIE DATA
    ------------------------------------
"""
try:
    with open("../../pickle/movie", 'rb') as infile:
        movie = pickle.load(infile)
except EnvironmentError:
    movie_cols = ["MovieID", "Title", "Genre"]
    temp = pd.read_csv("../../ml-1m/movies.dat", delimiter="::",
                       engine='python', names=movie_cols)
    # --------------------------------------------
    #     BREAK TITLE INTO YEAR AND NAME
    # --------------------------------------------
    title = temp['Title']
    data = []
    for i in title:
        data.append([i[:-7], int(i[-5:-1])])
    title_year_df = pd.DataFrame(data, columns=['Title', 'Year'])
    title_year_df.reset_index(drop=True, inplace=True)
    # --------------------------------------------

    # --------------------------------------------
    #    CREATING THE ONE HOT ENCODING ON GENRE
    # --------------------------------------------
    temp2 = []
    for i in temp['Genre']:
        temp2.append(i.split('|'))
    genre = {}
    for i, val in enumerate(list(set(x for l in temp2 for x in l))):
        genre[val] = i
    for i in temp2:
        for j in range(len(i)):
            i[j] = genre[i[j]]

    one_hot = np.zeros((len(title), 18), dtype='int')
    for i, lis in enumerate(temp2):
        for j, val in enumerate(lis):
            one_hot[i][val] = 1

    one_hot_df = pd.DataFrame(data=one_hot, columns=genre.keys())
    one_hot_df.reset_index(drop=True, inplace=True)
    # --------------------------------------------
    # --------------------------------------------
    #     CREATE MOVIE DATAFRAME AND STORE
    # --------------------------------------------
    movie = pd.concat([title_year_df, one_hot_df], sort=False, axis=1)
    try:
        with open('./pickle/movie', 'wb') as outfile:
            pickle.dump(movie, outfile)
    except EnvironmentError as error:
        print("Error {}".format(error))

print(movie.head())
