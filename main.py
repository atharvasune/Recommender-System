import pandas as pd

# import all classes from recommenders directory
from recommenders.collaborative import Collaborative
from recommenders.collab_with_baseline import CollaborativeWB
from recommenders.data import *

def main():

	'''
		Initialize objects for all classes (or maybe make static methods);
		and call functions to
		(1) get recommendations
		(2) calculate RMSE
		(3) calculate MAE
		(4) execution time
		for each of the recommender types
	'''
	# get movie details' table
	movie_ind = movie_index()

	# input user id
	user_id = int(input('Enter user id \n'))

	# recommendations using collaborative filtering
	cb = Collaborative()
	movie_list = cb.get_recommendation(user_id)
	movie_list = pd.merge(movie_list, movie_ind, on = 'movieId', how = 'inner')
	print('Recommendations using collaborative filtering for the user are \n', movie_list)
	print('RMSE is\n', cb.get_rmse())
	print('MAE is\n', cb.get_mae())

	# recommendations using collaborative filtering with baseline
	cwb = CollaborativeWB()
	movie_list = cwb.get_recommendation(user_id)
	movie_list = pd.merge(movie_list, movie_ind, on = 'movieId', how = 'inner')
	print('Recommendations using collaborative filtering with baseline for the user are \n', movie_list)
	print('RMSE is\n', cwb.get_rmse())
	print('MAE is\n', cwb.get_mae())


if __name__ == '__main__':
    main()