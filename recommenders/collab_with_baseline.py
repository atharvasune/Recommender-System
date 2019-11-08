# module for item-item collaborative filtering using baseline approach

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

class CollaborativeWB():

    def __init__(self):
        pass

    def get_recommendation(self, user_id):
        '''
            Function to return 10 movie recommendations for the given user_id
        '''

        m_u_c = self.get_trained_data()
        m_u_train, m_u_test = self.get_train_test()
        m_u_train = m_u_train.fillna(0)

        predicted = m_u_c.iloc[:, user_id - 1]
        training = m_u_train.iloc[:, user_id - 1]
        training = training.isnull().astype('int')

        predicted.mul(training)
        predicted = pd.DataFrame(predicted.sort_values(ascending = False)).reset_index()

        predicted.columns = ['movieId', 'predicted rating']
        return predicted[: 10]

    def get_trained_data(self):
        '''
            Returns various training data, as per the algorithm, after optimum preprocessing
            Returns data from pickle files if available, otherwise generates, stores pickle files and then
            returns
        '''

        # m_u_c stores movie-user ratings matrix predicted by item -item collaborative on training data
        try:
            with open('recommenders/training_data/m_u_c_baseline', 'rb') as infile:
                m_u_c = pickle.load(infile)

        except EnvironmentError:
            m_u_train, m_u_test = self.get_train_test()            
            m_u_c = m_u_train

            # row wise mean for all rows
            row_mean = m_u_c.mean(axis = 1)           

            # replace NaN with row means
            m_u_c = m_u_c.T.fillna(row_mean).T

            m_u_c = m_u_c.subtract(row_mean, axis = 0)

            # create mean centered ratings matrix
            # generate similarity matrix
            similarity = cosine_similarity(m_u_c)
            similarity = pd.DataFrame(similarity, index = m_u_train.index, columns = m_u_train.index)
            np.fill_diagonal(similarity.values, 1)

            sim_sum = (similarity.abs()).dot(pd.notna(m_u_train).astype('int'))

            m_u_c = similarity.dot(m_u_c.fillna(0))
            m_u_c = m_u_c.div(sim_sum).fillna(0)
            baseline = self.get_baseline()

            m_u_c = m_u_c.add(baseline).clip(1, 5)

            try:
                with open('recommenders/training_data/m_u_c_baseline', 'wb') as outfile:
                    pickle.dump(m_u_c, outfile)

            except EnvironmentError:
                print(EnvironmentError)
                return None

        return m_u_c        


    def get_train_test(self):
        '''
            Generate train and test matrices
        '''
        # load train, test from persistant storage
        try:
            with open('recommenders/training_data/m_u', 'rb') as infile:
                m_u_train = pickle.load(infile)
            with open('recommenders/testing_data/m_u', 'rb') as infile:
                m_u_test = pickle.load(infile)
            return m_u_train, m_u_test  
            
        except EnvironmentError:
            # if error, generate, store and then return
            try:
                m_u_raw = pd.read_csv('recommenders/csv/ratings.csv')
                m_u_train, m_u_test_list = train_test_split(m_u_raw, test_size = 0.2)
                m_u_train = pd.pivot_table(m_u_train, values = 'rating', index = 'movieId', columns = 'userId')
                m_u_test = pd.pivot_table(m_u_test_list, values = 'rating', index = 'movieId', columns = 'userId')

                # dump m_u and m_u_test
                try:
                    with open('recommenders/training_data/m_u', 'wb') as outfile:
                        pickle.dump(m_u_train, outfile)
                    with open('recommenders/testing_data/m_u', 'wb') as outfile:
                        pickle.dump(m_u_test, outfile)
                    with open('recommenders/testing_data/m_u_list', 'wb') as outfile:
                        pickle.dump(m_u_test_list, outfile)
                    return m_u_train, m_u_test

                except EnvironmentError:
                    print(EnvironmentError)
                    return None
                                 

            except EnvironmentError:
                print(EnvironmentError) 
                return None    


    def get_baseline(self):
        '''
            Return baseline rating prediction matrix
        '''
        
        try:
            # return from persistant storage
            with open ('recommenders/training_data/m_u_baseline', 'rb') as infile:
                baseline = pickle.load(infile)
            return baseline

        except EnvironmentError:
            # get training dataset
            m_u_train, m_u_test = self.get_train_test()

            # mean rating for every movie
            movie_mean = m_u_train.mean(axis = 1)
            # gloabal average rating
            universal_mean = movie_mean.mean(axis = 0)

            # mean rating given by every user
            user_mean_dev = (m_u_train.mean(axis = 0)).sub(universal_mean)

            # create baseline prediction matrix
            baseline = pd.DataFrame(index = m_u_train.index, columns = m_u_train.columns).fillna(0)
            baseline = (baseline.add(movie_mean, axis = 0)).add(user_mean_dev)
            # store in persistant storage
            try:
                with open('recommenders/training_data/m_u_baseline', 'wb') as outfile:
                    pickle.dump(baseline, outfile)
            except EnvironmentError:
                print(EnvironmentError)

        return baseline



    def get_rmse(self):
        '''
            Calculate and return RMSE value by comparing test and train data.
            Gets train/test data from corresponding functions.
        '''

        # get test dataset
        try:
            with open('recommenders/testing_data/m_u_list', 'rb') as infile:
                m_u_test_list = (pickle.load(infile)).reset_index()
        except EnvironmentError:
            print(EnvironmentError)

        # get predicted ratings
        m_u_c = self.get_trained_data()

        # initialise sum
        rmse = 0
        # initialise number of elements
        n = 0

        for index, rows in m_u_test_list.iterrows():
            user_id = rows['userId']
            movie_id = rows['movieId']
            if(movie_id in m_u_c.index):
                rmse += (rows['rating'] - (m_u_c.loc[movie_id])[user_id])**2
                n += 1

        rmse = (rmse/n)**(0.5)

        return rmse


    def get_mae(self):
        '''
            Calculate and return MAE value by comparing test and train data.
            Gets train/test data from corresponding functions.
        '''
        # get test dataset
        try:
            with open('recommenders/testing_data/m_u_list', 'rb') as infile:
                m_u_test_list = (pickle.load(infile)).reset_index()
        except EnvironmentError:
            print(EnvironmentError)

        # get predicted ratings
        m_u_c = self.get_trained_data()

        # initialise sum
        mae = 0
        # initialise number of elements
        n = 0

        for index, rows in m_u_test_list.iterrows():
            user_id = rows['userId']
            movie_id = rows['movieId']
            if(movie_id in m_u_c.index):
                mae += abs((rows['rating'] - (m_u_c.loc[movie_id])[user_id]))
                n += 1

        mae /= n

        return mae


if __name__ == '__main__':
    pass



        
