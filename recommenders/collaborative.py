# module for item-item collaborative filtering

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

class Collaborative():

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
            with open('recommenders/training_data/m_u_c', 'rb') as infile:
                m_u_c = pickle.load(infile)
        except EnvironmentError:
            m_u_train, m_u_test = self.get_train_test()            
            m_u_c = m_u_train

            # row wise mean for all rows
            row_mean = m_u_c.mean(axis = 1)

            # create mean centered ratings matrix
            m_u_c = m_u_c.T.fillna(row_mean).T
            m_u_c = m_u_c.subtract(row_mean, axis = 0)

            # generate similarity matrix
            similarity = cosine_similarity(m_u_c)
            similarity = pd.DataFrame(similarity, index = m_u_train.index, columns = m_u_train.index)
            np.fill_diagonal(similarity.values, 1)

            sim_sum = (similarity.abs()).dot(pd.notna(m_u_train).astype('int'))

            m_u_c = similarity.dot(m_u_train.fillna(0))
            m_u_c = m_u_c.div(sim_sum).fillna(0)

            try:
                with open('recommenders/training_data/m_u_c', 'wb') as outfile:
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
                m_u_raw = pd.pivot_table(m_u_raw, values = 'rating', index = 'movieId', columns = 'userId')
                m_u_train, m_u_test = train_test_split(m_u_raw, test_size = 0.2)

                # dump m_u and m_u_test
                try:
                    with open('recommenders/training_data/m_u', 'wb') as outfile:
                        pickle.dump(m_u_train, outfile)
                    with open('recommenders/testing_data/m_u', 'wb') as outfile:
                        pickle.dump(m_u_test, outfile)
                    return m_u_train, m_u_test
                except EnvironmentError:
                    print(EnvironmentError)
                    return None
                                 

            except EnvironmentError:
                print(EnvironmentError) 
                return None      



    def get_rmse(self):
        '''
            Calculate and return RMSE value by comparing test and train data.
            Gets train/test data from corresponding functions.
        '''
        m_u_c = self.get_trained_data()
        m_u_train, m_u_test = self.get_train_test()

    def get_mae(self):
        '''
            Calculate and return MAE value by comparing test and train data.
            Gets train/test data from corresponding functions.
        '''
        m_u_c = self.get_trained_data()
        m_u_train, m_u_test = self.get_train_test()


if __name__ == '__main__':
    pass