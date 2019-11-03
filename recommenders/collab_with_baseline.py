# module for collaborative filtering with baseline

class CollabBaseline():

	def __init__():
        pass


    # following are the public methods, add others as per need

    def get_recommendation(user_id):
        '''
            Function to return movie recommendations for the given userId
        '''
        pass

    def generate_train_data():
        '''
            Returns various training data, as per the algorithm, after optimum preprocessing
            Returns data from pickle files if available, otherwise generates, stores pickle files and then
            returns
        '''
        pass

    def generate_test_data():
        '''
            Returns various testing data, as per the algorithm, after optimum preprocessing
            Returns data from pickle files if available, otherwise generates, stores pickle files and then
            returns
        '''
        pass

    def get_rmse():
        '''
            Calculate and return RMSE value by comparing test and train data.
            Gets train/test data from corresponding functions.
        '''
        pass

    def get_mae():
        '''
            Calculate and return MAE value by comparing test and train data.
            Gets train/test data from corresponding functions.
        '''
        pass

if __name__ == '__main__':
	pass