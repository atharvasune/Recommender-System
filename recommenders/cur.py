# module for CUR 
import pandas as pd
import numpy as np
import time

class Cur():

    def __init__(self, r, ds=[], dim_reduce=None):
        self.r = r
        if ds == []:
            dataset = pd.read_csv("./csv/ratings.csv")
            pivot_table = pd.pivot_table(dataset, values = 'rating', index = 'userId', columns = 'movieId')
            self.M = pivot_table.iloc[:, :].fillna(0).values
        else:
            self.M = ds
        self.dim_reduce = dim_reduce
        
        tim = time.time()
        self.get_cur()
        tim = time.time() - tim
        
        # if dim_reduce == None:
        #     tim = time.time()
        #     self.get_cur()
        #     tim = time.time() - tim
        # else:
        #     tim = time.time()
        #     self.get_cur_reduce(dim_reduce)
        #     tim = time.time() - tim
        print(tim)
        # print(self.M)

    # following are the public methods, add others as per need

    def get_W(self):
        W = []
        M = self.M
        w = []
        for i in range(len(self.rows)):
            w = []
            for j in range(len(self.cols)):
                w.append(M[i][j])
            W.append(w)
        
        return np.array(W)
        print(W)

    def get_U(self, W):
        # print(W)
        X, S, Y = np.linalg.svd(W, full_matrices=False)
        print(np.sqrt(np.mean(W - (X @ S @ Y)) ** 2))
        # print(X, S, Y)
        
        if self.dim_reduce != None:
            diag_sq = 0
            sq_90 = 0

            diag_sq = np.sum(S**2)

            for i in range(len(S)):
                sq_90 += S[i]*S[i]
                if(sq_90 > 0.9*diag_sq):
                    num_val_retained = i+1
                    break
            print(num_val_retained)
            S = S[0:num_val_retained]
            X = X[:,0:num_val_retained]
            Y = Y[0:num_val_retained,:]
        
        print(S.shape)
        
        s = []
        # for i in S:
        #     if i != 0: s.append(1/i)
        #     else: s.append(i)
        s = np.reciprocal(S)
        S = np.diag(s)
        # print(S)
        U = Y.T @ (S/1e+15) @ X.T
        print('U\n', U)
        return U
        
    
    def get_cols(self):
        M = self.M
        denom = np.sum(M ** 2)
        prob = []
        for i in range(M.shape[1]):
            x = np.sum(M[:, i] ** 2)
            prob.append(x/denom)
        prob = np.array(prob)
        cols = []
        for i in range(self.r):
            cols.append(np.random.choice(M.shape[1], p=prob))
        cols = np.array(cols)

        C = np.array(M[:, cols[0]])
        
        for i in range(1, len(cols)):
            C = np.c_[C, M[:, cols[i]]]
        
        self.cols = cols
        return C

    def get_rows(self):
        M = self.M
        # print(M.shape)
        denom = np.sum(M ** 2)
        prob = []

        for i in range(M.shape[0]):
            x = np.sum(M[i, :] ** 2)
            prob.append(x/denom)
        prob = np.array(prob)
        rows = []

        for i in range(self.r):
            rows.append(np.random.choice(M.shape[0], p=prob))
        rows = np.array(rows)
        
        R = []
        for i in range(0, len(rows)):
            R.append(M[rows[i], :])
        R = np.array(R)
        self.rows = rows
        return R

    def get_cur(self):
        self.C = self.get_cols()
        self.R = self.get_rows()
        W = self.get_W()
        self.U = self.get_U(W)
        print(self.C @ self.U @ self.R)

    def get_recommendation(userId):
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

    def get_rmse(self):
        '''
            Calculate and return RMSE value by comparing test and train data.
            Gets train/test data from corresponding functions.
        '''
        P = self.C @ self.U @ self.R
        # print(P, '\n', self.M)
        err = self.M - P
        return np.sqrt(np.mean(err ** 2))

    def get_mae(self):
        '''
            Calculate and return MAE value by comparing test and train data.
            Gets train/test data from corresponding functions.
        '''
        P = self.C @ self.U @ self.R
        # print(P, '\n', self.M)
        err = self.M - P
        return np.mean(err)        

if __name__ == '__main__':
    cur = Cur(450, dim_reduce=0.9)
    # cur.get_cur()
    print(cur.get_rmse())
    print(cur.get_mae())
    # print(np.linalg.cur)