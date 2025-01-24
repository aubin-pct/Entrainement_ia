import numpy as np

class LinearRegression:

    def __init__(self):
        self.B = None

    def fit(self, X, Y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.B = np.linalg.inv(X.T @ X) @ (X.T @ Y) 
        self.__set_coef_determination(X, Y)

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.B
    
    def MSE(self, YR, YP):
        return np.mean((YR - YP)**2)
    
    def __set_coef_determination(self, X, Y):
        SCE = np.sum((self.predict(X[:, 1:]) - np.mean(Y)) ** 2)
        SCT = np.sum((Y - np.mean(Y)) ** 2)
        self.coef_determination = SCE / SCT

    def get_coef_determination(self):
        return self.coef_determination