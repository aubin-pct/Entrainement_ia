import numpy as np

class LinearRegression:

    def __init__(self):
        self.b0 = None
        self.b1 = None

    def fit(self, X, Y):
        X = X.ravel()
        Y = Y.ravel()
        x_mean = np.mean(X)
        y_mean = np.mean(Y)
        cov = (np.sum((X-x_mean) * (Y-y_mean)))/X.shape[0]
        varX = (np.sum((X-x_mean)**2))/X.shape[0]
        self.b1 = cov/varX
        self.b0 = y_mean - self.b1*x_mean
        self.__set_coef_determination(X, Y)

    def predict(self, X):
        return self.b0 + self.b1 * X
    
    def MSE(self, YR, YP):
        return np.mean((YR - YP)**2)
    
    def __set_coef_determination(self, X, Y):
        SCE = np.sum((self.predict(X) - np.mean(Y)) ** 2)
        SCT = np.sum((Y - np.mean(Y)) ** 2)
        self.coef_determination = SCE / SCT

    def get_coef_determination(self):
        return self.coef_determination