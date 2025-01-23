import numpy as np

class LinearRegression:

    def __init__(self):
        self.b0 = None
        self.b1 = None

    def fit(self, X, Y):
        x_mean = np.mean(X)
        y_mean = np.mean(Y)
        cov = (np.sum((X-x_mean) * (Y-y_mean)))/X.shape[0]
        varX = (np.sum((X-x_mean)**2))/X.shape[0]
        self.b1 = cov/varX
        self.b0 = y_mean - self.b1*x_mean

    def predict(self, X):
        return self.b0 + self.b1 * X