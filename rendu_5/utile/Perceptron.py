import numpy as np
import pandas as pd

class Perceptron:

    def __init__(self):
        self.W = None

    def fit(self, X, y, alpha=0.01, epochs=1000):
        self.W = np.zeros(X.shape[1]+1)
        X = np.c_[np.ones(X.shape[0]), X] 

        for _ in range(epochs):
            for xi, yi_true in zip(X, y):
                yi_pred = self.activation_function(xi @ self.W)
                self.W += alpha * (yi_true - yi_pred) * xi
        
    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]       
        return self.activation_function(X @ self.W)
    
    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)