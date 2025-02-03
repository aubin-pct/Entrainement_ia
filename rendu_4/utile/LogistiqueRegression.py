import numpy as np



class LogistiqueRegression:

    def __init__(self):
        self.B = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y, alpha=0.1, epochs = 1000):
        m, n = X.shape
        self.B = np.zeros(n)
        for _ in range(epochs):
            y_pred = self.sigmoid(X @ self.B)
            gradient = (1/m) * X.T @ (y_pred - y) 
            self.B -= alpha * gradient
        return self.B
    
    def predict(self, X):
        tab = self.sigmoid(X @ self.B)
        tab = np.where(tab >= 0.5, 1, 0)
        return tab