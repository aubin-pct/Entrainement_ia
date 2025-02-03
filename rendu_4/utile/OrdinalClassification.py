import numpy as np


class OrdinalClassification:

    def __init__(self, nb_classe):
        self.nb_classe = nb_classe
        self.theta = None

    def sigmoid(self, z):
        z = np.clip(z, -600, 600)
        return 1 / (1 + np.exp(-z))
    
    def proba(self, X, k=0):
        if (k == 0):
            return self.sigmoid(X @ self.theta[k, :])
        return self.sigmoid(X @ self.theta[k, :]) - self.sigmoid(X @ self.theta[k-1, :])

    def predict(self, X):
        n = X.shape[0] 
        tab = np.zeros((n, self.nb_classe))
        for k in range(self.nb_classe):
            tab[:, k] = self.proba(X, k)
        return np.argmax(tab, axis=1)
    
    def gradient(self, X, y, k):
        tab = np.where(y == k, 1, 0)
        return np.sum((self.proba(X, k) - tab)[:, np.newaxis] * X, axis=0)   # (P(yi=k|Xi) - (y=k)) * Xi
    
    def fit(self, X, y, alpha=0.1, epochs = 1000):
        n = X.shape[1]
        self.theta = np.zeros((self.nb_classe, n))
        for _ in range(epochs):
            for k in range(self.nb_classe):
                self.theta[k] -= alpha * self.gradient(X, y, k)
        return self.theta