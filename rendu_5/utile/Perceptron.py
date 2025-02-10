import numpy as np
import pandas as pd

class Perceptron:

    def __init__(self):
        self.W = None
        self.B = None

    def fit(self, X, y, alpha=0.01, epochs=1000):
        self.W = np.zeros(X.shape[1])
        self.B = 0
        accuracies = []
        for _ in range(epochs):
            correct_predictions = 0
            for xi, yi_true in zip(X, y):
                yi_pred = self.activation_function(xi @ self.W + self.B)
                self.W += alpha * (yi_true - yi_pred) * xi
                self.B += alpha * (yi_true - yi_pred)
                if yi_pred == yi_true:
                    correct_predictions += 1
            accuracy = correct_predictions / len(X)
            accuracies.append(accuracy)
        return accuracies
        
    def predict(self, X):     
        return self.activation_function(X @ self.W + self.B)
    
    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)
    
    def predict_proba(self, X):
        return self.sigmoid(X @ self.W + self.B)
    
    def sigmoid(self, z):
        z = np.clip(z, -600, 600)
        return 1 / (1 + np.exp(-z))