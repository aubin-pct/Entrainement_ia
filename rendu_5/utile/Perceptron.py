import numpy as np
import pandas as pd

class Perceptron:

    def __init__(self):
        self.W = None
        self.B = None
        self.activation_functions = {
            "step" : self.activation_step,
            "sigmoid" : self.activation_sigmoid,
            "tanh" : self.activation_tanh,
            "relu" : self.activation_relu
        }

    def fit(self, X, y, alpha=0.01, epochs=1000, activation="step", x_test=None, y_test=None):
        self.W = np.zeros(X.shape[1])
        self.B = 0
        accuracies = []
        test_accuracies = []
        activation_f = self.activation_functions[activation]
        for i in range(epochs):
            yi_pred = activation_f(X @ self.W + self.B)
            self.W -= alpha * (yi_pred - y) @ X
            self.B -= alpha * np.sum(yi_pred - y)

            y_train_pred = self.predict(X, activation=activation)
            accuracies.append(np.mean(y_train_pred == y))
            if (x_test is not None and y_test is not None):
                y_test_pred = self.predict(x_test, activation=activation)
                test_accuracies.append(np.mean(y_test_pred == y_test))
                
            if (np.mean(y_train_pred == y) >= 0.999):
                break
        if x_test is not None and y_test is not None:
            return accuracies, test_accuracies
        return accuracies
        
    def predict(self, X, activation="step"):
        activation_f = self.activation_functions[activation]
        output = activation_f(X @ self.W + self.B)
        
        if activation == "sigmoid":
            return (output >= 0.5).astype(int)
        elif activation == "tanh":
            return (output >= 0).astype(int)
        elif activation == "relu":
            return (output > 0).astype(int)
        
        return output

    
    def predict_proba(self, X):
        return self.activation_sigmoid(X @ self.W + self.B)
    
    def activation_step(self, z):
        return np.where(z >= 0, 1, 0)
    
    def activation_sigmoid(self, z):
        z = np.clip(z, -600, 600)
        return 1 / (1 + np.exp(-z))
    
    def activation_tanh(self, z):
        return np.tanh(z)
    
    def activation_relu(self, z):
        return np.maximum(0, z)
