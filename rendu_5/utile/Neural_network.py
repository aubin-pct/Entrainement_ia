import numpy as np
import pandas as pd


class Neural_network:

    def __init__(self, nb_hidden_layer, nb_neural_layer, activation="sigmoid"):
        self.W = []
        self.b = []
        self.activation_functions = {
            "step" : self.step,
            "sigmoid" : self.sigmoid,
            "tanh" : self.tanh,
            "relu" : self.relu
        }
        self.nb_hidden_layer = nb_hidden_layer
        self.nb_neural_layer = nb_neural_layer
        self.activation_function = self.activation_functions[activation]

    def __initialisation(self, nb_parametres):
        init = np.sqrt(1/nb_parametres)
        if self.activation_function == self.relu:
            init = np.sqrt(2/nb_parametres)
        if (self.nb_neural_layer > 0 or self.nb_hidden_layer > 0):
            self.W.append(np.random.randn(nb_parametres, self.nb_neural_layer) * init)  # première couche chachée
            self.b.append(np.zeros((1, self.nb_neural_layer)) * init) # biais
            for _ in range(self.nb_hidden_layer - 1):
                self.W.append(np.random.randn(self.nb_neural_layer, self.nb_neural_layer) * init) # autres couches cachées
                self.b.append(np.zeros((1, self.nb_neural_layer)) * init) # biais
            self.W.append(np.random.randn(self.nb_neural_layer, 1) * init) # neurone de sortie
        else:
            self.W.append(np.random.randn(nb_parametres, 1) * init) # neurone de sortie
        self.b.append(np.zeros((1, 1)) * init) # biais

    def fit(self, X, y, alpha=0.01, epochs=1000, x_test=None, y_test=None):
        self.__initialisation(X.shape[1])
        train_accuracies = []
        test_accuracies = []
        for _ in range(epochs):
            # Accuracy
            if (x_test is not None and y_test is not None):
                y_test_pred = self.predict(x_test)
                test_accuracies.append(np.mean(y_test_pred == y_test))
            y_train_pred = self.predict(X)
            train_accuracies.append(np.mean(y_train_pred == y))
            if (train_accuracies[-1] >= 0.999):
                break

            # Forward - back propagation
            A = self.predict_proba(X)
            self.__back_propagation(A, y, alpha)
        
        return train_accuracies, test_accuracies
    
    def sigmoid(self, z):
        z = np.clip(z, -600, 600)
        return 1 / (1 + np.exp(-z))
    
    def step(self, z):
        return np.where(z >= 0, 1, 0)
    
    
    def tanh(self, z):
        return np.tanh(z)
    
    def relu(self, z):
        return np.maximum(0, z)

    
    def predict_proba(self, X):
        A = []
        A.append(X)
        for i in range(len(self.W) - 1):
            A.append(self.activation_function(A[-1] @ self.W[i] + self.b[i]))
        A.append(self.activation_function(A[-1] @ self.W[-1] + self.b[-1]))
        return A
    
    def predict(self, X):
        tab = self.predict_proba(X)[-1]
        if self.activation_function == self.sigmoid:
            return np.where(tab >= 0.5, 1, 0)
        elif self.activation_function == self.tanh:
            return np.where(tab >= 0, 1, 0)
        elif self.activation_function == self.relu:
            return np.where(tab > 0, 1, 0)
        return tab
    
    def __back_propagation(self, A, y, alpha):
        dZ = A[-1] - y.reshape(-1,1)
        self.W[-1] -= alpha * 1/len(y) * A[-2].T @ dZ
        self.b[-1] -= alpha * 1/len(y) * np.sum(dZ, axis=0)
        for i in range(len(self.W) - 1, 0, -1):
            dA = dZ @ self.W[i].T

            if self.activation_function == self.sigmoid:
                dZ = dA * (A[i] * (1 - A[i]))           # Dérivée sigmoid
            elif self.activation_function == self.tanh:
                dZ = dA * (1 - A[i]**2)                 # Dérivée tanh
            elif self.activation_function == self.relu:
                dZ = dA * (A[i] > 0)                    # Dérivée ReLU
            elif self.activation_function == self.step:
                dZ = np.zeros_like(dA) 

            self.W[i-1] -= alpha * 1/len(y) * A[i-1].T @ dZ
            self.b[i-1] -= alpha * 1/len(y) * np.sum(dZ, axis=0)

