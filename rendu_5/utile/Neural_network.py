from curses import ACS_PI
import numpy as np
import pandas as pd


class Neural_network:

    def __init__(self, nb_hidden_layer, nb_neural_layer):
        self.W = []
        self.b = []
        self.nb_hidden_layer = nb_hidden_layer
        self.nb_neural_layer = nb_neural_layer

    def __initialisation(self, nb_parametres):
        self.W.append(np.zeros(nb_parametres, self.nb_neural_layer))  # première couche chachée
        self.b.append(np.zeros(1, self.nb_neural_layer)) # biais
        for _ in range(self.nb_hidden_layer - 1):
            self.W.append(np.zeros(self.nb_neural_layer, self.nb_neural_layer)) # autres couches cachées
            self.b.append(np.zeros(1, self.nb_neural_layer)) # biais
        self.W.append(np.zeros(self.nb_neural_layer, 1)) # neurone de sortie
        self.b.append(np.zeros(1, 1)) # biais

    def fit(self, X, y, alpha=0.01, epochs=1000):
        self.__initialisation(X.shape[1])
        X = np.c_[np.ones(X.shape[0]), X] 
        for _ in range(epochs):
            A = self.predict_proba(X)
            self.__back_propagation(A, y, alpha)
    
    def sigmoid(self, z):
        z = np.clip(z, -600, 600)
        return 1 / (1 + np.exp(-z))
    
    
    def predict_proba(self, X):
        A = []
        A.append(X)
        for i in range(self.nb_hidden_layer + 1):
            A.append(self.sigmoid(A[i] @ self.W[i] + self.b[i]))
        return A
    
    def predict(self, X):
        tab = self.predict_proba(X)[-1]
        tab = np.where(tab >= 0.5, 1, 0)
        return tab
    
    def __back_propagation(self, A, y, alpha):
        last_i = len(self.W)
        dZ = A[last_i] - y
        self.W[last_i] -= alpha * 1/len(y) * A[last_i-1].T @ dZ
        self.b[last_i] -= alpha * 1/len(y) * np.sum(dZ, axis=0)
        for i in range(len(self.W) - 1, 0, -1):
            dA = dZ @ self.W[i+1].T
            dZ = dA @ A[i] * (1 - A[i]) # (1 - A[i]) -> car sigmoid
            self.W[i] -= alpha * 1/len(y) * A[i-1].T @ dZ
            self.b[i] -= alpha * 1/len(y) * np.sum(dZ, axis=0)

