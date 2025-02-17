import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import utile.PerceptronVisualizer as perVizu


####################################  Dataset aléatoire  ####################################

X = np.random.randn(500, 2)
y = np.where(X[:, 1] > X[:, 0], 1, 0)

scaler = StandardScaler()
X = scaler.fit_transform(X)

vizualize = perVizu.PerceptronVizualizer()

##################################  perceptron simple  ##################################

parametres = [("sigmoid", 150, 0.01, 8),
               ("tanh", 100, 0.01, 64),
               ("relu", 100, 0.01, 64)]

vizualize.compare_activations(X, y, parametres)

##################################  perceptron serie  ##################################

parametres = [("sigmoid", 200, 0.1, 8),
               ("tanh", 100, 0.1, 16),
               ("relu", 100, 0.1, 16)]

vizualize.compare_activations(X, y, parametres, [1])

##################################  perceptron parallele  ##################################

parametres = [("sigmoid", 200, 0.05, 8),
               ("tanh", 200, 0.01, 8),
               ("relu", 200, 0.01, 8)]

vizualize.compare_activations(X, y, parametres, [2])


####################################  Dataset iris  ####################################

iris = load_iris()

y = iris.target
y = np.where(y == 0, 0, 1)

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)

# PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
X = X_pca[:, :2]

##################################  perceptron simple  ##################################

parametres = [("sigmoid", 150, 0.01, 16),
               ("tanh", 150, 0.005, 16),
               ("relu", 100, 0.005, 16)] # (activation, epochs, lr, batch_size)

vizualize.compare_activations(X, y, parametres)

##################################  perceptron serie  ##################################

parametres = [("sigmoid", 200, 0.1, 8),
               ("tanh", 100, 0.1, 16),
               ("relu", 100, 0.1, 16)]

vizualize.compare_activations(X, y, parametres, [1])

##################################  perceptron parallele  ##################################

parametres = [("sigmoid", 150, 0.1, 8),
               ("tanh", 150, 0.01, 8),
               ("relu", 150, 0.05, 8)]

vizualize.compare_activations(X, y, parametres, [2])
