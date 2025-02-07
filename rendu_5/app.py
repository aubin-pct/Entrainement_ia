import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utile.Perceptron as per

X = np.array([[10, 5], [7, 5], [5, 6], [1, 2]])
y = np.array([0, 0, 0, 1])

p = per.Perceptron()
p.fit(X, y)

w0, w1, w2 = p.W

# Définition des limites pour x1
x1_vals = np.array([X[:, 0].min() - 1, X[:, 0].max() + 1])

# Calcul de x2 en fonction de x1
x2_vals = - (w0 + w1 * x1_vals) / w2


plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', edgecolor='k', marker='o', s=100, label='Classe 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', edgecolor='k', marker='^', s=100, label='Classe 1')

plt.plot(x1_vals, x2_vals, 'k--', linewidth=2, label="Frontière de décision")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Frontière de décision du Perceptron")
plt.legend()
plt.show()