import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import utile.Perceptron as per
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



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

epochs = 10

# Perceptron seul
p = per.Perceptron()
accuracies = p.fit(X, y, epochs = epochs)

w1, w2 = p.W 
b = p.B

# Définition des limites pour x1
x1_vals = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])

# Calcul de x2 en fonction de x1
x2_vals = -(b + w1 * x1_vals) / w2


plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', edgecolor='k', marker='o', s=50, label='Label 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', edgecolor='k', marker='^', s=50, label='Label 1')

plt.plot(x1_vals, x2_vals, 'k--', linewidth=2, label="Frontière de décision")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Frontière de décision du Perceptron")
plt.legend()
plt.savefig("rendu_5/img/perceptron.png", format="png")
plt.show()

# Affichage Accuracy evolution
plt.plot(range(len(accuracies)), accuracies, linestyle='-', linewidth=1, label="Accuracy")
plt.xlabel("epochs")
plt.ylabel("Accuracy")
plt.title(f"Evolution de l'accuracy -> {epochs} epochs")
plt.legend()
plt.savefig("rendu_5/img/accuracy.png", format="png")
plt.show()

# 2 Perceptron en series 
X_new = p.predict_proba(X).reshape(-1, 1)
p2 = per.Perceptron()
p2.fit(X_new, y)

plt.scatter(X_new[y == 0], y[y == 0], color='blue', edgecolor='k', marker='o', s=50, label='Label 0')
plt.scatter(X_new[y == 1], y[y == 1], color='red', edgecolor='k', marker='^', s=50, label='Label 1')

w0 = p2.W
b = p2.B

# Définition des limites pour x1
x_val = -b / w0

plt.axvline(x=x_val, color='red', linestyle='--', label="Frontière de décision")

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Frontière de décision du Perceptron (serie de 2)")
plt.legend()
plt.savefig("rendu_5/img/perceptron_serie.png", format="png")
plt.show()


# 2 Perceptron en parallele
p1 = per.Perceptron()
p2 = per.Perceptron()
p3 = per.Perceptron()
X_1, X_2, y_1, y_2 = train_test_split(X, y, test_size=0.5, random_state=42)
p1.fit(X_1, y_1, epochs = epochs)
p2.fit(X_2, y_2, epochs = epochs)
p3.fit(np.c_[p1.predict_proba(X), p2.predict_proba(X)], y, epochs = epochs)

w1, w2 = p3.W 
b = p3.B

# Définition des limites pour x1
x1_vals = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])

# Calcul de x2 en fonction de x1
x2_vals = -(b + w1 * x1_vals) / w2


plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', edgecolor='k', marker='o', s=50, label='Label 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', edgecolor='k', marker='^', s=50, label='Label 1')

plt.plot(x1_vals, x2_vals, 'k--', linewidth=2, label="Frontière de décision")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Frontière de décision de Perceptron 2 en parallele")
plt.legend()
plt.savefig("rendu_5/img/perceptron_parallele.png", format="png")
plt.show()

