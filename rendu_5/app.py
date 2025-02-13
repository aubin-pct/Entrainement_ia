import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import utile.Perceptron as per
import utile.Neural_network as nrn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




X = np.random.randn(100, 2)
y = np.where(X[:, 1] + X[:, 0] > 1, 1, 0)
#scaler = StandardScaler()
#X = scaler.fit_transform(X)
epochs = 500
alpha = 0.01

activations = ["step", "sigmoid", "tanh", "relu"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

p_seul = per.Perceptron()
fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
axs = axs.flatten()
index = 0
# Perceptron seul
for a in activations:

    accuracies, test_accuracies = p_seul.fit(x_train, y_train, epochs=epochs, activation=a, alpha=alpha, x_test=x_test, y_test=y_test)

    w1, w2 = p_seul.W
    b = p_seul.B

    # Affichage graphe Frontiere de decision
    x1_vals = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
    x2_vals = -(b + w1 * x1_vals) / w2

    axs[index].scatter(X[y == 0, 0], X[y == 0, 1], color='blue', edgecolor='k', marker='o', s=50, label='Label 0')
    axs[index].scatter(X[y == 1, 0], X[y == 1, 1], color='red', edgecolor='k', marker='^', s=50, label='Label 1')

    axs[index].plot(x1_vals, x2_vals, 'k--', linewidth=2, label="Frontière de décision")
    axs[index].set_xlabel("X1")
    axs[index].set_ylabel("X2")
    axs[index].legend()
    axs[index].set_xlim([-3, 3]) 
    axs[index].set_ylim([-3, 3])
    axs[index].set_title(f"Activation : {a} | lr : {alpha}")

    # Affichage graphe accuracy
    axs[index+4].plot(range(len(accuracies)), accuracies, linestyle='-', linewidth=1, label="Train Accuracy")
    axs[index+4].plot(range(len(test_accuracies)), test_accuracies, linestyle='-', linewidth=1, label="Test Accuracy")
    axs[index+4].set_xlabel("epochs")
    axs[index+4].set_ylabel("Accuracy")
    axs[index+4].set_title(f"Evolution de l'accuracy -> {len(accuracies)} epochs")
    axs[index+4].legend()

    index += 1
fig.suptitle("Frontière de décision du Perceptron", fontsize=16)
plt.savefig("rendu_5/img/perceptron_simple_activations.png", format="png")
plt.show()



fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
axs = axs.flatten()
index = 0
epochs = 1000
alpha = 0.1
# Perceptron en serie
for a in activations:
    p_serie = nrn.Neural_network(nb_hidden_layer=1, nb_neural_layer=1, activation=a)
    train_accuracies, test_accuracies = p_serie.fit(x_train, y_train, epochs=epochs, alpha=alpha, x_test=x_test, y_test=y_test)

    w1 = p_serie.W[-1][:, 0]
    b = p_serie.b[-1][0, 0]

    # Définir les limites de la grille
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Créer une grille de points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                         np.linspace(y_min, y_max, 100))

    # Transformer la grille en une liste de points et faire des prédictions
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = p_serie.predict(grid_points)  # On prédit les classes pour chaque point de la grille
    
    # Reshape pour correspondre à la grille
    Z = Z.reshape(xx.shape)

    # Affichage de la frontière de décision
    axs[index].contourf(xx, yy, Z, alpha=0.3)  # Coloration des régions de décision
    axs[index].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')  # Points de données réels
    axs[index].set_xlabel("Feature 1")
    axs[index].set_ylabel("Feature 2")
    axs[index].set_title(f"Activation : {a} | lr : {alpha}")

    # Affichage graphe accuracy
    axs[index+4].plot(range(len(train_accuracies)), train_accuracies, linestyle='-', linewidth=1, label="Train Accuracy")
    axs[index+4].plot(range(len(test_accuracies)), test_accuracies, linestyle='-', linewidth=1, label="Test Accuracy")
    axs[index+4].set_xlabel("epochs")
    axs[index+4].set_ylabel("Accuracy")
    axs[index+4].set_title(f"Evolution de l'accuracy -> {len(train_accuracies)} epochs")
    axs[index+4].legend()

    index += 1
fig.suptitle("Frontière de décision | 2 Perceptrons en serie", fontsize=16)
plt.savefig("rendu_5/img/perceptron_serie_activations.png", format="png")
plt.show()


fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
axs = axs.flatten()
index = 0
epochs = 1000
alpha = 0.1
# Perceptron en parallele
for a in activations:
    p_para = nrn.Neural_network(nb_hidden_layer=1, nb_neural_layer=2, activation=a)
    train_accuracies, test_accuracies = p_para.fit(x_train, y_train, epochs=epochs, alpha=alpha, x_test=x_test, y_test=y_test)

    w1 = p_para.W[-1][:, 0]
    b = p_para.b[-1][0, 0]

    # Définir les limites de la grille
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Créer une grille de points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                         np.linspace(y_min, y_max, 100))

    # Transformer la grille en une liste de points et faire des prédictions
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = p_para.predict(grid_points)  # On prédit les classes pour chaque point de la grille
    
    # Reshape pour correspondre à la grille
    Z = Z.reshape(xx.shape)

    # Affichage de la frontière de décision
    axs[index].contourf(xx, yy, Z, alpha=0.3)  # Coloration des régions de décision
    axs[index].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')  # Points de données réels
    axs[index].set_xlabel("Feature 1")
    axs[index].set_ylabel("Feature 2")
    axs[index].set_title(f"Activation : {a} | lr : {alpha}")

    # Affichage graphe accuracy
    axs[index+4].plot(range(len(train_accuracies)), train_accuracies, linestyle='-', linewidth=1, label="Train Accuracy")
    axs[index+4].plot(range(len(test_accuracies)), test_accuracies, linestyle='-', linewidth=1, label="Test Accuracy")
    axs[index+4].set_xlabel("epochs")
    axs[index+4].set_ylabel("Accuracy")
    axs[index+4].set_title(f"Evolution de l'accuracy -> {len(train_accuracies)} epochs")
    axs[index+4].legend()

    index += 1
fig.suptitle("Frontière de décision | 2 Perceptrons en parallele", fontsize=16)
plt.savefig("rendu_5/img/perceptron_parallele_activations.png", format="png")
plt.show()




















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

epochs = 1000
alpha = 0.1


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

p_seul = per.Perceptron()
fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
axs = axs.flatten()
index = 0
# Perceptron seul
for a in activations:

    accuracies, test_accuracies = p_seul.fit(x_train, y_train, epochs=epochs, activation=a, alpha=alpha, x_test=x_test, y_test=y_test)

    w1, w2 = p_seul.W
    b = p_seul.B

    # Affichage graphe Frontiere de decision
    x1_vals = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
    x2_vals = -(b + w1 * x1_vals) / w2

    axs[index].scatter(X[y == 0, 0], X[y == 0, 1], color='blue', edgecolor='k', marker='o', s=50, label='Label 0')
    axs[index].scatter(X[y == 1, 0], X[y == 1, 1], color='red', edgecolor='k', marker='^', s=50, label='Label 1')

    axs[index].plot(x1_vals, x2_vals, 'k--', linewidth=2, label="Frontière de décision")
    axs[index].set_xlabel("X1")
    axs[index].set_ylabel("X2")
    axs[index].legend()
    axs[index].set_xlim([-3, 3]) 
    axs[index].set_ylim([-3, 3])
    axs[index].set_title(f"Activation : {a} | lr : {alpha}")

    # Affichage graphe accuracy
    axs[index+4].plot(range(len(accuracies)), accuracies, linestyle='-', linewidth=1, label="Train Accuracy")
    axs[index+4].plot(range(len(test_accuracies)), test_accuracies, linestyle='-', linewidth=1, label="Test Accuracy")
    axs[index+4].set_xlabel("epochs")
    axs[index+4].set_ylabel("Accuracy")
    axs[index+4].set_title(f"Evolution de l'accuracy -> {len(accuracies)} epochs")
    axs[index+4].legend()

    index += 1
fig.suptitle("Frontière de décision du Perceptron", fontsize=16)
plt.savefig("rendu_5/img/perceptron_simple_activations_iris.png", format="png")
plt.show()



fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
axs = axs.flatten()
index = 0
epochs = 1000
alpha = 0.1
# Perceptron en serie
for a in activations:
    p_serie = nrn.Neural_network(nb_hidden_layer=1, nb_neural_layer=1, activation=a)
    train_accuracies, test_accuracies = p_serie.fit(x_train, y_train, epochs=epochs, alpha=alpha, x_test=x_test, y_test=y_test)

    w1 = p_serie.W[-1][:, 0]
    b = p_serie.b[-1][0, 0]

    # Définir les limites de la grille
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Créer une grille de points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                         np.linspace(y_min, y_max, 100))

    # Transformer la grille en une liste de points et faire des prédictions
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = p_serie.predict(grid_points)  # On prédit les classes pour chaque point de la grille
    
    # Reshape pour correspondre à la grille
    Z = Z.reshape(xx.shape)

    # Affichage de la frontière de décision
    axs[index].contourf(xx, yy, Z, alpha=0.3)  # Coloration des régions de décision
    axs[index].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')  # Points de données réels
    axs[index].set_xlabel("Feature 1")
    axs[index].set_ylabel("Feature 2")
    axs[index].set_title(f"Activation : {a} | lr : {alpha}")

    # Affichage graphe accuracy
    axs[index+4].plot(range(len(train_accuracies)), train_accuracies, linestyle='-', linewidth=1, label="Train Accuracy")
    axs[index+4].plot(range(len(test_accuracies)), test_accuracies, linestyle='-', linewidth=1, label="Test Accuracy")
    axs[index+4].set_xlabel("epochs")
    axs[index+4].set_ylabel("Accuracy")
    axs[index+4].set_title(f"Evolution de l'accuracy -> {len(train_accuracies)} epochs")
    axs[index+4].legend()

    index += 1
fig.suptitle("Frontière de décision | 2 Perceptrons en serie", fontsize=16)
plt.savefig("rendu_5/img/perceptron_serie_activations_iris.png", format="png")
plt.show()


fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
axs = axs.flatten()
index = 0
epochs = 1000
alpha = 0.1
# Perceptron en parallele
for a in activations:
    p_para = nrn.Neural_network(nb_hidden_layer=1, nb_neural_layer=2, activation=a)
    train_accuracies, test_accuracies = p_para.fit(x_train, y_train, epochs=epochs, alpha=alpha, x_test=x_test, y_test=y_test)

    w1 = p_para.W[-1][:, 0]
    b = p_para.b[-1][0, 0]

    # Définir les limites de la grille
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Créer une grille de points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                         np.linspace(y_min, y_max, 100))

    # Transformer la grille en une liste de points et faire des prédictions
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = p_para.predict(grid_points)  # On prédit les classes pour chaque point de la grille
    
    # Reshape pour correspondre à la grille
    Z = Z.reshape(xx.shape)

    # Affichage de la frontière de décision
    axs[index].contourf(xx, yy, Z, alpha=0.3)  # Coloration des régions de décision
    axs[index].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')  # Points de données réels
    axs[index].set_xlabel("Feature 1")
    axs[index].set_ylabel("Feature 2")
    axs[index].set_title(f"Activation : {a} | lr : {alpha}")

    # Affichage graphe accuracy
    axs[index+4].plot(range(len(train_accuracies)), train_accuracies, linestyle='-', linewidth=1, label="Train Accuracy")
    axs[index+4].plot(range(len(test_accuracies)), test_accuracies, linestyle='-', linewidth=1, label="Test Accuracy")
    axs[index+4].set_xlabel("epochs")
    axs[index+4].set_ylabel("Accuracy")
    axs[index+4].set_title(f"Evolution de l'accuracy -> {len(train_accuracies)} epochs")
    axs[index+4].legend()

    index += 1
fig.suptitle("Frontière de décision | 2 Perceptrons en parallele", fontsize=16)
plt.savefig("rendu_5/img/perceptron_parallele_activations_iris.png", format="png")
plt.show()

