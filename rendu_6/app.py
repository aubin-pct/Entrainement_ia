import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import utile.Perceptron as per
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam





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
plt.savefig("rendu_6/img/perceptron_simple_activations.png", format="png")
plt.show()


fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
axs = axs.flatten()
index = 0
activations = ["sigmoid", "tanh", "relu"]
epochs = 100
alpha = 0.01
# Perceptron seul
for a in activations:

    model = Sequential([
        Dense(1, activation=a, input_shape=(X.shape[1],)),
        Dense(1, activation=a)
    ])
    optimizer = Adam(learning_rate=alpha)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    historique = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))


    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    
    Z = Z.reshape(xx.shape)

    # Affichage de la frontière de décision
    axs[index].contourf(xx, yy, Z, alpha=0.3)  # Coloration des régions de décision
    axs[index].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')  # Points de données réels
    axs[index].set_xlabel("Feature 1")
    axs[index].set_ylabel("Feature 2")
    axs[index].set_title(f"Activation : {a} | lr : {alpha}")


    # Affichage graphe accuracy
    axs[index+3].plot(historique.history['accuracy'], label='Accuracy')
    axs[index+3].set_title(f"Evolution de l'accuracy -> {epochs} epochs")
    axs[index+3].set_xlabel("Epoch")
    axs[index+3].set_ylabel("Accuracy")
    axs[index+3].legend()

    index += 1
fig.suptitle("Frontière de décision du Perceptron", fontsize=16)
plt.savefig("rendu_6/img/perceptron_simple_activations.png", format="png")
plt.show()

fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
axs = axs.flatten()
index = 0
activations = ["sigmoid", "tanh", "relu"]
epochs = 100
alpha = 0.01
# Perceptron seul
for a in activations:

    model = Sequential([
        Dense(2, activation=a, input_shape=(X.shape[1],)),
        Dense(1, activation=a)
    ])
    optimizer = Adam(learning_rate=alpha)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    historique = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))


    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    
    Z = Z.reshape(xx.shape)

    # Affichage de la frontière de décision
    axs[index].contourf(xx, yy, Z, alpha=0.3)  # Coloration des régions de décision
    axs[index].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')  # Points de données réels
    axs[index].set_xlabel("Feature 1")
    axs[index].set_ylabel("Feature 2")
    axs[index].set_title(f"Activation : {a} | lr : {alpha}")


    # Affichage graphe accuracy
    axs[index+3].plot(historique.history['accuracy'], label='Accuracy Train')
    axs[index+3].plot(historique.history['val_accuracy'], label='Accuracy Test')
    axs[index+3].set_title(f"Evolution de l'accuracy -> {epochs} epochs")
    axs[index+3].set_xlabel("Epoch")
    axs[index+3].set_ylabel("Accuracy")
    axs[index+3].legend()

    index += 1
fig.suptitle("Frontière de décision du Perceptron", fontsize=16)
plt.savefig("rendu_6/img/perceptron_simple_activations.png", format="png")
plt.show()