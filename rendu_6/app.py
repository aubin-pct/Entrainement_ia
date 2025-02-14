from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD


####################################  Dataset aléatoire  ####################################


X = np.random.randn(500, 2)
y = np.where(X[:, 1] > X[:, 0], 1, 0)

scaler = StandardScaler()
X = scaler.fit_transform(X)

kf = KFold(n_splits=5, shuffle=True, random_state=42)


##################################  perceptron simple  ##################################

fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(20, 10))
axs = axs.flatten()
index = 0

parametres = [("sigmoid", 150, 0.01, 8),
               ("tanh", 100, 0.01, 64),
               ("relu", 100, 0.01, 64)] # (activation, epochs, lr, batch_size)

for a, epochs, lr, batch_size in parametres:
    accuracies = []
    loss = []
    accuracies_test = []
    loss_test = []

    for train_index, test_index in kf.split(X, y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = Sequential([
            Dense(1, activation=a, input_shape=(X.shape[1],))
        ])
        optimizer = SGD(learning_rate=lr)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        historique = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), batch_size=batch_size)

        # Stocker les résultats sous forme de liste
        accuracies.append(historique.history["accuracy"])
        accuracies_test.append(historique.history["val_accuracy"])
        loss.append(historique.history["loss"])
        loss_test.append(historique.history["val_loss"])

    accuracies = np.array(accuracies)
    loss = np.array(loss)
    accuracies_test = np.array(accuracies_test)
    loss_test = np.array(loss_test)

    accuracies = np.mean(accuracies, axis=0)
    loss = np.mean(loss, axis=0)
    accuracies_test = np.mean(accuracies_test, axis=0)
    loss_test = np.mean(loss_test, axis=0)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    
    Z = Z.reshape(xx.shape)

    # Affichage de la frontière de décision
    axs[index].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)  # Coloration des régions de décision
    axs[index].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')  # Points de données réels
    axs[index].set_xlabel("Feature 1")
    axs[index].set_ylabel("Feature 2")
    axs[index].set_title(f"Activation : {a} | lr : {lr}")


    # Affichage graphe accuracy
    axs[index+3].plot(accuracies, label='Accuracy Train')
    axs[index+3].plot(accuracies_test, label='Accuracy Test')
    axs[index+3].set_title(f"Evolution de l'accuracy -> {epochs} epochs")
    axs[index+3].set_xlabel("Epoch")
    axs[index+3].set_ylabel("Accuracy")
    axs[index+3].legend()

    # Affichage perte
    axs[index+6].plot(loss, label='Loss Train')
    axs[index+6].plot(loss_test, label='Loss Test')
    axs[index+6].set_xlabel("Epoch")
    axs[index+6].set_ylabel("loss")
    axs[index+6].legend()

    index += 1
fig.suptitle("Frontière de décision du Perceptron", fontsize=16)
plt.savefig("rendu_6/img/perceptron_simple_activations.png", format="png")
plt.show()


##################################  perceptron serie  ##################################

fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(20, 10))
axs = axs.flatten()
index = 0

parametres = [("sigmoid", 200, 0.1, 8),
               ("tanh", 100, 0.1, 16),
               ("relu", 100, 0.1, 16)]

for a, epochs, lr, batch_size in parametres:
    accuracies = []
    loss = []
    accuracies_test = []
    loss_test = []

    for train_index, test_index in kf.split(X, y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = Sequential([
            Dense(1, activation=a, input_shape=(X.shape[1],)),
            Dense(1, activation="sigmoid")
        ])
        optimizer = SGD(learning_rate=lr)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        historique = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), batch_size=batch_size)

        # Stocker les résultats sous forme de liste
        accuracies.append(historique.history["accuracy"])
        accuracies_test.append(historique.history["val_accuracy"])
        loss.append(historique.history["loss"])
        loss_test.append(historique.history["val_loss"])

    accuracies = np.array(accuracies)
    loss = np.array(loss)
    accuracies_test = np.array(accuracies_test)
    loss_test = np.array(loss_test)

    accuracies = np.mean(accuracies, axis=0)
    loss = np.mean(loss, axis=0)
    accuracies_test = np.mean(accuracies_test, axis=0)
    loss_test = np.mean(loss_test, axis=0)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    
    Z = Z.reshape(xx.shape)

    # Affichage de la frontière de décision
    axs[index].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)  # Coloration des régions de décision
    axs[index].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')  # Points de données réels
    axs[index].set_xlabel("Feature 1")
    axs[index].set_ylabel("Feature 2")
    axs[index].set_title(f"Activation : {a} | lr : {lr}")


    # Affichage graphe accuracy
    axs[index+3].plot(accuracies, label='Accuracy Train')
    axs[index+3].plot(accuracies_test, label='Accuracy Test')
    axs[index+3].set_title(f"Evolution de l'accuracy -> {epochs} epochs")
    axs[index+3].set_xlabel("Epoch")
    axs[index+3].set_ylabel("Accuracy")
    axs[index+3].legend()

    # Affichage perte
    axs[index+6].plot(loss, label='Loss Train')
    axs[index+6].plot(loss_test, label='Loss Test')
    axs[index+6].set_xlabel("Epoch")
    axs[index+6].set_ylabel("loss")
    axs[index+6].legend()

    index += 1
fig.suptitle("Frontière de décision des Perceptrons en serie", fontsize=16)
plt.savefig("rendu_6/img/perceptron_serie_activations.png", format="png")
plt.show()



##################################  perceptron parallele  ##################################


fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(20, 10))
axs = axs.flatten()
index = 0

parametres = [("sigmoid", 200, 0.05, 8),
               ("tanh", 200, 0.01, 8),
               ("relu", 200, 0.01, 8)]

for a, epochs, lr, batch_size in parametres:
    accuracies = []
    loss = []
    accuracies_test = []
    loss_test = []

    for train_index, test_index in kf.split(X, y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = Sequential([
            Dense(2, activation=a, input_shape=(X.shape[1],)),
            Dense(1, activation="sigmoid")
        ])
        optimizer = SGD(learning_rate=lr)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        historique = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), batch_size=batch_size)

        # Stocker les résultats sous forme de liste
        accuracies.append(historique.history["accuracy"])
        accuracies_test.append(historique.history["val_accuracy"])
        loss.append(historique.history["loss"])
        loss_test.append(historique.history["val_loss"])

    accuracies = np.array(accuracies)
    loss = np.array(loss)
    accuracies_test = np.array(accuracies_test)
    loss_test = np.array(loss_test)

    accuracies = np.mean(accuracies, axis=0)
    loss = np.mean(loss, axis=0)
    accuracies_test = np.mean(accuracies_test, axis=0)
    loss_test = np.mean(loss_test, axis=0)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    
    Z = Z.reshape(xx.shape)

    # Affichage de la frontière de décision
    axs[index].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)  # Coloration des régions de décision
    axs[index].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')  # Points de données réels
    axs[index].set_xlabel("Feature 1")
    axs[index].set_ylabel("Feature 2")
    axs[index].set_title(f"Activation : {a} | lr : {lr}")


    # Affichage graphe accuracy
    axs[index+3].plot(accuracies, label='Accuracy Train')
    axs[index+3].plot(accuracies_test, label='Accuracy Test')
    axs[index+3].set_title(f"Evolution de l'accuracy -> {epochs} epochs")
    axs[index+3].set_xlabel("Epoch")
    axs[index+3].set_ylabel("Accuracy")
    axs[index+3].legend()

    # Affichage perte
    axs[index+6].plot(loss, label='Loss Train')
    axs[index+6].plot(loss_test, label='Loss Test')
    axs[index+6].set_xlabel("Epoch")
    axs[index+6].set_ylabel("loss")
    axs[index+6].legend()

    index += 1
fig.suptitle("Frontière de décision des perceptrons en parallele", fontsize=16)
plt.savefig("rendu_6/img/perceptron_para_activations.png", format="png")
plt.show()





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

kf = KFold(n_splits=5, shuffle=True, random_state=42)

##################################  perceptron simple  ##################################

fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(20, 10))
axs = axs.flatten()
index = 0

parametres = [("sigmoid", 150, 0.01, 16),
               ("tanh", 150, 0.005, 16),
               ("relu", 100, 0.005, 16)] # (activation, epochs, lr, batch_size)

for a, epochs, lr, batch_size in parametres:
    accuracies = []
    loss = []
    accuracies_test = []
    loss_test = []

    for train_index, test_index in kf.split(X, y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = Sequential([
            Dense(1, activation=a, input_shape=(X.shape[1],))
        ])
        optimizer = SGD(learning_rate=lr)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        historique = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), batch_size=batch_size)

        # Stocker les résultats sous forme de liste
        accuracies.append(historique.history["accuracy"])
        accuracies_test.append(historique.history["val_accuracy"])
        loss.append(historique.history["loss"])
        loss_test.append(historique.history["val_loss"])

    accuracies = np.array(accuracies)
    loss = np.array(loss)
    accuracies_test = np.array(accuracies_test)
    loss_test = np.array(loss_test)

    accuracies = np.mean(accuracies, axis=0)
    loss = np.mean(loss, axis=0)
    accuracies_test = np.mean(accuracies_test, axis=0)
    loss_test = np.mean(loss_test, axis=0)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    
    Z = Z.reshape(xx.shape)

    # Affichage de la frontière de décision
    axs[index].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)  # Coloration des régions de décision
    axs[index].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')  # Points de données réels
    axs[index].set_xlabel("Feature 1")
    axs[index].set_ylabel("Feature 2")
    axs[index].set_title(f"Activation : {a} | lr : {lr}")


    # Affichage graphe accuracy
    axs[index+3].plot(accuracies, label='Accuracy Train')
    axs[index+3].plot(accuracies_test, label='Accuracy Test')
    axs[index+3].set_title(f"Evolution de l'accuracy -> {epochs} epochs")
    axs[index+3].set_xlabel("Epoch")
    axs[index+3].set_ylabel("Accuracy")
    axs[index+3].legend()

    # Affichage perte
    axs[index+6].plot(loss, label='Loss Train')
    axs[index+6].plot(loss_test, label='Loss Test')
    axs[index+6].set_xlabel("Epoch")
    axs[index+6].set_ylabel("loss")
    axs[index+6].legend()

    index += 1
fig.suptitle("Frontière de décision du Perceptron", fontsize=16)
plt.savefig("rendu_6/img/perceptron_simple_activations_iris.png", format="png")
plt.show()


##################################  perceptron serie  ##################################

fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(20, 10))
axs = axs.flatten()
index = 0

parametres = [("sigmoid", 200, 0.1, 8),
               ("tanh", 100, 0.1, 16),
               ("relu", 100, 0.1, 16)]

for a, epochs, lr, batch_size in parametres:
    accuracies = []
    loss = []
    accuracies_test = []
    loss_test = []

    for train_index, test_index in kf.split(X, y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = Sequential([
            Dense(1, activation=a, input_shape=(X.shape[1],)),
            Dense(1, activation="sigmoid")
        ])
        optimizer = SGD(learning_rate=lr)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        historique = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), batch_size=batch_size)

        # Stocker les résultats sous forme de liste
        accuracies.append(historique.history["accuracy"])
        accuracies_test.append(historique.history["val_accuracy"])
        loss.append(historique.history["loss"])
        loss_test.append(historique.history["val_loss"])

    accuracies = np.array(accuracies)
    loss = np.array(loss)
    accuracies_test = np.array(accuracies_test)
    loss_test = np.array(loss_test)

    accuracies = np.mean(accuracies, axis=0)
    loss = np.mean(loss, axis=0)
    accuracies_test = np.mean(accuracies_test, axis=0)
    loss_test = np.mean(loss_test, axis=0)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    
    Z = Z.reshape(xx.shape)

    # Affichage de la frontière de décision
    axs[index].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)  # Coloration des régions de décision
    axs[index].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')  # Points de données réels
    axs[index].set_xlabel("Feature 1")
    axs[index].set_ylabel("Feature 2")
    axs[index].set_title(f"Activation : {a} | lr : {lr}")


    # Affichage graphe accuracy
    axs[index+3].plot(accuracies, label='Accuracy Train')
    axs[index+3].plot(accuracies_test, label='Accuracy Test')
    axs[index+3].set_title(f"Evolution de l'accuracy -> {epochs} epochs")
    axs[index+3].set_xlabel("Epoch")
    axs[index+3].set_ylabel("Accuracy")
    axs[index+3].legend()

    # Affichage perte
    axs[index+6].plot(loss, label='Loss Train')
    axs[index+6].plot(loss_test, label='Loss Test')
    axs[index+6].set_xlabel("Epoch")
    axs[index+6].set_ylabel("loss")
    axs[index+6].legend()

    index += 1
fig.suptitle("Frontière de décision des Perceptrons en serie", fontsize=16)
plt.savefig("rendu_6/img/perceptron_serie_activations_iris.png", format="png")
plt.show()



##################################  perceptron parallele  ##################################


fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(20, 10))
axs = axs.flatten()
index = 0

parametres = [("sigmoid", 150, 0.1, 8),
               ("tanh", 150, 0.01, 8),
               ("relu", 150, 0.05, 8)]

for a, epochs, lr, batch_size in parametres:
    accuracies = []
    loss = []
    accuracies_test = []
    loss_test = []

    for train_index, test_index in kf.split(X, y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = Sequential([
            Dense(2, activation=a, input_shape=(X.shape[1],)),
            Dense(1, activation="sigmoid")
        ])
        optimizer = SGD(learning_rate=lr)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        historique = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), batch_size=batch_size)

        # Stocker les résultats sous forme de liste
        accuracies.append(historique.history["accuracy"])
        accuracies_test.append(historique.history["val_accuracy"])
        loss.append(historique.history["loss"])
        loss_test.append(historique.history["val_loss"])

    accuracies = np.array(accuracies)
    loss = np.array(loss)
    accuracies_test = np.array(accuracies_test)
    loss_test = np.array(loss_test)

    accuracies = np.mean(accuracies, axis=0)
    loss = np.mean(loss, axis=0)
    accuracies_test = np.mean(accuracies_test, axis=0)
    loss_test = np.mean(loss_test, axis=0)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    
    Z = Z.reshape(xx.shape)

    # Affichage de la frontière de décision
    axs[index].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)  # Coloration des régions de décision
    axs[index].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')  # Points de données réels
    axs[index].set_xlabel("Feature 1")
    axs[index].set_ylabel("Feature 2")
    axs[index].set_title(f"Activation : {a} | lr : {lr}")


    # Affichage graphe accuracy
    axs[index+3].plot(accuracies, label='Accuracy Train')
    axs[index+3].plot(accuracies_test, label='Accuracy Test')
    axs[index+3].set_title(f"Evolution de l'accuracy -> {epochs} epochs")
    axs[index+3].set_xlabel("Epoch")
    axs[index+3].set_ylabel("Accuracy")
    axs[index+3].legend()

    # Affichage perte
    axs[index+6].plot(loss, label='Loss Train')
    axs[index+6].plot(loss_test, label='Loss Test')
    axs[index+6].set_xlabel("Epoch")
    axs[index+6].set_ylabel("loss")
    axs[index+6].legend()

    index += 1
fig.suptitle("Frontière de décision des perceptrons en parallele", fontsize=16)
plt.savefig("rendu_6/img/perceptron_para_activations_iris.png", format="png")
plt.show()

