import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import KFold


class PerceptronVizualizer:

    def __init__(self):
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)

    
    def compare_activations(self, X, y, parametres, hidden_layer = []):
        fig, axs = plt.subplots(ncols=len(parametres), nrows=3, figsize=(20, 10))
        axs = axs.flatten()
        index = 0

        for a, epochs, lr, batch_size in parametres:
            accuracies = []
            loss = []
            accuracies_test = []
            loss_test = []

            for train_index, test_index in self.kf.split(X, y):
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                network = []
                if (len(hidden_layer) == 0):
                    network.append(Dense(1, activation=a, input_shape=(X.shape[1],)))
                else:
                    network.append(Dense(hidden_layer[0], activation=a, input_shape=(X.shape[1],)))
                    for l in range(1, len(hidden_layer)):
                        network.append(Dense(hidden_layer[l], activation=a))
                    network.append(Dense(1, activation="sigmoid"))

                model = Sequential(network)
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