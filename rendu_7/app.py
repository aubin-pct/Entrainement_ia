import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

data = pd.read_csv("rendu_7/csv_files/dermatologie.csv")

X = data.iloc[:, :-1].to_numpy()
y_true = data.iloc[:, -1].to_numpy()
y = tf.keras.utils.to_categorical(y_true)

scaler = StandardScaler()
X = scaler.fit_transform(X)


accuracies = []
loss = []
accuracies_test = []
loss_test = []

for train_index, test_index in kf.split(X, y):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = tf.keras.models.Sequential([
            Dense(128, activation="tanh", input_shape=(X.shape[1],)),
            Dense(128, activation="tanh"),
            Dense(128, activation="tanh"),
            Dense(6, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    historique = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), batch_size=8)

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
# Affichage accuracy - loss
fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
axs = axs.flatten()
axs[0].plot(accuracies, label='Accuracy Train')
axs[0].plot(accuracies_test, label='Accuracy Test')
axs[0].set_title(f"Evolution de l'accuracy")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Accuracy")
axs[0].legend()

axs[1].plot(loss, label='Loss Train')
axs[1].plot(loss_test, label='Loss Test')
axs[1].set_title(f"Evolution de la perte")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Loss")
axs[1].legend()
plt.savefig("rendu_7/img/accuracy_loss.png", format="png")
plt.show()

y_pred = np.argmax(model.predict(X), axis=1)

cm = confusion_matrix(y_true, y_pred)

# Affichage matrice confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
plt.xlabel('y_pred')
plt.ylabel('y_true')
plt.title('Matrice de Confusion')
plt.savefig("rendu_7/img/matrice_confusion.png", format="png")
plt.show()


model = tf.keras.models.Sequential([
        Dense(128, activation="tanh", input_shape=(X.shape[1],)),
        Dense(128, activation="tanh"),
        Dense(128, activation="tanh"),
        Dense(6, activation="softmax")
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=100, batch_size=8)

# Courbe ROC
y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
y_pred_proba = model.predict(X)

# Calcul de la courbe ROC pour chaque classe
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(y_true_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Affichage de la courbe ROC
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
axs = axs.flatten()
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown'] 

for i in range(y_true_bin.shape[1]):
    axs[i].plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'Classe {i} (AUC = {roc_auc[i]:.4f})')

    axs[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    axs[i].set_xlim([0.0, 1.0])
    axs[i].set_ylim([0.0, 1.05])
    axs[i].set_xlabel('Taux de faux positifs')
    axs[i].set_ylabel('Taux de vrais positifs')
    axs[i].set_title(f'Courbes ROC - classe {i}')
    axs[i].legend(loc='lower right')
plt.savefig("rendu_7/img/courbe_roc.png", format="png")
plt.show()