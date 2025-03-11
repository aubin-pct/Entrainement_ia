# 📌 8ème rendu : Classification D'images - CNN

## 📝 Description du Rendu

Ce projet consiste en la mise en place d'un modèle de classification d'images basé sur un réseau de neurones convolutifs (CNN). L'objectif principal est d'entraîner un modèle sur le dataset MNIST afin de reconnaître des chiffres manuscrits. Le projet inclut également une expérimentation sur des images modifiées en augmentant leur taille et en les repositionnant de manière aléatoire avant de les réduire à la taille originale.

## 🏆 Objectifs du projet

* Construire un modèle CNN performant pour la classification des chiffres manuscrits du dataset MNIST.
* Visualiser l'évolution des métriques d'apprentissage (accuracy et loss).
* Générer et tester des images altérées pour observer la robustesse du modèle.
* Analyser les performances à l'aide de métriques telles que la matrice de confusion et le rapport de classification.

## 📊 Résultats

* **Performance du modèle** : Le modèle a été entraîné sur 10 époques et a atteint un niveau de précision significatif sur l'ensemble de test.
* **Visualisation des métriques** : Des graphiques d'évolution de l'accuracy et de la perte ont été générés pour mieux comprendre l'apprentissage du modèle.
* **Expérimentation sur images altérées** : Un ensemble d'images a été modifié pour tester la robustesse du modèle, et les résultats ont été analysés à l'aide d'une matrice de confusion.

## 📂 Structure du Rendu

├── model/                  		 # Dossier contenant le modèle entraîné
│   ├── mon_model.keras      # Modèle sauvegardé
├──img/
│   ├── accuracy_loss.png
│   ├── matrice_conf_1.png   # Matrice de confusion - données MNIST
│   ├── matrice_conf_2.png   # Matrice de confusion - données créées
├── README.md
├── rendu_8_CNN_RNN.ipynb       # Script contenant le code du modèle CNN x RNN
├── rendu_8_CNN.ipynb                 # Script contenant le code du modèle CNN simple

## Dataset MNIST

Le dataset MNIST est un ensemble de données bien connu pour la classification d'images de chiffres manuscrits. Il contient :

* 60 000 images pour l'entraînement
* 10 000 images pour le test
  Chaque image est en niveaux de gris et de taille 28x28 pixels.

## Existant

Atteindre une précision supérieure à 99% sur les tests MNIST n'est pas très compliqué. Cependant, il devient plus difficile pour le modèle de reconnaître les chiffres lorsque ceux-ci sont modifiés (rotation, décalage, zoom) via  *ImageDataGenerator* , et encore plus lorsqu'on utilise la fonction *create_imperfect_image* que j'ai écrite, qui réduit et affine les chiffres.

L'objectif ici est d'optimiser les performances du modèle face à ces deux types de modifications avant d'augmenter la taille du jeu de données.

Pour choisir la bonne architecture, le modèle a été entraîné avec les 60 000 images du jeu de données d'entraînement de base, complétées par 20 000 images générées par *ImageDataGenerator* et 25 000 images générées par  *create_imperfect_image* . Bien sûr, le jeu de données sera équilibré lors de l'entraînement final, l'objectif étant d'obtenir un modèle capable de bien généraliser.

### Architecture du modèle

Pour l'architecture de la couche de convolution, je me suis inspiré de cet article qui montre qu'après tests, une architecture avec deux couches de convolution suivies d'un pooling est plus efficace qu'une autre approche. (Source : [Brendan Artley - MNIST Keras Simple CNN](https://medium.com/@BrendanArtley/mnist-keras-simple-cnn-99-6-731b624aee7f))

En ce qui concerne l'utilisation des réseaux de neurones récurrents (RNN), comme le montre cet article ([RNN for MNIST Classification - Kaggle](https://www.kaggle.com/code/mikolajbabula/rnn-for-mnist-classification-tensor-flow)), un RNN simple ne surpasse pas un CNN. Toutefois, la question se pose : qu'en est-il de la combinaison des deux ? Faut-il privilégier un réseau entièrement connecté ou un réseau récurrent ?

Après avoir testé plusieurs architectures et optimisé les hyperparamètres grâce aux algorithmes de recherche aléatoire et bayésienne, et validé les modèles via la validation croisée, voici les résultats :

### Résultats

* **Modèle CNN** : Ce modèle atteint une précision de 98 % sur le jeu de test, qui comprend 10 000 images de base, 10 000 images générées par *create_imperfect_image* et 10 000 images générées par  *ImageDataGenerator* . Il est à la fois rapide, performant et génère des résultats similaires sur les ensembles d'entraînement et de validation, ce qui montre une bonne généralisation et une précision satisfaisante. (Voir Annexe 1 pour plus de détails sur la structure)
* **Modèle CNN x RNN (GRU)** : Ce modèle obtient une précision similaire à celle du modèle CNN, mais il est considérablement plus lent. (Voir Annexe 2 pour plus de détails sur la structure)

## 🚀 Lancemement

Ce rendu a été réalisé sur un notebook, où tous les résultats sont déjà affichés dans le fichier `rendu_8.ipynb`.

## **📸 Sorties**

Vous trouverez les sorties dans les notebook *rendu_8_CNN_RNN.ipynb* et *rendu_8_CNN_RNN.ipynb*

## Annexe

### 1/ Code CNN classique

```
model = tf.keras.models.Sequential([
      Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(28, 28, 1)),
      BatchNormalization(),
      Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(28, 28, 1)),
      BatchNormalization(),
      MaxPooling2D((2,2), strides=2),
      Dropout(0.3),

      Conv2D(64, (3, 3), activation='relu', padding="same"),
      BatchNormalization(),
      Conv2D(64, (3, 3), activation='relu', padding="same"),
      BatchNormalization(),
      MaxPooling2D((2,2), strides=2),
      Dropout(0.3),

      Conv2D(128, (3, 3), activation='relu', padding="same"),
      BatchNormalization(),
      Conv2D(128, (3, 3), activation='relu', padding="same"),
      BatchNormalization(),
      MaxPooling2D((2,2), strides=2),
      Dropout(0.3),

      Flatten(),

      Dense(512, activation='relu'),
      Dropout(0.25),
      BatchNormalization(),
      Dense(512, activation='relu'),
      Dropout(0.5),
      BatchNormalization(),

      Dense(10, activation="softmax")
  ])
  optimizer = AdamW(learning_rate=0.0001)
  model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
```

### 2/ Code CNN x RNN (GRU)

```
model = tf.keras.models.Sequential([
      Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(28, 28, 1)),
      BatchNormalization(),
      Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(28, 28, 1)),
      BatchNormalization(),
      MaxPooling2D((2,2), strides=2),
      Dropout(0.3),

      Conv2D(64, (3, 3), activation='relu', padding="same"),
      BatchNormalization(),
      Conv2D(64, (3, 3), activation='relu', padding="same"),
      BatchNormalization(),
      MaxPooling2D((2,2), strides=2),
      Dropout(0.3),

      Conv2D(128, (3, 3), activation='relu', padding="same"),
      BatchNormalization(),
      Conv2D(128, (3, 3), activation='relu', padding="same"),
      BatchNormalization(),
      MaxPooling2D((2,2), strides=2),
      Dropout(0.3),

      Reshape((-1, 128)),

      Bidirectional(GRU(512, activation='relu', return_sequences=True)),
      Dropout(0.25),

      Bidirectional(GRU(256, activation='relu')),
      Dropout(0.5),

      Dense(10, activation="softmax")
  ])
  optimizer = Adam(learning_rate=0.001, weight_decay=0.0001)
  model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
```
