# 9ème rendu : Comparaison CNN Simple vs CNN couplé à un RNN

## Présentation du projet

Cette étude compare deux architectures de deep learning pour la classification d'images, avec un focus sur la robustesse face aux variations de chiffres manuscrits :

* **Modèle 1** : Un CNN simple, optimisé pour la reconnaissance de chiffres modifiés.
* **Modèle 2** : Un CNN couplé à un RNN, explorant l'apport des réseaux récurrents.

Les implémentations détaillées sont disponibles dans les notebooks :

* [`rendu_9_CNN.ipynb`](https://rendu_9_cnn.ypynb/)
* [`rendu_9_CNN_RNN.ipynb`](https://rendu_9_cnn_rnn.ypynb/)

Les deux notebooks suivent une structure identique et documentent l'ensemble du processus de création des modèles, de l'augmentation des données à l'évaluation.

## Objectifs

* Améliorer la robustesse du modèle de reconnaissance de chiffres manuscrits face aux variations de position, d'orientation et de taille.
* Comparer l'efficacité d'un CNN simple optimisé avec un CNN couplé à un RNN.
* Optimiser les hyperparamètres des modèles via des techniques de recherche avancées.

## Données

* Jeu de données MNIST de base (60 000 images d'entraînement, 10 000 images de test).
* Données augmentées :
  * 60 000 + 10 000 images générées par transformations géométriques aléatoires (rotation, décalage, zoom).
  * 60 000 + 10 000 images générées par placement aléatoire des chiffres sur un fond plus grand, simulant des imperfections.

## ️ Méthodologie

1. **Préparation des données** :
   * Normalisation des images.
   * Inversion des couleurs (chiffres noirs sur fond blanc).
   * Création de données augmentées via `ImageDataGenerator` et `create_imperfect_image`.
2. **Architecture des modèles** :
   * CNN simple : Inspiration de l'article de Brendan Artley pour l'architecture des couches convolutives.
   * CNN+RNN : Exploration de l'ajout de couches GRU bidirectionnelles après les couches convolutives.
3. **Optimisation des hyperparamètres** :
   * Utilisation de Keras Tuner pour l'optimisation bayésienne.
   * Mise en œuvre de la recherche aléatoire pour comparer les approches.
4. **Entraînement et validation** :
   * Validation croisée K-Fold (5 folds).
   * Callback `ReduceLROnPlateau` pour ajuster le taux d'apprentissage.
5. **Évaluation** :
   * Mesure de l'accuracy et de la perte sur les ensembles d'entrainement et de validation.
   * Analyse de l'évolution de ces métriques au cours de l'entraînement.
6. **Entrainement sur 20 epochs** :
   * Observation de la convergence du modèle.
7. **Test** :
   * Le modèle final est évalué sur l'ensemble de test indépendant pour mesurer sa performance sur des données non vues.
   * Les métriques de performance enregistrées sont :
     * **Rapport de classification** : pour une analyse détaillée de la précision, du rappel et du score F1 pour chaque chiffre.
     * **Matrice de confusion** : pour visualiser les erreurs de classification et comprendre quelles classes sont les plus souvent confondues.

## Existant

Atteindre une précision supérieure à 99% sur les tests MNIST n'est pas très compliqué. Cependant, il devient plus difficile pour le modèle de reconnaître les chiffres lorsque ceux-ci sont modifiés (rotation, décalage, zoom) via *ImageDataGenerator*, et encore plus lorsqu'on utilise la fonction *create_imperfect_image* que j'ai écrite, qui réduit et affine les chiffres.

L'objectif ici est d'optimiser les performances du modèle face à ces deux types de modifications.

### Architecture du modèle

Pour l'architecture de la couche de convolution, je me suis inspiré de cet article qui montre qu'après tests, une architecture avec deux couches de convolution suivies d'un pooling est plus efficace qu'une autre approche. (Source : [Brendan Artley - MNIST Keras Simple CNN](https://medium.com/@BrendanArtley/mnist-keras-simple-cnn-99-6-731b624aee7f))

En ce qui concerne l'utilisation des réseaux de neurones récurrents (RNN), comme le montre cet article ([RNN for MNIST Classification - Kaggle](https://www.kaggle.com/code/mikolajbabula/rnn-for-mnist-classification-tensor-flow)), un RNN simple ne surpasse pas un CNN. Toutefois, la question se pose : qu'en est-il de la combinaison des deux ? Faut-il privilégier un réseau entièrement connecté ou un réseau récurrent ?

### Justification des choix

* **Double convolution et Max Pooling** : Permettent d'extraire des caractéristiques complexes tout en réduisant la dimensionnalité.
* **Batch Normalization et Dropout** : Améliorent la stabilité de l'entraînement et réduisent le surapprentissage.
* **Optimisation bayésienne (Keras Tuner)** : Recherche efficace des hyperparamètres optimaux.
* **Validation croisée** : Estimation robuste de la performance du modèle.
* **Callbacks (ReduceLROnPlateau)** : Ajustement dynamique du taux d'apprentissage.
* **Entraînement sur 20 epochs** : Convergence améliorée du modèle.
* **Couches GRU bidirectionnelles** : Capture des dépendances temporelles dans les caractéristiques extraites.
* **AdamW** : Optimiseur performant.

  **Des comparaisons ont été effectuées dans le fichier _rendu_9_comparaisons.ipynb._**

  Ce notebook complète les justifications précédentes en explorant plus en profondeur l'impact des choix architecturaux et des techniques de régularisation sur le modèle CNN simple. Il s'agit d'une démarche expérimentale consistant à retirer ou modifier certaines composantes clés du modèle, comme les couches de convolution, le Dropout et la Batch Normalization, afin d'observer leur influence sur la performance.

  En comparant les résultats obtenus avec différentes configurations, le notebook met en évidence l'importance de chaque élément pour la capacité du modèle à apprendre, à généraliser et à converger efficacement. Cette analyse permet de valider empiriquement les choix de conception justifiés précédemment et de confirmer l'architecture optimale pour la classification de chiffres manuscrits.

## Résultats clés

### Performance des modèles (cross-validation)

| Métrique          | CNN simple | CNN+RNN | Amélioration    |
| :----------------- | :--------- | :------ | :--------------- |
| Précision (train) | 96,15%     | 97,43%  | **+1.28%** |
| Précision (val)   | 97,56%     | 98,08%  | **+0.52%** |
| Précision (test)  | 98,29%     | 98,45%  | **+0.16%** |

~ Réduction d'erreur sur le jeu de test :**9.35%**

### Temps d'exécution

* **Entraînement** : Modèle CNN+RNN légèrement plus lent.
* **Inférence** : Les deux modèles restent rapides pour la prédiction d'images.

## Conclusion

Le modèle **CNN+RNN** démontre :

* ✅ Une meilleure performance globale (+1.28% en entraînement, +0.52% en validation).
* ✅ Une erreur réduite de 9.35% sur les données de test.
* ✅ Un temps de prédiction pratique pour des applications réelles.

Malgré un temps d'entraînement légèrement supérieur, ses avantages en précision et en robustesse en font l'architecture recommandée pour ce cas d'usage.
