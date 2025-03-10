# 9ème rendu : Comparaison CNN Simple vs CNN couplé à un RNN

## 📌 Présentation du projet

Cette étude compare deux architectures de deep learning pour la classification d'images :

* **Modèle 1** : Un CNN simple
* **Modèle 2** : Un CNN couplé à un RNN

Les implémentations détaillées sont disponibles dans les notebooks :

* [`rendu_9_CNN.ypynb`](https://rendu_9_cnn.ypynb/)
* [`rendu_9_CNN_RNN.ypynb`](https://rendu_9_cnn_rnn.ypynb/)

Les deux notebooks suivent une structure identique et documentent l'ensemble du processus de création des modèles.

## Résultats clés

### Performance des modèles (cross-validation)

| Métrique          | CNN simple | CNN+RNN | Amélioration    |
| ------------------ | ---------- | ------- | ---------------- |
| Précision (train) | 96,15%     | 97,43%  | **+1.28%** |
| Précision (val)   | 97,56%     | 98,08%  | **+0.52%** |
| Précision (test)  | 98,29%     | 98,45%  | **+0.16%** |

    ~ Réduction d'erreur sur le jeu de test :**9.35%**

### Temps d'exécution

* **Entraînement** : Modèle CNN+RNN légèrement plus lent
* **Inférence** : Les deux modèles restent rapides pour la prédiction d'images

## Conclusion

Le modèle **CNN+RNN** démontre :
✅ Une meilleure performance globale (+1.28% en entraînement, +0.52% en validation)
✅ Une erreur réduite de 9.35% sur les données de test
✅ Un temps de prédiction pratique pour des applications réelles

Malgré un temps d'entraînement légèrement supérieur, ses avantages en précération en font l'architecture recommandée pour ce cas d'usage.
