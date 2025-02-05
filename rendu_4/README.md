# 📌 4ème rendu : Ordinal Classification avec PCA et Analyse de Corrélation

## 📝 Description du Rendu

Ce projet implémente une classification ordinale en combinant plusieurs techniques d'analyse de données, notamment :

* L'analyse de corrélation entre les variables
* La régression linéaire pour les variables fortement corrélées
* La transformation d'une variable cible en une variable catégorielle
* L'application de l'Analyse en Composantes Principales (PCA)
* L'évaluation des performances du modèle et comparaison avec d'autres modéles avec des courbes ROC

## 📂 Structure du Rendu

Le projet est organisé comme suit :

* `OrdinalClassification.py` : Classe implémentant la régression logistique ordinale.
* `LogistiqueRegression.py` : Implémentation de la régression logistique standard.
* `Scaler.py` : Classe implémentant la normalisation min et max.
* `PolynomialRegression.py` : Classe implémentant la régression polynomiale, calcule le MSE & R² et normalise avec Scaler.py.

## 🚀 Lancement

```
	python app.py
```

    ou

```
	python3 app.py

```

## 🧠 Utilisation de la PCA

L'Analyse en Composantes Principales (PCA) a été utilisée pour réduire la dimensionnalité du jeu de données, afin de :

* **Simplifier le modèle** tout en conservant l'information essentielle.
* **Améliorer les performances** en éliminant les variables redondantes ou peu informatives, ce qui permet de réduire le bruit et prévenir le sur-apprentissage.
* **Rendre les variables moins corrélées** en transformant les données de manière à ce que les nouvelles composantes principales soient indépendantes entre elles.Faciliter l'interprétation et la visualisation des données en projetant celles-ci dans un espace de dimension réduite.

## 📊 Résultats

* **Matrice de corrélation** : Identification des variables fortement corrélées pour mieux orienter la régression.
* **Régressions linéaires** : Analyse des relations entre variables corrélées.
* **Test de Spearman** : Corrélations significatives entre les variables, par exemple :
  * **MEDV x LSTAT** : Corrélation négative forte (-0.85), indiquant que le prix des maisons diminue avec un statut socio-économique plus bas.
  * **MEDV x RM** : Corrélation positive forte (0.63), suggérant que plus de pièces par logement est associé à un prix plus élevé.
* **Test de Chi2** : Mise en évidence de relations entre la variable cible et des variables qualitatives :
  * **MEDV_category x RAD** : Relation forte avec le nombre de routes accessibles (p < 0.05).
  * **MEDV_category x CHAS** : Relation plus faible avec la proximité de la rivière Charles (p ≈ 0.05).
* **Test d'ANOVA** : Impact des variables continues sur `MEDV_category` :
  * **RM** : Plus de pièces par logement est associé à un prix plus élevé (p ≪ 0.05).
  * **LSTAT** : Un statut socio-économique plus bas correspond à des prix plus faibles (p ≪ 0.05).
* **Analyse comparative des modèles** : La régression logistique ordinale avec PCA atteint une précision moyenne de  57 % , avec une meilleure prédiction des classes  0 et 2 , tandis que la classe 1 reste plus difficile à classifier. En revanche, Random Forest et XGBoost affichent une précision  parfaitement optimisée (100%) , ce qui peut indiquer un sur-apprentissage. SVM offre un bon compromis avec une précision de  83 % , montrant une capacité de généralisation supérieure à la régression logistique tout en évitant l'overfitting observé avec les modèles d'ensemble.

### **📸 Sorties**

#### 1/ Matrice de corrélation

![matrice de corrélation](img/matrice_correlation.png)

#### 2/ Régression des variables fortement corrélées

![graphes de regression](img/regression_graphe.png)

#### 3/ Régressions linéaires entre les entrées et la sortie (transformée en variable catégorielle)

![graphes régressions de la variable cible](img/regression_target_graphe.png)

#### 4/ Courbes ROC

![graphe ROC](https://file+.vscode-resource.vscode-cdn.net/home/ob1/Documents/Entrainement_ia/rendu_4/img/ROC_graphe.png)

## ✨ Auteurs

Ce projet a été réalisé dans le cadre de l'analyse et la modélisation de données avec une approche de classification ordinale et réduction de dimension.
