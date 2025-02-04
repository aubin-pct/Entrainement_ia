## 4ème rendu : Ordinal Classification avec PCA et Analyse de Corrélation

## Description du Rendu

Ce projet implémente une classification ordinale en combinant plusieurs techniques d'analyse de données, notamment :

* L'analyse de corrélation entre les variables
* La régression linéaire pour les variables fortement corrélées
* La transformation d'une variable cible en une variable catégorielle
* L'application de l'Analyse en Composantes Principales (PCA)
* L'évaluation des performances du modèle avec une courbe ROC

## Structure du Rendu

Le projet est organisé comme suit :

* `OrdinalClassification.py` : Classe implémentant la régression logistique ordinale.
* `LogistiqueRegression.py` : Implémentation de la régression logistique standard.
* `Scaler.py` : Classe implémentant la normalisation min et max.
* `PolynomialRegression.py` : Classe implémentant la régression polynomiale.

## Lancement

```
	python app.py
```

    ou

```
	python3 app.py

```

## Utilisation de la PCA

L'Analyse en Composantes Principales (PCA) a été utilisée pour réduire la dimensionnalité du jeu de données, afin de :

* **Simplifier le modèle** tout en conservant l'information essentielle.
* **Améliorer les performances** en éliminant les variables redondantes ou peu informatives, ce qui permet de réduire le bruit et prévenir le sur-apprentissage.
* ****Rendre les variables moins corrélées** **en transformant les données de manière à ce que les nouvelles composantes principales soient indépendantes entre elles.Faciliter l'interprétation et la visualisation des données en projetant celles-ci dans un espace de dimension réduite.

## Résultats

* **Matrice de corrélation** : Identification des variables fortement corrélées pour mieux orienter la régression.
* **Régressions linéaires** : Analyse des relations entre variables corrélées.
* **Test de Spearman** : Corrélations significatives entre les variables, par exemple :
  * **MEDV x LSTAT** : Corrélation négative forte (-0.85), indiquant que le prix des maisons diminue avec un statut socio-économique plus bas.
  * **MEDV x RM** : Corrélation positive forte (0.63), suggérant que plus de pièces par logement est associé à un prix plus élevé.
* **Classification Ordinale avec PCA** : Le modèle a montré des performances modérées, avec une précision moyenne de 57% (Validation croisée). Les classes 0 et 2 ont montré de meilleures performances que la classe 1, qui nécessiterait des ajustements.

## Auteurs

Ce projet a été réalisé dans le cadre de l'analyse et la modélisation de données avec une approche de classification ordinale et réduction de dimension.
