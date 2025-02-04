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

## Lancement

```
	python app.py
```

    ou

```
	python3 app.py

```

## Résultats

* **Matrice de corrélation** : Identifie les variables les plus fortement corrélées (>0.7 ou <-0.7).
* **Régressions linéaires** : Permet d'examiner les relations entre les variables corrélées.
* **Test de Spearman** : Affiché dans le terminal pour analyser la corrélation entre la variable cible et les autres variables.
* **Classification Ordinale avec PCA** : Réduit la dimensionnalité des données avant d'entraîner le modèle.
* **Courbe ROC** : Évalue la performance du modèle de classification.

## Auteurs

Ce projet a été réalisé dans le cadre de l'analyse et la modélisation de données avec une approche de classification ordinale et réduction de dimension.
