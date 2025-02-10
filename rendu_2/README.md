# **2ème Rendu : Regression Linéaire**

Dans ce deuxième rendu, il s'agit d'un programme permettant d'effectuer une régression linéaire simple et d'afficher la courbe de projection du modèle créé pour la comparer aux résultats de la bibliothèque scikit-learn.

Les indicateurs de performance (MSE et R2) sont également calculés et affichés dans le terminal afin dêtre aussi comparés aux résultats de la bibliothèque scikit-learn.

## Lancement

```
	python app.py
```

    ou

```
	python3 app.py
```

## Analyse

    Les valeurs de MSE sont très élevées, ce qui est attendu étant donné que les valeurs de population sont de l'ordre de plusieurs dizaines de millions. Les deux modèles ont des valeurs de MSE très proches, ce qui indique que les performances des deux modèles sont très similaires en termes d'erreur de prédiction.

    Le coefficient de détermination étant proche de 1 indique que le modèle explique bien la variance des données. Les deux modèles ont des valeurs de R² très proches de 0.966, ce qui
est très bon. Cela signifie que les deux modèles expliquent environ 96.6% de la variance des données de population.

## Annexe

    Vous trouverez le graphique dans le dossier img
