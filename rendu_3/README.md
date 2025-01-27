# **3ème Rendu**

Dans ce troisième rendu, l'objectif est d'améliorer les résultats obtenus précédemment en intégrant une régression polynomiale. Pour cela :

1. Une classe `Scaler` a été créée pour normaliser les données.
2. La classe `LinearRegression` a été améliorée pour prendre en charge les régressions multiples.
3. Une nouvelle classe `PolynomialRegression` a été ajoutée.

Les données d'entraînement du modèle sont importées depuis un fichier CSV situé dans le dossier `csv_files`.

L'affichage final compare les performances du nouveau modèle de régression polynomiale avec celles de l'ancien modèle de régression linéaire simple. De plus, le graphique précédent a été mis à jour pour inclure les deux modèles, permettant une visualisation claire de leurs différences sur les nouvelles données.

## Lancement

```
	python app.py
```

    ou

```
	python3 app.py
```

## Analyse

Les résultats montrent une nette supériorité de la régression polynomiale par rapport à la régression linéaire simple :

* **MSE (Erreur Quadratique Moyenne)** :
  * Régression polynomiale : **8.06** (faible erreur).
  * Régression simple : **142.83** (erreur significativement plus élevée).
* **R² (Coefficient de Détermination)** :
  * Régression polynomiale : **0.945** (très proche de 1, indiquant un excellent ajustement).
  * Régression simple : **0.031** (proche de 0, suggérant un ajustement très médiocre).

La régression polynomiale est clairement plus adaptée pour modéliser les données, offrant un meilleur ajustement et des prédictions beaucoup plus précises que la régression linéaire simple.

## Annexe

    Vous trouverez les graphiques dans le dossier img.
    Vous trouverez le fichier CSV dans le dossier csv_files.
