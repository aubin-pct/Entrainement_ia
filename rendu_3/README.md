# **3ème Rendu**


Dans ce troisième rendu, l'objectif est d'améliorer les résultats obtenus précédemment en intégrant une régression polynomiale.

Pour cela :

* Une classe `Scaler` a été créée pour normaliser les données.
* La classe `LinearRegression` a été améliorée pour gérer les régressions multiples.
* Une nouvelle classe `PolynomialRegression` a été ajoutée.

En plus de cela, plusieurs graphiques ont été ajoutés pour comparer les performances des modèles :

1. Un premier graphique compare les résultats obtenus pour différents degrés de régression polynomiale.
2. Des graphiques supplémentaires montrent l'évolution de l'erreur quadratique moyenne (MSE) et du coefficient de détermination (R²) en fonction du degré.
3. Enfin, l'utilisateur est invité à choisir l'ordre qu'il préfère pour afficher le graphique correspondant.

## Lancement

```
	python app.py
```

    ou

```
	python3 app.py
```

## Analyse

Les résultats montrent une nette supériorité de la régression polynomiale par rapport à la régression linéaire simple.
La régression polynomiale est clairement plus adaptée pour modéliser les données, offrant un meilleur ajustement et des prédictions beaucoup plus précises que la régression linéaire simple.

## Annexe

    Vous trouverez les graphiques dans le dossier img.
    Vous trouverez les fichiers CSV dans le dossier csv_files.
