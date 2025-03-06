# **Mon entrainement Python / IA**

Dépot de mon entrainement pour coder une ia en python sous la forme de plusieurs rendues

L'IA codée a pour but de reconnaître le chiffre écrit par la sourie de l'utilisateur.

## Détails

#### 1er Rendu : Déssin & Traitement d'image

Ce rendu présente un programme permettant de dessiner avec la souris dans une fenêtre.
Lorsque la fenêtre est fermée, l'image est enregistrée et traitée pour être conforme aux normes MNIST.

#### 2ème Rendu : Regression Linéaire

Ce rendu inclut un programme pour effectuer une régression linéaire simple.
Il affiche la courbe de projection du modèle créé et la compare aux résultats de la bibliothèque scikit-learn.
Les indicateurs de performance (MSE et R²) sont également calculés et affichés pour comparaison.

#### 3ème Rendu : Regression Polynomial

Ce rendu améliore le précédent en utilisant une régression polynomiale.
Une classe `Scaler` est créée pour normaliser les données, et la classe `LinearRegression` est améliorée pour gérer les régressions multiples.
Différents degrés de polynômes sont comparés pour déterminer le modèle optimal.
L'affichage final compare ce nouveau modèle de régression polynomiale à l'ancien modèle de régression linéaire simple.7

#### 4ème Rendu : Ordinal Classification avec PCA et Analyse de Corrélation

Ce projet aborde la classification ordinale en appliquant des techniques d'analyse de corrélation, de régression linéaire et d'Analyse en Composantes Principales (PCA) pour réduire la dimensionnalité et améliorer les performances du modèle. Des tests statistiques comme le test de Spearman et d'ANOVA ont permis d'explorer les relations entre les variables. En comparaison, la régression logistique ordinale avec PCA a montré une précision moyenne de 57%, tandis que des modèles comme SVM ont atteint 83% de précision.

#### 5ème Rendu : Perceptron

Ce projet implémente un perceptron multicouche (MLP) et analyser son efficacité sur deux jeux de données différents. L'objectif est d'observer comment différentes architectures et fonctions d'activation influencent la convergence et la précision du modèle. L'étude inclut une analyse approfondie des performances à travers des métriques classiques (accuracy, perte) et une visualisation des frontières de décision.

#### 6ème Rendu : Perceptron - Tensorflow

Ce projet est similaire au précédent tout en explorant l'utilisation de TensorFlow

#### 7ème Rendu : Classification Dermatologique avec Réseau de Neurones

Ce projet met en œuvre un réseau de neurones en utilisant TensorFlow/Keras pour classifier des maladies dermatologiques en fonction de leur gravité. Le modèle est entraîné avec un ensemble de données médicales prétraitées, en appliquant une pondération des classes pour compenser le déséquilibre des échantillons. Une validation croisée (K-Fold) est utilisée pour évaluer la robustesse du modèle, et les performances sont mesurées à l'aide d'une matrice de confusion et d'autres métriques clés.

#### 8ème Rendu : Classification D'images - CNN

Dans ce projet, un réseau de neurones convolutifs (CNN) est conçu pour la reconnaissance de chiffres manuscrits à partir du dataset MNIST. En plus de l'entraînement classique, une expérimentation est menée pour évaluer la robustesse du modèle face à des images altérées (changement de taille, repositionnement aléatoire). L'analyse inclut une étude de l’évolution des métriques d'apprentissage et l'utilisation d'outils comme la matrice de confusion pour affiner l’interprétation des résultats.

## -Projet en cours-

## 📝 Installation

  1 - Clonez le dépot :

```
	git clone https://github.com/aubin-pct/Entrainement_ia.git
	cd Entrainement_ia
```

  2 - Configurez l'environnement :

```
	python -m venv venv
	source venv/bin/activate    # Sous Windows : venv\Scripts\activate
	pip install -r requirements.txt
```

## 🚀 Lancement

    un fichier app.py est présent dans chaque rendu.

```
	python rendue_<numero_rendu>/app.py
```
