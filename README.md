# **Mon entrainement Python / IA**

Dépot de mon entrainement pour coder une ia en python sous la forme de plusieurs rendues

L'IA codée a pour but de reconnaître le chiffre écrit par la sourie de l'utilisateur.

## Détails

#### 1er Rendu

Ce rendu présente un programme permettant de dessiner avec la souris dans une fenêtre. 
Lorsque la fenêtre est fermée, l'image est enregistrée et traitée pour être conforme aux normes MNIST.

#### 2ème Rendu

Ce rendu inclut un programme pour effectuer une régression linéaire simple. 
Il affiche la courbe de projection du modèle créé et la compare aux résultats de la bibliothèque scikit-learn. 
Les indicateurs de performance (MSE et R²) sont également calculés et affichés pour comparaison.

#### 3ème Rendu

Ce rendu améliore le précédent en utilisant une régression polynomiale. 
Une classe `Scaler` est créée pour normaliser les données, et la classe `LinearRegression` est améliorée pour gérer les régressions multiples. 
Différents degrés de polynômes sont comparés pour déterminer le modèle optimal. 
L'affichage final compare ce nouveau modèle de régression polynomiale à l'ancien modèle de régression linéaire simple.

### -Projet en cours-

## Installation

  1 - Clonez le dépot :
	git clone https://github.com/aubin-pct/Entrainement_ia.git
	cd Entrainement_ia
  2 - Configurez l'environnement :
	python -m venv venv
	source venv/bin/activate    # Sous Windows : venv\Scripts\activate
	pip install -r requirements.txt

## Lancement

    un fichier app.py est présent dans chaque rendu.

```
	python rendue_<numero_rendu>/app.py
```
