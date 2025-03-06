# 10ème rendu : Chatbot avec Ollama

## 📌 Description

Ce projet est une application web Flask intégrant un chatbot fonctionnant avec Ollama, une solution de modèles d'IA en local. L'interface permet aux utilisateurs de converser avec l'IA, de suivre l'historique des discussions et de choisir parmi différents modèles pré-entraînés.

## 🚀 Fonctionnalités

* Envoi de requêtes à un modèle IA exécuté en local via Ollama.
* Suivi de l'historique des conversations.
* Possibilité de changer dynamiquement de modèle.
* Interface web simple et intuitive.
* Création de nouveaux modèles personnalisés avec des instructions système spécifiques.

## 🛠 Technologies utilisées

* **Backend** : Flask (Python)
* **Frontend** : HTML, Tailwind CSS, JavaScript
* **Modèle IA** : Ollama (modèles deepseek-r1)

## 📥 Installation

### 🔹 Prérequis

* Python 3.8+
* Flask
* Ollama
* joblib

### 🔹 Instructions

1. Cloner le dépôt :
   ```
   git clone https://github.com/aubin-pct/Entrainement_ia.git
   cd Entrainement_ia
   ```
2. Installer les dépendances :
   ```
   python -m venv venv
   source venv/bin/activate    # Sous Windows : venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Lancer l'application Flask :
   ```
   cd rendu_10
   flask run
   ```
4. Accéder à l'application via le navigateur :
   ```
   http://127.0.0.1:5000/
   ```

## 📂 Structure du projet

```
/
├── templates/
│   ├── general.html   # Modèle de page générale
│   ├── general/
│   │   ├── header.html
│   ├── pages/
│   │   ├── index.html    # Page d'accueil du chatbot
│   │   ├── config.html   # Page de configuration
├── app.py            # Code principal de l'application Flask
├── README.md         # Documentation du projet
```

## 🎯 Utilisation

* Accédez à la page d'accueil pour utiliser le chatbot.
* Changez de modèle via le menu déroulant.
* Consultez l'historique de la discussion.
* Allez sur la page **Configuration** pour créer un modèle personnalisé.

## 🔮 Améliorations possibles

* Ajout d'un stockage persistant pour l'historique des conversations.
* Intégration d'autres modèles IA pour plus de flexibilité.

## 👥 Auteurs

* Aubin PERCHET - Développement et conception du projet.
