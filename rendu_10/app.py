from flask import Flask, request, jsonify, render_template, session

from joblib import PrintTime
import ollama
import json
import re



app = Flask(__name__)

# Modele par défaut 
current_model = "deepseek-r1:1.5b"

def get_response_from_ollama(messages):
    # Get the response from the locally running Ollama model
    response = ollama.chat(model=current_model, messages=messages)
    print(response)

    if 'message' in response:
        return response['message']['content']
    else:
        return 'Sorry, something went wrong.'

@app.route("/")
@app.route("/index/")
def index():
    model_list = []
    models = ollama.list()['models']

    # Affichage des noms des modèles
    for model in models:
        model_list.append(model['model'])

    return render_template("general.html", current_page = "index", models = model_list, current_model = current_model)

@app.route("/<page>/")
def page(page):
    return render_template("general.html", current_page = page)

# chat : question -> reflexion + reponse | historique
@app.route("/chat", methods=["GET", "POST"])
def chat():
    data = request.get_json()
    
    user_input = data.get("user_input")
    chat_history = data.get("chat_history")
    chat_history.append({'role': 'user', 'content': user_input})

    bot_response = get_response_from_ollama(chat_history)

    if (bot_response == 'Sorry, something went wrong.'):
        think = ""
        response = 'Sorry, something went wrong.'
    else:
        _, think, response = re.split(r'<think>|</think>', bot_response)
    
    chat_history.append({'role': 'assistant', 'content': response})
    rep = {'user_input': user_input, 'bot_think': think, 'bot_response': response}

    return {'reponse': rep, 'chat_history': chat_history}


@app.route("/create_model", methods=["POST"])
def create_model():
    ollama.create(model=request.form.get("nom"), from_=request.form.get("model"), system=request.form.get("system_message"))
    return render_template("general.html", current_page = "config")


@app.route("/change_model", methods=["POST"])
def change_model():
    global current_model
    current_model = request.json.get("new_model")
    return jsonify({"message": f"Modèle changé pour : {current_model}"})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
