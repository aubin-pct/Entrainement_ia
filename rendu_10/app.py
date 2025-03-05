from flask import Flask, request, jsonify, render_template
import ollama
import json
import re



app = Flask(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
ollama.create(model='agent_1', from_='deepseek-r1:1.5b', system="Les réponses sont attendues en francais.")

def get_response_from_ollama(user_input):
    # Get the response from the locally running Ollama model
    response = ollama.chat(model="agent_1", messages=[{"role": "user", "content": user_input}])
    print(response)

    if 'message' in response:
        return response['message']['content']
    else:
        return 'Sorry, something went wrong.'

@app.route("/")
def index():
    return render_template("general.html", chat_history=[], current_page = "index")

@app.route("/<page>/")
def page(page):
    return render_template("general.html", current_page = page)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    data = request.get_json()
    
    #chat_history = data.get("chat_history")
    user_input = data.get("user_input")
    bot_response = get_response_from_ollama(user_input)
    _, think, response = re.split(r'<think>|</think>', bot_response)
    
    rep = {'user_input': user_input, 'bot_think': think, 'bot_response': response}
    
    return rep

if __name__ == "__main__":
    app.run(port=5000, debug=True)
