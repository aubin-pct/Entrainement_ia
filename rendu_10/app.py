from flask import Flask, request, jsonify, render_template
import ollama


app = Flask(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"


def get_response_from_ollama(user_input):
    # Get the response from the locally running Ollama model
    response = ollama.chat(model="mon_model", messages=[{"role": "user", "content": user_input}])
    print(response)

    if 'message' in response:
        return response['message']['content']
    else:
        return 'Sorry, something went wrong.'

@app.route("/", methods=["GET", "POST"])
def index():
    user_input = ""
    bot_response = ""
    
    if request.method == "POST":
        user_input = request.form["user_input"]
        bot_response = get_response_from_ollama(user_input)
    
    return render_template("index.html", user_input=user_input, bot_response=bot_response)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
