# Modifikasi fungsi chatBot di app.py
from flask import Flask, jsonify, request
from main import chatWithBot, get_random_suggestion   # Pastikan Anda sudah mengimpor fungsi chatWithBot yang sudah dimodifikasi

app = Flask(__name__)

@app.route("/chat", methods=['GET','POST'])
def chatBot():
        chatInput = request.form.get('chatInput')
        return jsonify(chatBotReply=chatWithBot(chatInput))
    
@app.route("/get_suggestions", methods=['GET'])
def get_random_suggestions(num_suggestions=6):
    # Gantilah logika untuk mendapatkan saran respons cepat
    all_patterns = get_random_suggestion(num_suggestions)
    return jsonify(randomSuggestions=all_patterns)
    
if __name__ == '__main__':
    app.run(debug=True)
