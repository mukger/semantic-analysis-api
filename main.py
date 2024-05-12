from flask import Flask, jsonify, request, make_response
from text_processor import TextProcessor
from word_sim_calculator import WordSimCalculator
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": "http://localhost:3001"}})
app.config['CORS_HEADERS'] = 'Content-Type'
DIMENSIONS = 300
MODEL_PATH = 'input/GoogleNews-vectors-negative300.bin'
word_sim_calc = WordSimCalculator(MODEL_PATH, DIMENSIONS)


@app.route('/api/keywords', methods=['POST', 'OPTIONS'])
def determine_key_words():
    response = make_response()
    if request.method == 'OPTIONS':
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
        return response

    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'Field "text" is required'}), 400

    text = request.json.get('text')
    text_processor = TextProcessor()
    keywords = text_processor.find_key_words(text)
    return jsonify(keywords)


@app.route('/api/similarity', methods=['POST', 'OPTIONS'])
def determine_words_similarity():
    response = make_response()
    if request.method == 'OPTIONS':
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
        return response

    data = request.json
    if 'first_word_dict' not in data or 'second_word_dict' not in data:
        return jsonify({'error': 'Fields "first_word_dict" and "second_word_dict" is required'}), 400

    first_word_dict = request.json.get('first_word_dict')
    second_word_dict = request.json.get('second_word_dict')

    similarity = word_sim_calc.calculate_dicts_similarity(first_word_dict, second_word_dict)
    return jsonify({'similarity': similarity})


@app.route('/api/simmatrix', methods=['POST', 'OPTIONS'])
def determine_similarity_matrix():
    response = make_response()
    if request.method == 'OPTIONS':
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
        return response

    data = request.json
    if 'first_word_dict' not in data or 'second_word_dict' not in data:
        return jsonify({'error': 'Fields "first_word_dict" and "second_word_dict" is required'}), 400

    first_word_dict = request.json.get('first_word_dict')
    second_word_dict = request.json.get('second_word_dict')

    simmatrix = word_sim_calc.determine_word_simmatrix(first_word_dict, second_word_dict)
    return jsonify(simmatrix)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
