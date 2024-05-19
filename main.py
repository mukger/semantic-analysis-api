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
text_processor = TextProcessor()


@app.route('/api/keywords', methods=['POST'])
def determine_course_keywords():
    data = request.json
    if 'materials' not in data or 'name' not in data:
        return jsonify({'error': 'Fields "materials" and "name" are required'}), 400

    materials = request.json.get('materials')
    name = request.json.get('name')

    course_keywords = text_processor.find_materials_keywords(materials.replace(" / ", " ").replace("/", ""))
    name_keywords = text_processor.find_name_keywords(name.replace(" / ", " ").replace("/", ""))
    course_keywords[name_keywords] = 15

    return jsonify(course_keywords)


@app.route('/api/similarity', methods=['POST'])
def determine_words_similarity():
    data = request.json
    if 'first_word_dict' not in data or 'second_word_dict' not in data:
        return jsonify({'error': 'Fields "first_word_dict" and "second_word_dict" are required'}), 400

    first_word_dict = request.json.get('first_word_dict')
    second_word_dict = request.json.get('second_word_dict')

    similarity = word_sim_calc.calculate_dicts_similarity(first_word_dict, second_word_dict)
    return jsonify({'similarity': similarity})


@app.route('/api/simmatrix', methods=['POST'])
def determine_similarity_matrix():
    data = request.json
    if 'first_word_dict' not in data or 'second_word_dict' not in data:
        return jsonify({'error': 'Fields "first_word_dict" and "second_word_dict" are required'}), 400

    first_word_dict = request.json.get('first_word_dict')
    second_word_dict = request.json.get('second_word_dict')

    simmatrix = word_sim_calc.determine_word_simmatrix(first_word_dict, second_word_dict)
    return jsonify(simmatrix)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
