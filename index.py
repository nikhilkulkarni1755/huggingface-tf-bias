from flask import Flask, request, jsonify
from bias_classifier import BiasClassifier
import os

app = Flask(__name__)

# Initialize the classifier
classifier = BiasClassifier()

# Load the model when the server starts
if os.path.exists('./bias_classifier_model'):
    classifier.load_model()
    print("Loaded existing model")
else:
    print("No existing model found. Training new model...")
    classifier.train_model()
    classifier.load_model()
    print("Model trained and loaded")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Bias Classification API',
        'endpoints': {
            'POST /predict': 'Classify single text',
            'POST /predict_batch': 'Classify multiple texts',
            'GET /health': 'Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': classifier.classifier is not None})

@app.route('/predict', methods=['POST'])
def predict_single():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Please provide text in JSON format: {"text": "your text here"}'}), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        result = classifier.predict(text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Please provide texts in JSON format: {"texts": ["text1", "text2"]}'}), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({'error': 'texts must be a list'}), 400
        
        if len(texts) == 0:
            return jsonify({'error': 'texts list cannot be empty'}), 400
        
        # Limit batch size for performance
        if len(texts) > 50:
            return jsonify({'error': 'Maximum batch size is 50 texts'}), 400
        
        results = classifier.batch_predict(texts)
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6000)