from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from model_predictor import LearningStylePredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the predictor
try:
    predictor = LearningStylePredictor()
    logger.info("Learning Style Predictor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize predictor: {e}")
    predictor = None

@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'Learning Style Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'API information',
            '/predict': 'Make learning style predictions',
            '/model-info': 'Get model information',
            '/health': 'Health check'
        },
        'usage': {
            'POST /predict': 'Send user data to get learning style prediction',
            'GET /model-info': 'Get information about available models',
            'GET /health': 'Check if the API is running'
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    if predictor is None:
        return jsonify({'status': 'error', 'message': 'Models not loaded'}), 500
    
    return jsonify({
        'status': 'healthy',
        'message': 'Learning Style Prediction API is running',
        'models_loaded': True
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the trained models"""
    if predictor is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        info = predictor.get_model_info()
        return jsonify({
            'status': 'success',
            'data': info
        })
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make learning style predictions"""
    if predictor is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        # Get input data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'Age', 'Age_Group', 'Gender', 'Education_Level', 'Preferred_Study_Time',
            'Learning_Goal', 'Language_Preference', 'Prefers_Diagrams', 'Likes_Listening',
            'Enjoys_Reading', 'HandsOn_Activities', 'Remembers_Pictures', 'Prefers_Lectures',
            'Writes_Notes', 'Builds_Models', 'Watches_Videos', 'Participates_Discussions',
            'Reads_Textbooks', 'Does_Experiments', 'Draws_Mindmaps', 
            'Listens_MusicWhileStudying', 'Moves_WhileLearning'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields
            }), 400
        
        # Get model type from request (default to 'random_forest')
        model_type = data.get('model_type', 'random_forest')
        if model_type not in ['random_forest']:
            return jsonify({'error': 'Invalid model_type. Must be: random_forest'}), 400
        
        # Make predictions
        predictions = predictor.predict(data, model_type)
        
        # Get learning style descriptions
        results = {}
        for model_name, prediction_data in predictions.items():
            style = prediction_data['prediction']
            description = predictor.get_learning_style_description(style)
            
            results[model_name] = {
                'prediction': prediction_data['prediction'],
                'probabilities': prediction_data['probabilities'],
                'confidence': prediction_data['confidence'],
                'description': description
            }
        
        return jsonify({
            'status': 'success',
            'data': results,
            'input_data': data
        })
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/sample', methods=['GET'])
def sample_prediction():
    """Get a sample prediction with example data"""
    if predictor is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    # Sample input data
    sample_data = {
        'Age': 25,
        'Age_Group': 'Young Adult',
        'Gender': 'Male',
        'Education_Level': 'Undergraduate',
        'Preferred_Study_Time': 'Evening',
        'Learning_Goal': 'Career Growth',
        'Language_Preference': 'English',
        'Prefers_Diagrams': 4,
        'Likes_Listening': 3,
        'Enjoys_Reading': 2,
        'HandsOn_Activities': 5,
        'Remembers_Pictures': 3,
        'Prefers_Lectures': 2,
        'Writes_Notes': 4,
        'Builds_Models': 5,
        'Watches_Videos': 3,
        'Participates_Discussions': 2,
        'Reads_Textbooks': 1,
        'Does_Experiments': 5,
        'Draws_Mindmaps': 2,
        'Listens_MusicWhileStudying': 4,
        'Moves_WhileLearning': 5
    }

    try:
        predictions = predictor.predict(sample_data, 'random_forest')
        results = {}
        for model_name, prediction_data in predictions.items():
            style = prediction_data['prediction']
            description = predictor.get_learning_style_description(style)

            results[model_name] = {
                'prediction': prediction_data['prediction'],
                'probabilities': prediction_data['probabilities'],
                'confidence': prediction_data['confidence'],
                'description': description
            }
        return jsonify({
            'status': 'success',
            'data': results,
            'sample_input': sample_data,
            'note': 'This is a sample prediction. Use POST /predict with your own data for real predictions.'
        })

    except Exception as e:
        logger.error(f"Error making sample prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Check if models exist
    if not os.path.exists('models'):
        print("Models directory not found. Please run model_trainer.py first to train the models.")
        print("You can still start the server, but predictions will fail until models are trained.")

    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 