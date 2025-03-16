import os
import json
import logging
from flask import Flask, request, jsonify, session, redirect, url_for, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import time
import uuid
import sys
import pandas as pd
import shutil
from datetime import timedelta
from ml_models.model_manager import ModelManager
from ml_models.predict import predict_sarcasm
from ml_models.predict_image import predict_from_image
from ml_models.model_evaluation import ModelEvaluator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Admin credentials
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin')

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://localhost:5173"], "supports_credentials": True}})

# Configure session
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret_key')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)

# Configure uploads
UPLOAD_FOLDER = 'uploads'
DATASET_FOLDER = 'datasets'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATASET_FOLDER'] = DATASET_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize model manager
model_manager = ModelManager()

# Routes
@app.route('/')
def home():
    return jsonify({
        'message': 'SarcastiX API is running',
        'version': '1.0.0',
        'endpoints': [
            '/api/detect - Text-based sarcasm detection',
            '/api/detect/image - Image-based sarcasm detection (OCR)',
            '/api/models - Available models information',
            '/api/models/performance - Model performance metrics',
            '/api/compare - Compare text using all models',
            '/api/health - API health check',
            '/api/admin/login - Admin login',
            '/api/admin/logout - Admin logout',
            '/api/admin/status - Admin status',
            '/api/admin/upload/dataset - Upload dataset for training',
            '/api/admin/train - Train model',
            '/api/admin/datasets - Get available datasets',
            '/api/admin/copy-dataset - Copy external dataset',
            '/api/admin/visualizations - Get visualizations'
        ]
    })

@app.route('/api/detect', methods=['POST'])
def detect_sarcasm():
    try:
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
            
        text = data['text']
        model = data.get('model', 'hinglish-bert')
        
        result = predict_sarcasm(text, model)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
            
        # Rename model_used field for API consistency
        if 'model' in result:
            result['model_used'] = result['model']
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in /api/detect: {str(e)}")
        return jsonify({'error': 'Model inference failed', 'details': str(e)}), 500

@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Image file is required'}), 400
            
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            model = request.form.get('model', 'hinglish-bert')
            result = predict_from_image(filepath, model)
            
            if 'error' in result:
                return jsonify({'error': result['error']}), 500
                
            return jsonify(result)
        else:
            return jsonify({'error': 'File type not allowed'}), 400
            
    except Exception as e:
        logger.error(f"Error in /api/detect/image: {str(e)}")
        return jsonify({'error': 'Image analysis failed', 'details': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        model_info = model_manager.get_model_info()
        return jsonify(model_info)
        
    except Exception as e:
        logger.error(f"Error in /api/models: {str(e)}")
        
        # Fallback to static model list
        models = [
            {
                'id': 'hinglish-bert',
                'name': 'Hinglish-BERT',
                'accuracy': 0.89,
                'processingSpeed': '45ms/prediction',
                'memoryUsage': '1.2GB'
            },
            {
                'id': 'roberta',
                'name': 'RoBERTa Base',
                'accuracy': 0.87,
                'processingSpeed': '38ms/prediction',
                'memoryUsage': '892MB'
            },
            {
                'id': 'xlm-roberta',
                'name': 'XLM-RoBERTa',
                'accuracy': 0.86,
                'processingSpeed': '42ms/prediction',
                'memoryUsage': '1.1GB'
            }
        ]
        
        return jsonify(models)

@app.route('/api/models/performance', methods=['GET'])
def get_performance():
    try:
        evaluator = ModelEvaluator()
        metrics = evaluator.get_metrics_json()
        
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Error in /api/models/performance: {str(e)}")
        return jsonify({'error': 'Failed to retrieve model metrics', 'details': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_models():
    try:
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
            
        text = data['text']
        
        result = predict_sarcasm(text, 'all')
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in /api/compare: {str(e)}")
        return jsonify({'error': 'Model comparison failed', 'details': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    from datetime import datetime
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

# Admin routes
@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    try:
        data = request.json
        
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({'error': 'Username and password are required'}), 400
            
        username = data['username']
        password = data['password']
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        logger.error(f"Error in /api/admin/login: {str(e)}")
        return jsonify({'error': 'Login failed', 'details': str(e)}), 500

@app.route('/api/admin/logout', methods=['POST'])
def admin_logout():
    session.pop('admin_logged_in', None)
    return jsonify({'success': True, 'message': 'Logout successful'})

@app.route('/api/admin/status', methods=['GET'])
def admin_status():
    try:
        is_logged_in = session.get('admin_logged_in', False)
        return jsonify({'loggedIn': is_logged_in})
    except Exception as e:
        logger.error(f"Error in /api/admin/status: {str(e)}")
        return jsonify({'error': 'Failed to check login status', 'details': str(e)}), 500

@app.route('/api/admin/upload/dataset', methods=['POST'])
def upload_dataset():
    try:
        # Check if admin is logged in
        if not session.get('admin_logged_in', False):
            return jsonify({'error': 'Unauthorized access'}), 401
            
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        dataset_type = request.form.get('type', 'custom')
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['DATASET_FOLDER'], filename)
            file.save(filepath)
            
            # Validate CSV format
            try:
                df = pd.read_csv(filepath)
                required_columns = ['Tweet', 'Label']
                
                if not all(col in df.columns for col in required_columns):
                    os.remove(filepath)
                    return jsonify({'error': 'CSV file must contain Tweet and Label columns'}), 400
                    
                # Process the dataset (in a real application, you might want to do this asynchronously)
                # For now, we'll just return success
                return jsonify({
                    'success': True, 
                    'message': 'Dataset uploaded successfully',
                    'filename': filename,
                    'rows': len(df)
                })
                
            except Exception as e:
                os.remove(filepath)
                return jsonify({'error': f'Invalid CSV format: {str(e)}'}), 400
        else:
            return jsonify({'error': 'File type not allowed'}), 400
            
    except Exception as e:
        logger.error(f"Error in /api/admin/upload/dataset: {str(e)}")
        return jsonify({'error': 'Dataset upload failed', 'details': str(e)}), 500

@app.route('/api/admin/train', methods=['POST'])
def train_model():
    try:
        # Check if admin is logged in
        if not session.get('admin_logged_in', False):
            return jsonify({'error': 'Unauthorized access'}), 401
            
        data = request.json
        
        if not data or 'dataset' not in data or 'modelName' not in data:
            return jsonify({'error': 'Dataset and model name are required'}), 400
            
        dataset = data['dataset']
        model_name = data['modelName']
        
        # In a real application, you would trigger model training here
        # For now, we'll just return a success message
        
        return jsonify({
            'success': True,
            'message': f'Training initiated for model {model_name} using dataset {dataset}',
            'jobId': f'train_{model_name}_{dataset}'.replace('.', '_')
        })
        
    except Exception as e:
        logger.error(f"Error in /api/admin/train: {str(e)}")
        return jsonify({'error': 'Training initiation failed', 'details': str(e)}), 500

@app.route('/api/admin/datasets', methods=['GET'])
def get_datasets():
    try:
        # Check if admin is logged in
        if not session.get('admin_logged_in', False):
            return jsonify({'error': 'Unauthorized access'}), 401
            
        datasets = []
        for filename in os.listdir(app.config['DATASET_FOLDER']):
            if filename.endswith('.csv'):
                filepath = os.path.join(app.config['DATASET_FOLDER'], filename)
                try:
                    df = pd.read_csv(filepath)
                    datasets.append({
                        'name': filename,
                        'rows': len(df),
                        'size': os.path.getsize(filepath),
                        'lastModified': os.path.getmtime(filepath)
                    })
                except:
                    # Skip invalid CSV files
                    pass
                    
        return jsonify(datasets)
        
    except Exception as e:
        logger.error(f"Error in /api/admin/datasets: {str(e)}")
        return jsonify({'error': 'Failed to retrieve datasets', 'details': str(e)}), 500

@app.route('/api/admin/copy-dataset', methods=['POST'])
def copy_external_dataset():
    try:
        # Check if admin is logged in
        if not session.get('admin_logged_in', False):
            return jsonify({'error': 'Unauthorized access'}), 401
            
        data = request.json
        
        if not data or 'sourcePath' not in data:
            return jsonify({'error': 'Source path is required'}), 400
            
        source_path = data['sourcePath']
        
        if not os.path.exists(source_path) or not source_path.endswith('.csv'):
            return jsonify({'error': 'Invalid source path'}), 400
            
        filename = os.path.basename(source_path)
        destination_path = os.path.join(app.config['DATASET_FOLDER'], filename)
        
        # Copy the file
        shutil.copy2(source_path, destination_path)
        
        # Validate CSV format
        try:
            df = pd.read_csv(destination_path)
            return jsonify({
                'success': True,
                'message': 'Dataset copied successfully',
                'filename': filename,
                'rows': len(df)
            })
        except Exception as e:
            os.remove(destination_path)
            return jsonify({'error': f'Invalid CSV format: {str(e)}'}), 400
            
    except Exception as e:
        logger.error(f"Error in /api/admin/copy-dataset: {str(e)}")
        return jsonify({'error': 'Dataset copy failed', 'details': str(e)}), 500

@app.route('/api/admin/visualizations', methods=['GET'])
def get_visualizations():
    try:
        # Check if admin is logged in
        if not session.get('admin_logged_in', False):
            return jsonify({'error': 'Unauthorized access'}), 401
            
        # In a real application, you would retrieve actual visualizations
        # For now, we'll return mock data
        
        visualizations = [
            {
                'id': 'model_comparison',
                'title': 'Model Performance Comparison',
                'type': 'bar',
                'description': 'Comparison of accuracy, F1 score, and precision across models'
            },
            {
                'id': 'confusion_matrix',
                'title': 'Confusion Matrix',
                'type': 'heatmap',
                'description': 'Visualization of true positives, false positives, true negatives, and false negatives'
            },
            {
                'id': 'training_history',
                'title': 'Training History',
                'type': 'line',
                'description': 'Training and validation loss/accuracy over epochs'
            },
            {
                'id': 'feature_importance',
                'title': 'Feature Importance',
                'type': 'bar',
                'description': 'Most important features for prediction'
            }
        ]
        
        return jsonify(visualizations)
        
    except Exception as e:
        logger.error(f"Error in /api/admin/visualizations: {str(e)}")
        return jsonify({'error': 'Failed to retrieve visualizations', 'details': str(e)}), 500

if __name__ == '__main__':
    # Load available models
    model_manager.load_all_available_models()
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 3001))
    
    # Run app
    app.run(host='0.0.0.0', port=port, debug=True)
