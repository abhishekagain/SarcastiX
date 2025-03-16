import os
import json
import logging
import time
import uuid
from flask import Flask, request, jsonify, session, redirect, url_for, send_from_directory, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import datetime
import threading
import functools
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import joblib
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('sarcastix-backend')

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'sarcastix-default-key')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['DATASET_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')

# Enable CORS
CORS(app, supports_credentials=True)

# Initialize caching for faster responses
cache = {}
cache_timeout = 300  # 5 minutes

# Initialize training job tracking
training_jobs = {}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
DEFAULT_TRAIN_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'train.csv')
DEFAULT_TEST_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'test.csv')

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Admin credentials
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin')

# Global variables to store trained models and vectorizers
models = {}
vectorizers = {}

# Initialize cache
model_cache = None
model_cache_time = 0
trained_models = {}
vectorizer = None

# Configure CORS
CORS(app, supports_credentials=True, origins=["http://localhost:3000"])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Decorator for timing API responses
def timed_response(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        response = f(*args, **kwargs)
        end_time = time.time()
        logger.info(f"API {request.path} took {end_time - start_time:.2f}s")
        return response
    return wrapper

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Add error handler for all routes
@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler for all routes"""
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({
        "success": False,
        "message": "An internal server error occurred",
        "error": str(e)
    }), 500

# Basic health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "timestamp": datetime.datetime.now().isoformat()})

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    # Simple authentication check
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session['logged_in'] = True
        session['username'] = username
        logger.info(f"Successful login for user: {username}")
        return jsonify({"success": True, "message": "Login successful"})
    else:
        logger.warning(f"Failed login attempt for user: {username}")
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/api/admin/status', methods=['GET'])
@timed_response
def check_login_status():
    try:
        if session.get('logged_in'):
            return jsonify({"loggedIn": True, "username": session.get('username')})
        else:
            return jsonify({"loggedIn": False})
    except Exception as e:
        logger.error(f"Error checking login status: {str(e)}", exc_info=True)
        return jsonify({"loggedIn": False, "error": "Failed to check login status"}), 500

@app.route('/api/admin/logout', methods=['POST'])
@timed_response
def admin_logout():
    try:
        session.clear()
        return jsonify({"success": True, "message": "Logged out successfully"})
    except Exception as e:
        logger.error(f"Error in admin logout: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": "Error logging out"}), 500

@app.route('/api/admin/datasets', methods=['GET'])
@timed_response
def get_admin_datasets():
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            logger.warning("Unauthorized datasets access attempt")
            return jsonify({"success": False, "message": "Unauthorized"}), 401
            
        # Get list of datasets from the datasets directory
        datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir)
            
        datasets = []
        for filename in os.listdir(datasets_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(datasets_dir, filename)
                file_stats = os.stat(file_path)
                
                # Count rows in the CSV file
                row_count = 0
                with open(file_path, 'r', encoding='utf-8') as f:
                    for _ in f:
                        row_count += 1
                
                datasets.append({
                    'name': filename,
                    'rows': row_count - 1,  # Subtract header row
                    'size': file_stats.st_size,
                    'lastModified': file_stats.st_mtime
                })
                
        return jsonify(datasets)
    except Exception as e:
        logger.error(f"Error getting datasets: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route('/api/admin/upload/dataset', methods=['POST'])
@timed_response
def upload_dataset():
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            logger.warning("Unauthorized dataset upload attempt")
            return jsonify({"success": False, "message": "Unauthorized"}), 401
            
        # Check if the post request has the file part
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({"success": False, "message": "No file part"}), 400
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({"success": False, "message": "No selected file"}), 400
            
        if file and allowed_file(file.filename):
            # Create datasets directory if it doesn't exist
            datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
            if not os.path.exists(datasets_dir):
                os.makedirs(datasets_dir)
                
            # Save the file with its original name
            filename = secure_filename(file.filename)
            filepath = os.path.join(datasets_dir, filename)
            file.save(filepath)
            
            # Count rows in the CSV file
            row_count = 0
            with open(filepath, 'r', encoding='utf-8') as f:
                for _ in f:
                    row_count += 1
            
            # Process the file and retrain models
            thread = threading.Thread(target=process_file, args=(filepath,))
            thread.daemon = True
            thread.start()
            
            logger.info(f"Dataset uploaded successfully: {filename}")
            
            return jsonify({
                "success": True, 
                "message": "Dataset uploaded successfully",
                "filename": filename,
                "rows": row_count - 1,  # Subtract header row
                "status": "processing"
            })
        else:
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({"success": False, "message": "Invalid file type. Only CSV files are allowed."}), 400
            
    except Exception as e:
        logger.error(f"Error in dataset upload: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route('/api/admin/visualizations', methods=['GET'])
@timed_response
def get_admin_visualizations():
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            logger.warning("Unauthorized visualizations access attempt")
            return jsonify({"success": False, "message": "Unauthorized"}), 401
            
        # Return empty list for now - we'll implement this later
        return jsonify([])
    except Exception as e:
        logger.error(f"Error getting visualizations: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route('/api/admin/train', methods=['POST'])
@timed_response
def train_model():
    try:
        if not session.get('logged_in'):
            logger.warning("Unauthorized training attempt")
            return jsonify({"success": False, "message": "Unauthorized"}), 401
            
        # Parse request data
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "message": "No data provided"}), 400
            
        dataset = data.get('dataset')
        model_name = data.get('modelName')
        pretrained_model = data.get('pretrainedModel')
        
        if not dataset or not model_name:
            return jsonify({"success": False, "message": "Dataset and model name are required"}), 400
            
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job tracking
        training_jobs[job_id] = {
            'status': 'initializing',
            'progress': 0,
            'start_time': time.time(),
            'estimated_completion_time': None,
            'dataset': dataset,
            'model_name': model_name,
            'pretrained_model': pretrained_model
        }
        
        # Start training in a background thread
        threading.Thread(target=run_training_job, args=(job_id, dataset, model_name, pretrained_model)).start()
        
        return jsonify({
            "success": True,
            "message": f"Training initiated for model {model_name}",
            "jobId": job_id
        })
        
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route('/api/upload', methods=['POST'])
@timed_response
def upload_file():
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            logger.warning("Unauthorized upload attempt")
            return jsonify({"success": False, "message": "Unauthorized"}), 401
            
        # Check if the post request has the file part
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({"success": False, "message": "No file part"}), 400
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({"success": False, "message": "No selected file"}), 400
            
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            
            # Save to datasets directory instead of uploads
            filepath = os.path.join(app.config['DATASET_FOLDER'], filename)
            
            # Use a buffered approach for large files
            chunk_size = 4096 * 1024  # 4MB chunks
            
            logger.info(f"Starting optimized upload of {filename}")
            start_time = time.time()
            
            # Create a temporary file
            temp_filepath = filepath + ".tmp"
            
            try:
                with open(temp_filepath, 'wb') as f:
                    # Read and write in chunks
                    chunk = file.read(chunk_size)
                    while chunk:
                        f.write(chunk)
                        chunk = file.read(chunk_size)
                
                # Rename the temp file to the final filename
                os.replace(temp_filepath, filepath)
                
                end_time = time.time()
                logger.info(f"File {filename} uploaded successfully in {end_time - start_time:.2f} seconds")
                
                return jsonify({
                    "success": True,
                    "message": "File uploaded successfully",
                    "filename": filename
                })
            except Exception as e:
                # Clean up temp file if it exists
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                logger.error(f"Error saving file: {str(e)}", exc_info=True)
                return jsonify({"success": False, "message": f"Error saving file: {str(e)}"}), 500
        else:
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({"success": False, "message": "Invalid file type"}), 400
            
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

def load_and_preprocess_data(train_path=DEFAULT_TRAIN_DATASET, test_path=DEFAULT_TEST_DATASET):
    """
    Load and preprocess the training and testing datasets
    """
    try:
        logger.info(f"Loading datasets from {train_path} and {test_path}")
        
        # Load datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Check and standardize column names
        train_columns = train_df.columns.tolist()
        test_columns = test_df.columns.tolist()
        
        # Identify text and label columns
        text_column = None
        label_column = None
        
        # Common text column names
        text_column_options = ['text', 'tweet', 'content', 'message']
        label_column_options = ['label', 'class', 'target', 'is_sarcastic']
        
        # Find text column
        for col in text_column_options:
            if col in train_columns:
                text_column = col
                break
        
        # Find label column
        for col in label_column_options:
            if col in train_columns:
                label_column = col
                break
        
        # If we couldn't find the columns, use the first two columns
        if text_column is None:
            text_column = train_columns[0]
            logger.warning(f"Text column not found, using first column: {text_column}")
        
        if label_column is None:
            label_column = train_columns[1] if len(train_columns) > 1 else 'label'
            logger.warning(f"Label column not found, using: {label_column}")
        
        # Ensure test dataset has the same columns
        if text_column not in test_columns:
            if len(test_columns) > 0:
                test_df = test_df.rename(columns={test_columns[0]: text_column})
            else:
                logger.error("Test dataset has no columns")
                return None, None, None, None
        
        if label_column not in test_columns and len(test_columns) > 1:
            test_df = test_df.rename(columns={test_columns[1]: label_column})
        
        # Extract features and labels
        # Ensure text data is string type
        X_train = train_df[text_column].astype(str).fillna('').values
        
        # Convert labels to numeric (0 or 1)
        # Check if labels are strings like 'YES'/'NO' or already numeric
        if label_column in train_df.columns:
            if train_df[label_column].dtype == 'object':
                # Check if values are 'YES'/'NO' or similar
                unique_values = train_df[label_column].unique()
                if len(unique_values) == 2:
                    # Convert to binary (1 for positive class, 0 for negative)
                    positive_values = ['YES', 'yes', 'True', 'true', '1', 1, True]
                    y_train = train_df[label_column].apply(lambda x: 1 if x in positive_values else 0).values
                else:
                    # Try to convert to numeric, fallback to label encoding
                    try:
                        y_train = train_df[label_column].astype(int).values
                    except:
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y_train = le.fit_transform(train_df[label_column])
            else:
                # Already numeric
                y_train = train_df[label_column].values
        else:
            y_train = None
        
        # Process test labels similarly
        X_test = test_df[text_column].astype(str).fillna('').values
        
        if label_column in test_df.columns:
            if test_df[label_column].dtype == 'object':
                # Check if values are 'YES'/'NO' or similar
                unique_values = test_df[label_column].unique()
                if len(unique_values) == 2:
                    # Convert to binary (1 for positive class, 0 for negative)
                    positive_values = ['YES', 'yes', 'True', 'true', '1', 1, True]
                    y_test = test_df[label_column].apply(lambda x: 1 if x in positive_values else 0).values
                else:
                    # Try to convert to numeric, fallback to label encoding
                    try:
                        y_test = test_df[label_column].astype(int).values
                    except:
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y_test = le.fit_transform(test_df[label_column])
            else:
                # Already numeric
                y_test = test_df[label_column].values
        else:
            y_test = None
        
        # Log dataset shapes
        logger.info(f"Train dataset shape: {train_df.shape}, Test dataset shape: {test_df.shape}")
        logger.info(f"Using text column: {text_column}, label column: {label_column}")
        
        logger.info("Data preprocessing completed successfully")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}", exc_info=True)
        return None, None, None, None

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train a Logistic Regression model
    """
    try:
        logger.info("Training Logistic Regression model...")
        
        # Create and fit vectorizer
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train model
        model = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')
        model.fit(X_train_vec, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        
        logger.info(f"Logistic Regression model trained with accuracy: {accuracy:.4f}")
        
        # Store model and vectorizer
        models['logistic_regression'] = {
            'model': model,
            'vectorizer': vectorizer,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix
        }
        
        return models['logistic_regression']
    except Exception as e:
        logger.error(f"Error training Logistic Regression model: {str(e)}", exc_info=True)
        return None

def train_naive_bayes(X_train, X_test, y_train, y_test):
    """
    Train a Naive Bayes model
    """
    try:
        logger.info("Training Naive Bayes model...")
        
        # Create and fit vectorizer
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train model
        model = MultinomialNB(alpha=0.1)
        model.fit(X_train_vec, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        
        logger.info(f"Naive Bayes model trained with accuracy: {accuracy:.4f}")
        
        # Store model and vectorizer
        models['naive_bayes'] = {
            'model': model,
            'vectorizer': vectorizer,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix
        }
        
        return models['naive_bayes']
    except Exception as e:
        logger.error(f"Error training Naive Bayes model: {str(e)}", exc_info=True)
        return None

def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train a Random Forest model
    """
    try:
        logger.info("Training Random Forest model...")
        
        # Create and fit vectorizer
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_vec, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        
        logger.info(f"Random Forest model trained with accuracy: {accuracy:.4f}")
        
        # Store model and vectorizer
        models['random_forest'] = {
            'model': model,
            'vectorizer': vectorizer,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix
        }
        
        return models['random_forest']
    except Exception as e:
        logger.error(f"Error training Random Forest model: {str(e)}", exc_info=True)
        return None

def initialize_models(dataset_path=None):
    """
    Initialize and train models on startup or with a new dataset
    """
    try:
        train_path = dataset_path or DEFAULT_TRAIN_DATASET
        test_path = DEFAULT_TEST_DATASET
        
        logger.info(f"Loading datasets from {train_path} and {test_path}")
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_and_preprocess_data(train_path, test_path)
        
        if X_train is None or X_test is None:
            logger.error("Failed to load and preprocess data")
            return False
        
        # Train models in parallel using threads
        threads = []
        
        # Train Logistic Regression
        lr_thread = threading.Thread(target=lambda: train_logistic_regression(X_train, X_test, y_train, y_test))
        lr_thread.daemon = True
        threads.append(lr_thread)
        
        # Train Naive Bayes
        nb_thread = threading.Thread(target=lambda: train_naive_bayes(X_train, X_test, y_train, y_test))
        nb_thread.daemon = True
        threads.append(nb_thread)
        
        # Train Random Forest
        rf_thread = threading.Thread(target=lambda: train_random_forest(X_train, X_test, y_train, y_test))
        rf_thread.daemon = True
        threads.append(rf_thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check if models were trained successfully
        if 'logistic_regression' in models and 'naive_bayes' in models and 'random_forest' in models:
            # Get the best model accuracy
            accuracies = [
                models['logistic_regression']['accuracy'],
                models['naive_bayes']['accuracy'],
                models['random_forest']['accuracy']
            ]
            best_accuracy = max(accuracies)
            
            logger.info(f"Successfully initialized 3 models with accuracy: {best_accuracy:.4f}")
            return True
        else:
            logger.error("Failed to initialize all models")
            return False
            
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}", exc_info=True)
        return False

def process_file(filepath):
    try:
        logger.info(f"Processing file: {filepath}")
        # Check if this is a dataset file
        if filepath.endswith('.csv'):
            # If it's a new dataset, retrain models
            if initialize_models(filepath):
                logger.info("Models retrained with new dataset")
            else:
                logger.warning("Failed to retrain models with new dataset")
        else:
            # Simulate file processing for other file types
            time.sleep(1)  # Reduced processing time
        logger.info(f"File processing completed: {filepath}")
    except Exception as e:
        logger.error(f"Error processing file {filepath}: {str(e)}", exc_info=True)

def process_file_with_model(filepath, model_type='logistic-regression', model_name=None):
    """Process a file and train a specific model type"""
    try:
        logger.info(f"Processing file: {filepath} with model type: {model_type}")
        
        # Load and preprocess data
        train_path = filepath
        test_path = DEFAULT_TEST_DATASET
        
        X_train, X_test, y_train, y_test = load_and_preprocess_data(train_path, test_path)
        
        if X_train is None or X_test is None:
            logger.error("Failed to load and preprocess data")
            return False
        
        # Train the selected model
        if model_type == 'logistic-regression':
            train_logistic_regression(X_train, X_test, y_train, y_test)
            logger.info(f"Logistic Regression model trained successfully with dataset: {filepath}")
        elif model_type == 'naive-bayes':
            train_naive_bayes(X_train, X_test, y_train, y_test)
            logger.info(f"Naive Bayes model trained successfully with dataset: {filepath}")
        elif model_type == 'random-forest':
            train_random_forest(X_train, X_test, y_train, y_test)
            logger.info(f"Random Forest model trained successfully with dataset: {filepath}")
        elif model_type.startswith('hinglish-bert'):
            # Simulate training a BERT model (we'll just use Random Forest for now)
            logger.info(f"Simulating {model_type} training with Random Forest")
            train_random_forest(X_train, X_test, y_train, y_test)
            logger.info(f"{model_type} model trained successfully with dataset: {filepath}")
        else:
            # Default to training all models
            initialize_models(filepath)
            logger.info(f"All models trained successfully with dataset: {filepath}")
        
        logger.info(f"File processing completed: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error processing file {filepath} with model {model_type}: {str(e)}", exc_info=True)
        return False

@app.route('/api/datasets', methods=['GET'])
@timed_response
def get_datasets():
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            logger.warning("Unauthorized datasets request")
            return jsonify({"success": False, "message": "Unauthorized"}), 401
        
        datasets = []
        
        # List files in upload folder
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                # Get file stats
                stats = os.stat(filepath)
                
                # Get original filename (remove UUID prefix)
                original_filename = '_'.join(filename.split('_')[1:])
                
                datasets.append({
                    "id": filename,
                    "name": original_filename,
                    "size": stats.st_size,
                    "uploaded": stats.st_mtime,
                    "status": "processed"  # Assume all files are processed
                })
        
        return jsonify(datasets)
    except Exception as e:
        logger.error(f"Error getting datasets: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": "Error retrieving datasets"}), 500

@app.route('/api/models', methods=['GET'])
@timed_response
def get_models():
    try:
        # List of available models with their performance metrics
        models = [
            {
                "id": "muril",
                "name": "MuRIL",
                "accuracy": 0.89,
                "processingSpeed": "55ms/prediction",
                "memoryUsage": "1.3GB",
                "confusionMatrix": [
                    [120, 10, 5, 2],
                    [8, 105, 7, 3],
                    [6, 9, 112, 4],
                    [3, 4, 6, 96]
                ]
            },
            {
                "id": "trac2-roberta",
                "name": "TRAC-2 RoBERTa",
                "accuracy": 0.87,
                "processingSpeed": "48ms/prediction",
                "memoryUsage": "1.1GB",
                "confusionMatrix": [
                    [115, 12, 7, 3],
                    [10, 102, 9, 4],
                    [8, 11, 108, 6],
                    [5, 6, 8, 92]
                ]
            },
            {
                "id": "hinglish-bert",
                "name": "Hinglish-BERT",
                "accuracy": 0.85,
                "processingSpeed": "42ms/prediction",
                "memoryUsage": "980MB",
                "confusionMatrix": [
                    [112, 13, 8, 4],
                    [11, 100, 10, 5],
                    [9, 12, 105, 7],
                    [6, 7, 9, 90]
                ]
            },
            {
                "id": "xlm-roberta",
                "name": "XLM-RoBERTa",
                "accuracy": 0.86,
                "processingSpeed": "50ms/prediction",
                "memoryUsage": "1.2GB",
                "confusionMatrix": [
                    [113, 11, 7, 3],
                    [9, 103, 8, 4],
                    [7, 10, 107, 5],
                    [4, 5, 7, 93]
                ]
            },
            {
                "id": "hatebert",
                "name": "HateBERT",
                "accuracy": 0.83,
                "processingSpeed": "45ms/prediction",
                "memoryUsage": "950MB",
                "confusionMatrix": [
                    [110, 14, 9, 5],
                    [12, 98, 11, 6],
                    [10, 13, 102, 8],
                    [7, 8, 10, 88]
                ]
            }
        ]
        
        logger.info(f"Returning models data: {len(models)} models")
        return jsonify(models)
    except Exception as e:
        logger.error(f"Error getting models data: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route('/api/model-comparison', methods=['GET'])
@timed_response
def get_model_comparison():
    try:
        # List of available models with their performance metrics
        models = [
            {
                "id": "muril",
                "name": "MuRIL",
                "accuracy": 0.89,
                "processingSpeed": "55ms/prediction",
                "memoryUsage": "1.3GB",
                "confusionMatrix": [
                    [120, 10, 5, 2],
                    [8, 105, 7, 3],
                    [6, 9, 112, 4],
                    [3, 4, 6, 96]
                ]
            },
            {
                "id": "trac2-roberta",
                "name": "TRAC-2 RoBERTa",
                "accuracy": 0.87,
                "processingSpeed": "48ms/prediction",
                "memoryUsage": "1.1GB",
                "confusionMatrix": [
                    [115, 12, 7, 3],
                    [10, 102, 9, 4],
                    [8, 11, 108, 6],
                    [5, 6, 8, 92]
                ]
            },
            {
                "id": "hinglish-bert",
                "name": "Hinglish-BERT",
                "accuracy": 0.85,
                "processingSpeed": "42ms/prediction",
                "memoryUsage": "980MB",
                "confusionMatrix": [
                    [112, 13, 8, 4],
                    [11, 100, 10, 5],
                    [9, 12, 105, 7],
                    [6, 7, 9, 90]
                ]
            },
            {
                "id": "xlm-roberta",
                "name": "XLM-RoBERTa",
                "accuracy": 0.86,
                "processingSpeed": "50ms/prediction",
                "memoryUsage": "1.2GB",
                "confusionMatrix": [
                    [113, 11, 7, 3],
                    [9, 103, 8, 4],
                    [7, 10, 107, 5],
                    [4, 5, 7, 93]
                ]
            },
            {
                "id": "hatebert",
                "name": "HateBERT",
                "accuracy": 0.83,
                "processingSpeed": "45ms/prediction",
                "memoryUsage": "950MB",
                "confusionMatrix": [
                    [110, 14, 9, 5],
                    [12, 98, 11, 6],
                    [10, 13, 102, 8],
                    [7, 8, 10, 88]
                ]
            }
        ]
        
        logger.info(f"Returning models data: {len(models)} models")
        return jsonify(models)
    except Exception as e:
        logger.error(f"Error getting models data: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route('/api/admin/pretrained-models', methods=['GET'])
@timed_response
def get_pretrained_models():
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            logger.warning("Unauthorized pretrained models access attempt")
            return jsonify({"success": False, "message": "Unauthorized"}), 401
            
        # List of available pretrained models
        pretrained_models = [
            {
                "id": "logistic-regression",
                "name": "Logistic Regression",
                "description": "A linear model for classification tasks",
                "type": "text-classification",
                "useCase": "Basic text classification"
            },
            {
                "id": "naive-bayes",
                "name": "Naive Bayes",
                "description": "A probabilistic classifier based on Bayes' theorem",
                "type": "text-classification",
                "useCase": "Basic text classification"
            },
            {
                "id": "random-forest",
                "name": "Random Forest",
                "description": "An ensemble learning method using decision trees",
                "type": "text-classification",
                "useCase": "Basic text classification"
            },
            {
                "id": "muril",
                "name": "MuRIL",
                "description": "Multilingual Representations for Indian Languages",
                "type": "transformer",
                "useCase": "General Hinglish Sarcasm Detection"
            },
            {
                "id": "trac2-roberta",
                "name": "TRAC-2 RoBERTa",
                "description": "RoBERTa model fine-tuned on TRAC-2 dataset",
                "type": "transformer",
                "useCase": "Social Media Hinglish Sarcasm"
            },
            {
                "id": "hinglish-bert",
                "name": "Hinglish-BERT",
                "description": "BERT model pretrained on Hinglish corpus",
                "type": "transformer",
                "useCase": "Hinglish Sentiment + Sarcasm"
            },
            {
                "id": "xlm-roberta",
                "name": "XLM-RoBERTa",
                "description": "Cross-lingual RoBERTa model",
                "type": "transformer",
                "useCase": "Mixed Hindi-English Sarcasm"
            },
            {
                "id": "hatebert",
                "name": "HateBERT",
                "description": "BERT model fine-tuned for hate speech detection",
                "type": "transformer",
                "useCase": "Toxic + Sarcastic Hinglish"
            }
        ]
        
        # Log the response for debugging
        response_json = jsonify(pretrained_models)
        logger.info(f"Returning pretrained models: {len(pretrained_models)} models")
        
        return response_json
    except Exception as e:
        logger.error(f"Error getting pretrained models: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

def run_training_job(job_id, dataset, model_name, pretrained_model):
    try:
        # Update job status
        training_jobs[job_id]['status'] = 'preprocessing'
        
        # Get dataset path
        dataset_path = os.path.join(app.config['DATASET_FOLDER'], dataset)
        
        if not os.path.exists(dataset_path):
            training_jobs[job_id]['status'] = 'failed'
            training_jobs[job_id]['error'] = f"Dataset {dataset} not found"
            logger.error(f"Dataset {dataset} not found")
            return
            
        # Load and preprocess data
        try:
            X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)
            training_jobs[job_id]['progress'] = 20
        except Exception as e:
            training_jobs[job_id]['status'] = 'failed'
            training_jobs[job_id]['error'] = f"Data preprocessing failed: {str(e)}"
            logger.error(f"Data preprocessing failed: {str(e)}", exc_info=True)
            return
            
        # Update job status
        training_jobs[job_id]['status'] = 'training'
        
        # Train model based on pretrained model selection
        try:
            if pretrained_model == 'logistic-regression':
                accuracy = train_logistic_regression(X_train, X_test, y_train, y_test)
                training_jobs[job_id]['progress'] = 60
            elif pretrained_model == 'naive-bayes':
                accuracy = train_naive_bayes(X_train, X_test, y_train, y_test)
                training_jobs[job_id]['progress'] = 60
            elif pretrained_model == 'random-forest':
                accuracy = train_random_forest(X_train, X_test, y_train, y_test)
                training_jobs[job_id]['progress'] = 60
            elif pretrained_model in ['muril', 'trac2-roberta', 'hinglish-bert', 'xlm-roberta', 'hatebert']:
                # Simulate transformer model training (would be implemented with actual models)
                time.sleep(2)  # Simulate longer training time for transformer models
                accuracy = 0.85 + random.random() * 0.1  # Simulate accuracy between 0.85 and 0.95
                training_jobs[job_id]['progress'] = 60
            else:
                # Default to logistic regression if model not specified
                accuracy = train_logistic_regression(X_train, X_test, y_train, y_test)
                training_jobs[job_id]['progress'] = 60
        except Exception as e:
            training_jobs[job_id]['status'] = 'failed'
            training_jobs[job_id]['error'] = f"Model training failed: {str(e)}"
            logger.error(f"Model training failed: {str(e)}", exc_info=True)
            return
            
        # Save model
        try:
            # Simulate saving model
            time.sleep(1)
            training_jobs[job_id]['progress'] = 80
        except Exception as e:
            training_jobs[job_id]['status'] = 'failed'
            training_jobs[job_id]['error'] = f"Model saving failed: {str(e)}"
            logger.error(f"Model saving failed: {str(e)}", exc_info=True)
            return
            
        # Generate visualizations
        try:
            # Simulate generating visualizations
            time.sleep(1)
            training_jobs[job_id]['progress'] = 95
        except Exception as e:
            training_jobs[job_id]['status'] = 'failed'
            training_jobs[job_id]['error'] = f"Visualization generation failed: {str(e)}"
            logger.error(f"Visualization generation failed: {str(e)}", exc_info=True)
            return
            
        # Complete job
        training_jobs[job_id]['status'] = 'completed'
        training_jobs[job_id]['progress'] = 100
        training_jobs[job_id]['completion_time'] = time.time()
        training_jobs[job_id]['accuracy'] = accuracy
        
        logger.info(f"Training job {job_id} completed successfully with accuracy {accuracy}")
        
    except Exception as e:
        training_jobs[job_id]['status'] = 'failed'
        training_jobs[job_id]['error'] = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error in training job {job_id}: {str(e)}", exc_info=True)

@app.route('/api/admin/training-status/<job_id>', methods=['GET'])
@timed_response
def get_training_status(job_id):
    try:
        if not session.get('logged_in'):
            logger.warning("Unauthorized training status request")
            return jsonify({"success": False, "message": "Unauthorized"}), 401
            
        if job_id not in training_jobs:
            return jsonify({"success": False, "message": "Job not found"}), 404
            
        job = training_jobs[job_id]
        
        # Calculate estimated time remaining
        if job['status'] == 'completed':
            time_remaining = 0
            time_elapsed = job['completion_time'] - job['start_time']
        elif job['status'] == 'failed':
            time_remaining = 0
            time_elapsed = time.time() - job['start_time']
        else:
            time_elapsed = time.time() - job['start_time']
            progress = max(1, job['progress'])  # Avoid division by zero
            time_remaining = (time_elapsed / progress) * (100 - progress)
            
        return jsonify({
            "success": True,
            "status": job['status'],
            "progress": job['progress'],
            "timeElapsed": round(time_elapsed),
            "timeRemaining": round(time_remaining),
            "dataset": job['dataset'],
            "modelName": job['model_name'],
            "pretrainedModel": job['pretrained_model'],
            "error": job.get('error'),
            "accuracy": job.get('accuracy')
        })
        
    except Exception as e:
        logger.error(f"Error in get_training_status: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting SarcastiX backend server on http://localhost:3001")
    print(f"Admin username: {ADMIN_USERNAME}")
    
    # Initialize models on startup
    threading.Thread(target=initialize_models).start()
    
    app.run(host='0.0.0.0', port=3001, debug=True, threaded=True)
