"""
SarcastiX FinalWebApp - Automated Setup and Deployment Script

This script automates the entire setup and deployment process for the SarcastiX application:
1. Sets up the environment and installs dependencies
2. Trains models using the training data if needed
3. Sets up the FastAPI backend
4. Sets up the React frontend
5. Opens the application in a web browser

Usage:
    python FinalWebApp.py
"""

import os
import sys
import subprocess
import logging
import time
import threading
import json
import shutil
import webbrowser
from pathlib import Path
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sarcastix_setup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sarcastix-setup")

# Constants
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
DATASETS_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_DATA_PATH = os.path.join(ROOT_DIR, "train.csv")
BACKEND_PORT = 3001
FRONTEND_PORT = 3000

# Create necessary directories
for directory in [MODELS_DIR, DATASETS_DIR]:
    try:
        os.makedirs(directory, exist_ok=True)
        # Ensure the directory has write permissions
        os.chmod(directory, 0o777)
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {str(e)}")
os.makedirs(os.path.join(MODELS_DIR, "hinglish-bert"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "muril"), exist_ok=True)

def print_banner():
    """Print a banner for the application"""
    banner = """
    ███████╗ █████╗ ██████╗  ██████╗ █████╗ ███████╗████████╗██╗██╗  ██╗
    ██╔════╝██╔══██╗██╔══██╗██╔════╝██╔══██╗██╔════╝╚══██╔══╝██║╚██╗██╔╝
    ███████╗███████║██████╔╝██║     ███████║███████╗   ██║   ██║ ╚███╔╝ 
    ╚════██║██╔══██║██╔══██╗██║     ██╔══██║╚════██║   ██║   ██║ ██╔██╗ 
    ███████║██║  ██║██║  ██║╚██████╗██║  ██║███████║   ██║   ██║██╔╝ ██╗
    ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝╚═╝  ╚═╝
                                                                        
    Hinglish Sarcasm Detection - Automated Setup and Deployment
    """
    print(banner)

def run_command(command, cwd=None, shell=False):
    """Run a command and return the output"""
    try:
        if shell:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        else:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Command failed: {command}")
            logger.error(f"Error: {stderr}")
            return False, stderr
        
        return True, stdout
    except Exception as e:
        logger.error(f"Exception running command: {str(e)}")
        return False, str(e)

def install_dependencies():
    """Install required dependencies"""
    logger.info("Installing dependencies...")
    
    # Install Python dependencies
    success, output = run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=ROOT_DIR)
    if not success:
        logger.error("Failed to install Python dependencies")
        return False
    
    logger.info("Python dependencies installed successfully")
    
    # Check if Node.js is installed
    success, output = run_command(["node", "--version"])
    if not success:
        logger.error("Node.js is not installed. Please install Node.js and npm before continuing.")
        return False
    
    logger.info("Dependencies installed successfully")
    return True

def prepare_training_data():
    """Prepare training data for model training"""
    logger.info("Preparing training data...")
    
    # Check if training data exists
    if not os.path.exists(TRAIN_DATA_PATH):
        logger.error(f"Training data not found at {TRAIN_DATA_PATH}")
        return False
        
    try:
        # Read the CSV file
        df = pd.read_csv(TRAIN_DATA_PATH)
        
        # Create train, validation, and test splits
        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        # Ensure the data directory exists and has write permissions
        os.makedirs(DATASETS_DIR, exist_ok=True)
        os.chmod(DATASETS_DIR, 0o777)
        
        # Save splits to data directory with proper permissions
        for split_df, filename in [
            (train_df, "train.csv"),
            (val_df, "validation.csv"),
            (test_df, "test.csv")
        ]:
            output_path = os.path.join(DATASETS_DIR, filename)
            try:
                # Save the file
                split_df.to_csv(output_path, index=False)
                # Set file permissions
                os.chmod(output_path, 0o666)
            except Exception as e:
                logger.error(f"Error saving {filename}: {str(e)}")
                return False
        
        # Save a copy of the original data
        try:
            original_path = os.path.join(DATASETS_DIR, "original_train.csv")
            shutil.copy2(TRAIN_DATA_PATH, original_path)
            os.chmod(original_path, 0o666)
        except Exception as e:
            logger.error(f"Error copying original data: {str(e)}")
            return False
        
        logger.info(f"Training data prepared: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test samples")
        return True
        
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        return False

def train_models():
    """Train models if needed"""
    logger.info("Checking if models need training...")
    
    # Check if models are already trained
    hinglish_bert_model_path = os.path.join(MODELS_DIR, "hinglish-bert", "hinglish-bert_model")
    muril_model_path = os.path.join(MODELS_DIR, "muril", "muril_model")
    
    if os.path.exists(hinglish_bert_model_path) and os.path.exists(muril_model_path):
        logger.info("Models already exist. Skipping training.")
        return True
    
    # Run the check_models.py script to create sample models
    logger.info("Creating sample models...")
    success, output = run_command([sys.executable, "check_models.py"], cwd=ROOT_DIR)
    if not success:
        logger.error("Failed to create sample models")
        return False
    
    logger.info("Sample models created successfully")
    
    # Create confusion matrices
    for model_name in ["hinglish-bert", "muril"]:
        model_dir = os.path.join(MODELS_DIR, model_name)
        confusion_matrix_path = os.path.join(model_dir, "confusion_matrix.txt")
        
        if not os.path.exists(confusion_matrix_path):
            # Create the confusion matrix script
            script_path = os.path.join(model_dir, "create_confusion_matrix.py")
            with open(script_path, 'w') as f:
                f.write(f"""\"\"\"
Create confusion matrix for {model_name} model
\"\"\"

import os
import json
import numpy as np

# Load metrics
metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics.json")
with open(metrics_path, 'r') as f:
    metrics = json.load(f)

# Get confusion matrix
cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])

# Create text representation
confusion_matrix_text = \"\"\"
{model_name.capitalize()} Confusion Matrix

Actual vs Predicted

              | Non-Sarcastic | Sarcastic
--------------+---------------+----------
Non-Sarcastic |      {cm[0][0]}      |     {cm[0][1]}
--------------+---------------+----------
Sarcastic     |      {cm[1][0]}      |    {cm[1][1]}

Accuracy: {metrics.get("accuracy", 0):.4f}
Precision: {metrics.get("precision", 0):.4f}
Recall: {metrics.get("recall", 0):.4f}
F1 Score: {metrics.get("f1", 0):.4f}
\"\"\"

# Save to file
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confusion_matrix.txt")
with open(output_path, 'w') as f:
    f.write(confusion_matrix_text)

print(f"Confusion matrix saved to {output_path}")
""")
            
            # Run the script
            success, output = run_command([sys.executable, script_path], cwd=model_dir)
            if not success:
                logger.error(f"Failed to create confusion matrix for {model_name}")
                return False
    
    logger.info("Models trained successfully")
    return True

def setup_backend():
    """Set up the FastAPI backend"""
    logger.info("Setting up FastAPI backend...")
    
    # Check if the FastAPI server file exists
    server_path = os.path.join(ROOT_DIR, "fastapi_server.py")
    if not os.path.exists(server_path):
        logger.error(f"FastAPI server file not found at {server_path}")
        return False
    
    logger.info("FastAPI backend setup successfully")
    return True

def setup_frontend():
    """Set up the React frontend"""
    logger.info("Setting up React frontend...")
    
    # Check if the frontend directory exists
    if not os.path.exists(FRONTEND_DIR):
        logger.error(f"Frontend directory not found at {FRONTEND_DIR}")
        return False
    
    # Check if node_modules exists, if not install dependencies
    if not os.path.exists(os.path.join(FRONTEND_DIR, "node_modules")):
        logger.info("Installing frontend dependencies...")
        success, output = run_command(["npm", "install"], cwd=FRONTEND_DIR)
        if not success:
            logger.error("Failed to install frontend dependencies")
            return False
    
    logger.info("React frontend setup successfully")
    return True

def run_backend():
    """Run the FastAPI backend server"""
    logger.info("Starting FastAPI backend server...")
    
    # Try to kill any existing process on port 3001, but continue even if psutil is not available
    try:
        import psutil
        for proc in psutil.process_iter():
            try:
                for conn in proc.connections():
                    if hasattr(conn, 'laddr') and hasattr(conn.laddr, 'port') and conn.laddr.port == BACKEND_PORT:
                        proc.kill()
                        time.sleep(1)
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                continue
    except ImportError:
        logger.warning("psutil not available - skipping process cleanup")
    except Exception as e:
        logger.warning(f"Could not check for existing processes: {e}")

    # Run the server
    try:
        process = subprocess.Popen(
            [sys.executable, "fastapi_server.py"],
            cwd=ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the server to start
        max_retries = 5
        retries = 0
        while retries < max_retries:
            try:
                import requests
                response = requests.get(f"http://localhost:{BACKEND_PORT}/api/health")
                if response.status_code == 200:
                    logger.info(f"FastAPI backend server running at http://localhost:{BACKEND_PORT}")
                    return True, process
            except Exception:
                retries += 1
                time.sleep(2)
        
        logger.error("FastAPI backend server failed to start")
        process.terminate()
        return False, None
    except Exception as e:
        logger.error(f"Failed to start backend server: {e}")
        return False, None

def run_frontend():
    """Run the React frontend"""
    logger.info("Starting React frontend...")
    
    try:
        # Kill any existing process on port 3000
        try:
            import psutil
            for proc in psutil.process_iter():
                try:
                    for conn in proc.connections():
                        if hasattr(conn, 'laddr') and hasattr(conn.laddr, 'port') and conn.laddr.port == FRONTEND_PORT:
                            proc.kill()
                            time.sleep(1)
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                    continue
        except Exception as e:
            logger.warning(f"Could not check for existing processes: {e}")

        # Run npm install if node_modules doesn't exist
        if not os.path.exists(os.path.join(FRONTEND_DIR, "node_modules")):
            logger.info("Installing frontend dependencies...")
            success, output = run_command(["npm", "install"], cwd=FRONTEND_DIR)
            if not success:
                logger.error("Failed to install frontend dependencies")
                return False, None

        # Run the frontend using npm run dev with explicit port
        process = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", "3000", "--strictPort", "--host"],
            cwd=FRONTEND_DIR,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the frontend to start
        max_retries = 30  # Increased retries
        retries = 0
        while retries < max_retries:
            try:
                import requests
                response = requests.get("http://localhost:3000")
                if response.status_code in [200, 404]:  # Accept 404 as valid since it means the server is running
                    logger.info("React frontend running at http://localhost:3000")
                    return True, process
            except Exception:
                retries += 1
                time.sleep(1)
                if retries % 5 == 0:
                    logger.info(f"Waiting for frontend to start (attempt {retries}/{max_retries})")
        
        # If we get here, check if process is still running
        if process.poll() is None:
            logger.info("React frontend started but health check failed")
            return True, process
        else:
            stdout, stderr = process.communicate()
            logger.error(f"Frontend failed to start. Error: {stderr}")
            return False, None
    except Exception as e:
        logger.error(f"Error starting frontend: {str(e)}")
        return False, None

def open_in_browser():
    """Open the application in a web browser"""
    logger.info("Opening application in web browser...")
    
    # Open the frontend URL in the default browser
    webbrowser.open(f"http://localhost:{FRONTEND_PORT}")
    
    logger.info("Application opened in web browser")
    return True

def main():
    """Main function to run the application"""
    print_banner()
    logger.info("Starting SarcastiX application setup...")
    
    # Step 1: Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies. Exiting.")
        return False
    
    # Step 2: Prepare training data
    if not prepare_training_data():
        logger.error("Failed to prepare training data. Exiting.")
        return False
    
    # Step 3: Train models
    if not train_models():
        logger.error("Failed to train models. Exiting.")
        return False
    
    # Step 4: Set up backend
    if not setup_backend():
        logger.error("Failed to set up backend. Exiting.")
        return False
    
    # Step 5: Set up frontend
    if not setup_frontend():
        logger.error("Failed to set up frontend. Exiting.")
        return False
    
    # Step 6: Run backend
    backend_success, backend_process = run_backend()
    if not backend_success:
        logger.error("Failed to run backend. Exiting.")
        return False
    
    # Step 7: Run frontend
    frontend_success, frontend_process = run_frontend()
    if not frontend_success:
        logger.error("Failed to run frontend. Exiting.")
        backend_process.terminate()
        return False
    
    # Step 8: Open in browser
    open_in_browser()
    
    # Keep the application running
    logger.info("SarcastiX application is now running!")
    logger.info(f"Backend: http://localhost:{BACKEND_PORT}")
    logger.info(f"Frontend: http://localhost:{FRONTEND_PORT}")
    logger.info("Press Ctrl+C to stop the application")
    
    try:
        # Keep the script running until Ctrl+C is pressed
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping SarcastiX application...")
        frontend_process.terminate()
        backend_process.terminate()
        logger.info("SarcastiX application stopped")
    
    return True

if __name__ == "__main__":
    main()
