"""
Script to run the SarcastiX application
This script:
1. Checks if model files exist and creates them if they don't
2. Installs dependencies
3. Runs the FastAPI server
"""

import os
import sys
import subprocess
import logging
import time
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('run-app')

def run_check_models():
    """Run the check_models.py script"""
    logger.info("Checking model files...")
    
    try:
        subprocess.check_call([sys.executable, "check_models.py"])
        logger.info("Model files check completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error checking model files: {str(e)}")
        return False

def install_dependencies():
    """Install required dependencies"""
    logger.info("Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {str(e)}")
        return False

def run_backend():
    """Run the FastAPI backend server"""
    logger.info("Starting FastAPI backend server...")
    
    try:
        # Run the server
        subprocess.check_call([sys.executable, "fastapi_server.py"])
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running backend server: {str(e)}")
        return False
    except KeyboardInterrupt:
        logger.info("Backend server stopped by user")
        return True

def run_frontend():
    """Run the React frontend"""
    logger.info("Starting React frontend...")
    
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    
    if not os.path.exists(frontend_dir):
        logger.error(f"Frontend directory not found: {frontend_dir}")
        return False
    
    try:
        # Change to frontend directory
        os.chdir(frontend_dir)
        
        # Check if node_modules exists
        if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
            logger.info("Installing frontend dependencies...")
            subprocess.check_call(["npm", "install"])
        
        # Start the frontend
        subprocess.check_call(["npm", "start"])
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running frontend: {str(e)}")
        return False
    except KeyboardInterrupt:
        logger.info("Frontend stopped by user")
        return True
    finally:
        # Change back to original directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function to run the application"""
    logger.info("Starting SarcastiX application...")
    
    # Check model files
    if not run_check_models():
        logger.error("Failed to check model files. Exiting.")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies. Exiting.")
        sys.exit(1)
    
    # Run backend and frontend in separate threads
    backend_thread = threading.Thread(target=run_backend)
    frontend_thread = threading.Thread(target=run_frontend)
    
    # Start threads
    backend_thread.start()
    
    # Wait a bit for the backend to start
    time.sleep(5)
    
    frontend_thread.start()
    
    # Wait for threads to finish
    try:
        backend_thread.join()
        frontend_thread.join()
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    
    logger.info("SarcastiX application stopped")

if __name__ == "__main__":
    main()
