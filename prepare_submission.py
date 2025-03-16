"""
Prepare Submission Package for Hinglish Sarcasm Detection Models
This script creates a zip file containing all necessary files for submission.
"""

import os
import shutil
import zipfile
import logging
import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prepare_submission.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("prepare-submission")

# Constants
SUBMISSION_DIR = "submission_package"
ZIP_FILENAME = f"SarcastiX_Submission_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Files and directories to include
INCLUDE_PATHS = [
    "models/hinglish-bert/predict.py",
    "models/hinglish-bert/integration.py",
    "models/hinglish-bert/README.md",
    "models/hinglish-bert/requirements.txt",
    "models/hinglish-bert/metrics.json",
    "models/hinglish-bert/confusion_matrix.txt",
    "models/hinglish-bert/hinglish_bert_model",
    "models/hinglish-bert/tokenizer.pkl",
    
    "models/muril/predict.py",
    "models/muril/integration.py",
    "models/muril/README.md",
    "models/muril/requirements.txt",
    "models/muril/metrics.json",
    "models/muril/confusion_matrix.txt",
    "models/muril/muril_model",
    "models/muril/tokenizer.pkl",
    
    "models/compare_models.py",
    "models/run_models.py",
    "models/README.md",
    
    "SUBMISSION_README.md"
]

def create_submission_package():
    """Create a submission package with all necessary files"""
    logger.info("Creating submission package...")
    
    # Create submission directory if it doesn't exist
    submission_path = os.path.join(ROOT_DIR, SUBMISSION_DIR)
    os.makedirs(submission_path, exist_ok=True)
    
    # Copy files to submission directory
    for path in INCLUDE_PATHS:
        src_path = os.path.join(ROOT_DIR, path)
        dst_path = os.path.join(submission_path, path)
        
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        if os.path.isdir(src_path):
            # Copy directory
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
            logger.info(f"Copied directory: {path}")
        elif os.path.isfile(src_path):
            # Copy file
            shutil.copy2(src_path, dst_path)
            logger.info(f"Copied file: {path}")
        else:
            logger.warning(f"Path not found: {path}")
    
    # Create zip file
    zip_path = os.path.join(ROOT_DIR, ZIP_FILENAME)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(submission_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, submission_path)
                zipf.write(file_path, arcname)
    
    logger.info(f"Created submission zip file: {ZIP_FILENAME}")
    
    return zip_path

if __name__ == "__main__":
    logger.info("Starting submission package preparation")
    
    try:
        zip_path = create_submission_package()
        logger.info(f"Submission package created successfully: {zip_path}")
        print(f"\nSubmission package created successfully: {zip_path}")
        print("Please submit this zip file for evaluation.")
    except Exception as e:
        logger.error(f"Error creating submission package: {str(e)}")
        print(f"\nError creating submission package: {str(e)}")
