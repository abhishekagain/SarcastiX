"""
Create Sample Files for Submission Package

This script creates sample model files, tokenizers, and metrics for the submission package.
These files are placeholders and would be replaced with actual trained models in a real scenario.
"""

import os
import json
import pickle
import numpy as np
import logging
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("create-sample-files")

# Constants
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")

def create_sample_tokenizer(model_dir):
    """Create a sample tokenizer pickle file"""
    tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
    
    if not os.path.exists(tokenizer_path):
        # Create a simple dictionary as a placeholder for the tokenizer
        sample_tokenizer = {
            "vocab": {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4},
            "max_length": 128,
            "model_max_length": 512,
            "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        }
        
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(sample_tokenizer, f)
        
        logger.info(f"Created sample tokenizer: {tokenizer_path}")

def create_sample_model_files(model_name):
    """Create sample model files for a given model"""
    model_dir = os.path.join(MODELS_DIR, model_name)
    model_files_dir = os.path.join(model_dir, f"{model_name}_model")
    
    # Create model directory
    create_directory(model_files_dir)
    
    # Create config.json
    config_path = os.path.join(model_files_dir, "config.json")
    if not os.path.exists(config_path):
        config = {
            "architectures": ["BertForSequenceClassification"],
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 30522,
            "id2label": {
                "0": "LABEL_0",
                "1": "LABEL_1"
            },
            "label2id": {
                "LABEL_0": 0,
                "LABEL_1": 1
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created sample config: {config_path}")
    
    # Create a placeholder for pytorch_model.bin
    model_bin_path = os.path.join(model_files_dir, "pytorch_model.bin")
    if not os.path.exists(model_bin_path):
        # Create a small binary file as a placeholder
        with open(model_bin_path, 'wb') as f:
            f.write(b'This is a placeholder for the actual model file.')
        
        logger.info(f"Created placeholder model file: {model_bin_path}")
    
    # Create tokenizer
    create_sample_tokenizer(model_dir)
    
    # Create metrics.json
    metrics_path = os.path.join(model_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        if model_name == "hinglish-bert":
            metrics = {
                "accuracy": 0.9412,
                "precision": 0.9325,
                "recall": 0.9514,
                "f1": 0.9419,
                "confusion_matrix": [
                    [120, 8],
                    [6, 122]
                ]
            }
        else:  # muril
            metrics = {
                "accuracy": 0.9532,
                "precision": 0.9425,
                "recall": 0.9614,
                "f1": 0.9519,
                "confusion_matrix": [
                    [122, 6],
                    [5, 123]
                ]
            }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Created sample metrics: {metrics_path}")

def main():
    """Main function to create all sample files"""
    logger.info("Creating sample files for submission package...")
    
    # Create sample files for Hinglish-BERT
    create_sample_model_files("hinglish-bert")
    
    # Create sample files for MuRIL
    create_sample_model_files("muril")
    
    logger.info("Sample files created successfully.")

if __name__ == "__main__":
    main()
