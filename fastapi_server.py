"""
FastAPI Backend for SarcastiX Hinglish Sarcasm Detection
This server provides API endpoints for the SarcastiX web application.
"""

import os
import sys
import json
import logging
import asyncio
import random
from typing import Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sarcastix_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sarcastix-backend")

# Initialize FastAPI app
app = FastAPI(
    title="SarcastiX API",
    description="API for Hinglish Sarcasm Detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATASETS_DIR = os.path.join(ROOT_DIR, "data")

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

# Add training history storage
training_history = []

# Add training jobs storage
training_jobs: Dict[str, asyncio.Task] = {}

async def run_training(job_id: str, model_name: str, dataset: str, pretrained_model: str) -> None:
    try:
        record = next((r for r in training_history if r["id"] == job_id), None)
        if not record:
            logger.error(f"Training record not found for job {job_id}")
            return

        # Simulate training epochs
        total_epochs = 10
        for epoch in range(total_epochs):
            if record["status"] != "in_progress":
                break

            record["currentEpoch"] = epoch + 1
            
            # Simulate epoch training
            await asyncio.sleep(5)  # Simulate work
            
            # Update progress
            record["progress"] = ((epoch + 1) / total_epochs) * 100
            record["timeElapsed"] = (epoch + 1) * 5  # 5 seconds per epoch
            record["timeRemaining"] = (total_epochs - (epoch + 1)) * 5

        if record["status"] == "in_progress":
            # Training completed successfully
            record["status"] = "completed"
            record["endTime"] = datetime.now().isoformat()
            record["accuracy"] = 0.85 + (random.random() * 0.1)  # Random accuracy between 0.85 and 0.95
            
            # Update model data
            models = await get_models()
            for model in models:
                if model["id"] == pretrained_model:
                    if "trainingHistory" not in model:
                        model["trainingHistory"] = []
                    model["trainingHistory"].append(record)
                    break

    except Exception as e:
        logger.error(f"Error during training for job {job_id}: {str(e)}")
        if record:
            record["status"] = "failed"
            record["error"] = str(e)
            record["endTime"] = datetime.now().isoformat()
    finally:
        # Clean up job
        if job_id in training_jobs:
            del training_jobs[job_id]

# Model definitions
class PredictionRequest(BaseModel):
    text: str
    model: str = "hinglish-bert"

class PredictionResponse(BaseModel):
    text: str
    is_sarcastic: bool
    confidence: float
    model: str

# API Routes
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/api/models")
async def get_models():
    try:
        # Get model data
        models = [
            {
                "id": "muril",
                "name": "MuRIL",
                "accuracy": 0.89,
                "processingSpeed": "Fast",
                "memoryUsage": "Medium",
                "useCase": "Hinglish text classification",
                "confusionMatrix": {
                    "truePositive": 85,
                    "falsePositive": 10,
                    "trueNegative": 80,
                    "falseNegative": 15
                },
                "trainingHistory": [
                    {
                        "id": "train-1",
                        "modelName": "MuRIL",
                        "dataset": "hinglish_train.csv",
                        "startTime": "2025-02-27T10:00:00",
                        "endTime": "2025-02-27T11:30:00",
                        "status": "completed",
                        "accuracy": 0.89,
                        "epochs": 10,
                        "currentEpoch": 10
                    }
                ]
            },
            {
                "id": "hinglish-bert",
                "name": "Hinglish-BERT",
                "accuracy": 0.85,
                "processingSpeed": "Medium",
                "memoryUsage": "High",
                "useCase": "Hinglish sentiment analysis",
                "confusionMatrix": {
                    "truePositive": 82,
                    "falsePositive": 12,
                    "trueNegative": 78,
                    "falseNegative": 18
                },
                "trainingHistory": [
                    {
                        "id": "train-2",
                        "modelName": "Hinglish-BERT",
                        "dataset": "hinglish_train.csv",
                        "startTime": "2025-02-27T12:00:00",
                        "endTime": "2025-02-27T14:00:00",
                        "status": "completed",
                        "accuracy": 0.85,
                        "epochs": 10,
                        "currentEpoch": 10
                    }
                ]
            }
        ]
        return models
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train")
async def train_model(training_request: dict):
    try:
        model_name = training_request.get("modelName")
        dataset = training_request.get("dataset")
        pretrained_model = training_request.get("pretrainedModel")
        
        if not all([model_name, dataset, pretrained_model]):
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        # Validate dataset exists
        dataset_path = os.path.join(DATASETS_DIR, dataset)
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail=f"Dataset {dataset} not found")
        
        # Create training record
        job_id = f"train-{len(training_history) + 1}"
        training_record = {
            "id": job_id,
            "modelName": model_name,
            "dataset": dataset,
            "startTime": datetime.now().isoformat(),
            "status": "in_progress",
            "epochs": 10,
            "currentEpoch": 0,
            "progress": 0,
            "timeElapsed": 0,
            "timeRemaining": 50  # 10 epochs * 5 seconds
        }
        
        training_history.append(training_record)
        
        # Start training task
        task = asyncio.create_task(
            run_training(job_id, model_name, dataset, pretrained_model)
        )
        training_jobs[job_id] = task
        
        return {"jobId": job_id}
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training-status/{job_id}")
async def get_training_status(job_id: str):
    try:
        # Find training record
        record = next((r for r in training_history if r["id"] == job_id), None)
        
        if not record:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        # Check if job is still running
        if job_id in training_jobs:
            task = training_jobs[job_id]
            if task.done():
                if task.exception():
                    record["status"] = "failed"
                    record["error"] = str(task.exception())
                    record["endTime"] = datetime.now().isoformat()
                del training_jobs[job_id]
        
        return record
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/training/{job_id}")
async def delete_training(job_id: str):
    try:
        # Find and remove training record
        record = next((r for r in training_history if r["id"] == job_id), None)
        
        if not record:
            raise HTTPException(status_code=404, detail="Training record not found")
        
        # Cancel ongoing training if active
        if job_id in training_jobs:
            task = training_jobs[job_id]
            if not task.done():
                task.cancel()
            del training_jobs[job_id]
        
        training_history.remove(record)
        
        return {"message": "Training record deleted"}
    except Exception as e:
        logger.error(f"Error deleting training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """Predict sarcasm in text"""
    try:
        # Import prediction module
        sys.path.insert(0, MODELS_DIR)
        from run_models import predict_sarcasm
        
        # Make prediction
        result = predict_sarcasm(request.text, request.model)
        
        return PredictionResponse(
            text=request.text,
            is_sarcastic=result["is_sarcastic"],
            confidence=result["confidence"],
            model=request.model
        )
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if sys.path[0] == MODELS_DIR:
            sys.path.pop(0)

@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Predict sarcasm for a batch of texts"""
    try:
        # Save uploaded file
        file_path = os.path.join(DATASETS_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Import prediction module
        sys.path.insert(0, MODELS_DIR)
        from run_models import predict_batch
        
        # Make predictions
        results = predict_batch(file_path)
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if sys.path[0] == MODELS_DIR:
            sys.path.pop(0)

# Visualization endpoints
@app.get("/api/visualizations/performance")
async def get_model_performance():
    """Get model performance metrics for visualization"""
    try:
        # Sample performance data (replace with actual data from your models)
        return {
            "models": ["BERT", "RoBERTa", "XLM"],
            "metrics": {
                "accuracy": [0.85, 0.87, 0.83],
                "precision": [0.84, 0.86, 0.82],
                "recall": [0.83, 0.85, 0.81],
                "f1": [0.835, 0.855, 0.815]
            }
        }
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/visualizations/confusion")
async def get_confusion_matrix():
    """Get confusion matrix data for visualization"""
    try:
        # Sample confusion matrix data (replace with actual data from your models)
        return {
            "models": {
                "BERT": {
                    "matrix": [[850, 150], [120, 880]],
                    "labels": ["Non-Sarcastic", "Sarcastic"]
                },
                "RoBERTa": {
                    "matrix": [[870, 130], [110, 890]],
                    "labels": ["Non-Sarcastic", "Sarcastic"]
                },
                "XLM": {
                    "matrix": [[830, 170], [140, 860]],
                    "labels": ["Non-Sarcastic", "Sarcastic"]
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting confusion matrix: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print(f"Starting FastAPI server on http://localhost:3001")
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=3001, reload=True)
