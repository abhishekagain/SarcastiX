import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

class ModelTrainer:
    def __init__(self, train_path, test_path):
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        self.models = {}
        self.histories = {}
        self.results = {}
        
        # Create directories for saving models and visualizations
        os.makedirs('models', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
        
    def preprocess_data(self, text_column='text', label_column='label'):
        # Basic text preprocessing
        self.train_data[text_column] = self.train_data[text_column].str.lower()
        self.test_data[text_column] = self.test_data[text_column].str.lower()
        
        # Convert labels to numpy arrays
        self.train_labels = np.array(self.train_data[label_column])
        self.test_labels = np.array(self.test_data[label_column])
        
        print("Data preprocessing completed")
        
    def train_hinglish_bert(self):
        print("Training Hinglish-BERT model...")
        tokenizer = AutoTokenizer.from_pretrained("monsoon-nlp/hindi-bert")
        model = TFAutoModelForSequenceClassification.from_pretrained("monsoon-nlp/hindi-bert", num_labels=2)
        
        # Tokenize data
        train_encodings = tokenizer(self.train_data['text'].tolist(), truncation=True, padding=True)
        test_encodings = tokenizer(self.test_data['text'].tolist(), truncation=True, padding=True)
        
        # Convert to TensorFlow dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            self.train_labels
        )).shuffle(1000).batch(16)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encodings),
            self.test_labels
        )).batch(16)
        
        # Compile and train
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
        
        history = model.fit(train_dataset, validation_data=test_dataset, epochs=3)
        
        # Save model and results
        model.save_pretrained('models/hinglish-bert')
        self.models['hinglish-bert'] = model
        self.histories['hinglish-bert'] = history.history
        
        # Evaluate
        results = model.evaluate(test_dataset)
        self.results['hinglish-bert'] = {
            'loss': results[0],
            'accuracy': results[1]
        }
        
    def train_roberta(self):
        print("Training RoBERTa model...")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = TFAutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
        
        # Similar training process as Hinglish-BERT
        train_encodings = tokenizer(self.train_data['text'].tolist(), truncation=True, padding=True)
        test_encodings = tokenizer(self.test_data['text'].tolist(), truncation=True, padding=True)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            self.train_labels
        )).shuffle(1000).batch(16)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encodings),
            self.test_labels
        )).batch(16)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
        
        history = model.fit(train_dataset, validation_data=test_dataset, epochs=3)
        
        model.save_pretrained('models/roberta')
        self.models['roberta'] = model
        self.histories['roberta'] = history.history
        
        results = model.evaluate(test_dataset)
        self.results['roberta'] = {
            'loss': results[0],
            'accuracy': results[1]
        }
        
    def train_xlm_roberta(self):
        print("Training XLM-RoBERTa model...")
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        model = TFAutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
        
        # Similar training process
        train_encodings = tokenizer(self.train_data['text'].tolist(), truncation=True, padding=True)
        test_encodings = tokenizer(self.test_data['text'].tolist(), truncation=True, padding=True)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            self.train_labels
        )).shuffle(1000).batch(16)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encodings),
            self.test_labels
        )).batch(16)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
        
        history = model.fit(train_dataset, validation_data=test_dataset, epochs=3)
        
        model.save_pretrained('models/xlm-roberta')
        self.models['xlm-roberta'] = model
        self.histories['xlm-roberta'] = history.history
        
        results = model.evaluate(test_dataset)
        self.results['xlm-roberta'] = {
            'loss': results[0],
            'accuracy': results[1]
        }
        
    def train_muril(self):
        print("Training MuRIL model...")
        tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
        model = TFAutoModelForSequenceClassification.from_pretrained("google/muril-base-cased", num_labels=2)
        
        # Similar training process
        train_encodings = tokenizer(self.train_data['text'].tolist(), truncation=True, padding=True)
        test_encodings = tokenizer(self.test_data['text'].tolist(), truncation=True, padding=True)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            self.train_labels
        )).shuffle(1000).batch(16)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encodings),
            self.test_labels
        )).batch(16)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
        
        history = model.fit(train_dataset, validation_data=test_dataset, epochs=3)
        
        model.save_pretrained('models/muril')
        self.models['muril'] = model
        self.histories['muril'] = history.history
        
        results = model.evaluate(test_dataset)
        self.results['muril'] = {
            'loss': results[0],
            'accuracy': results[1]
        }
        
    def create_visualizations(self):
        print("Creating visualizations...")
        
        # Accuracy comparison plot
        plt.figure(figsize=(10, 6))
        for model_name, history in self.histories.items():
            plt.plot(history['accuracy'], label=f'{model_name} (train)')
            plt.plot(history['val_accuracy'], label=f'{model_name} (val)')
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('visualizations/accuracy_comparison.png')
        plt.close()
        
        # Loss comparison plot
        plt.figure(figsize=(10, 6))
        for model_name, history in self.histories.items():
            plt.plot(history['loss'], label=f'{model_name} (train)')
            plt.plot(history['val_loss'], label=f'{model_name} (val)')
        plt.title('Model Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('visualizations/loss_comparison.png')
        plt.close()
        
        # Final accuracy comparison
        accuracies = {model: results['accuracy'] for model, results in self.results.items()}
        plt.figure(figsize=(8, 6))
        sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
        plt.title('Final Model Accuracy Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/final_accuracy_comparison.png')
        plt.close()
        
        # Save results to JSON
        with open('results.json', 'w') as f:
            json.dump(self.results, f, indent=4)
            
    def train_all_models(self):
        self.preprocess_data()
        self.train_hinglish_bert()
        self.train_roberta()
        self.train_xlm_roberta()
        self.train_muril()
        self.create_visualizations()
        print("All models trained and evaluated successfully!")

if __name__ == "__main__":
    trainer = ModelTrainer(
        train_path="C:/Users/LENOVO/Desktop/SarcastiX/Coding Files/sarcastiX/train.csv",
        test_path="C:/Users/LENOVO/Desktop/SarcastiX/Coding Files/sarcastiX/test.csv"
    )
    trainer.train_all_models()
