import pandas as pd
import numpy as np
import torch
import os
import warnings
import time
import logging
from datetime import datetime
from dataclasses import dataclass

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Optimizations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.set_num_threads(8)
warnings.filterwarnings('ignore')

@dataclass
class ClassificationResult:
    category: str
    confidence: float
    all_scores: dict
    processing_time: float
    model_info: str

class WorkingEmailClassifier:
    """🏆 Working Email Classifier for Transformers 4.55.2"""
    
    def __init__(self):
        self.categories = [
            "Action Required",
            "Informational/Updates", 
            "Social/Personal",
            "Scheduling/Events",
            "Support/Help",
            "Promotional/Marketing",
            "Transactional/Confirmations",
            "Urgent/Time-Sensitive"
        ]
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🍎 Device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.categories)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("🏆 Working Email Classifier Ready!")
        print(f"📋 Categories: {len(self.categories)}")
    
    def load_data(self, csv_path: str):
        """Load your dataset"""
        print(f"\n📊 LOADING DATA")
        print("="*40)
        
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded: {len(df):,} emails")
            
            # Clean data
            df['subject'] = df['subject'].fillna('').astype(str)
            df['body'] = df['body'].fillna('').astype(str)
            
            # Add default confidence if missing
            if 'confidence' not in df.columns:
                df['confidence'] = 0.8
            
            # Show categories
            print(f"\n📋 Distribution:")
            for cat, count in df['category'].value_counts().head().items():
                print(f"   {cat}: {count:,}")
            
            self.data = df
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def train(self, confidence_threshold=0.5):
        """Train the model"""
        print(f"\n🚀 TRAINING")
        print("="*30)
        
        try:
            # Get training data
            train_data = self.data[self.data['confidence'] > confidence_threshold]
            print(f"✅ Training samples: {len(train_data):,}")
            
            if len(train_data) < 1000:
                print("⚠️  Using lower threshold")
                train_data = self.data[self.data['confidence'] > 0.3]
                print(f"✅ Adjusted: {len(train_data):,}")
            
            # Prepare data
            texts = []
            labels = []
            
            for _, row in train_data.iterrows():
                text = f"{row['subject']} {row['body']}"[:300]  # Keep short
                texts.append(text)
                labels.append(row['category'])
            
            # Encode labels
            encoded_labels = self.label_encoder.transform(labels)
            
            # Split
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, encoded_labels, test_size=0.15, random_state=42, stratify=encoded_labels
            )
            
            print(f"📊 Train: {len(train_texts):,}, Val: {len(val_texts):,}")
            
            # Load model
            model_name = "distilbert-base-uncased"
            print(f"📥 Loading {model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=len(self.categories)
            ).to(self.device)
            
            # Tokenize
            train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=256)
            val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=256)
            
            # Datasets
            train_dataset = EmailDataset(train_encodings, train_labels)
            val_dataset = EmailDataset(val_encodings, val_labels)
            
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=2,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=16,
                warmup_steps=100,
                weight_decay=0.01,
                learning_rate=3e-5,
                logging_steps=50,
                eval_strategy="epoch",  
                save_strategy="epoch",
                dataloader_num_workers=0,
                report_to=None,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics,
            )
            
            print("🔥 Training...")
            start = time.time()
            trainer.train()
            
            # Evaluate
            results = trainer.evaluate()
            training_time = time.time() - start
            
            print(f"\n🎉 COMPLETE!")
            print(f"✅ Accuracy: {results['eval_accuracy']:.4f}")
            print(f"⏱️  Time: {training_time/60:.1f} min")
            
            # Save
            self.model.save_pretrained("./email_classifier")
            self.tokenizer.save_pretrained("./email_classifier")
            
            return results['eval_accuracy']
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            self.logger.error(f"Training error: {e}")
            return 0.0
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}
    
    def classify(self, subject: str, body: str):
        """Classify an email"""
        if self.model is None:
            return "No model trained"
        
        text = f"{subject} {body}"[:300]
        inputs = self.tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt").to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        
        pred_idx = np.argmax(probs)
        return self.categories[pred_idx], float(probs[pred_idx])
    
    def test(self):
        """Test the classifier"""
        print(f"\n🧪 TESTING")
        print("-" * 25)
        
        tests = [
            ("Meeting", "Can we meet tomorrow?"),
            ("Order", "Your order shipped"),
            ("Birthday", "Happy birthday!"),
            ("Alert", "System maintenance"),
            ("Sale", "50% off today!"),
            ("Help", "Need assistance"),
        ]
        
        for subject, body in tests:
            category, confidence = self.classify(subject, body)
            print(f"📧 '{subject}' → {category} ({confidence:.3f})")

# Simple Dataset
class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
    def __len__(self):
        return len(self.labels)

# Main
def main():
    print("🏆 EMAIL CLASSIFIER - TRANSFORMERS 4.55.2")
    print("="*50)
    
    try:
        # Initialize
        classifier = WorkingEmailClassifier()
        
        # Load data
        csv_path = '/Users/adeshnarayanatellakua/Documents/mini_project/universal_maximum_accuracy_labeled_emails.csv'
        
        if not classifier.load_data(csv_path):
            return
        
        # Train
        accuracy = classifier.train()
        
        if accuracy > 0:
            # Test
            classifier.test()
            print(f"\n🎯 SUCCESS! Accuracy: {accuracy:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()