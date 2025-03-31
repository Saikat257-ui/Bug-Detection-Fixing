"""
Training module for the code bug detection model.
"""
import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm, trange
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import set_global_seed, get_device, save_model
from src.model import CodeBugDetector, CodeBugClassifier

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class to handle model training."""
    
    def __init__(self, model, tokenizer, device=None):
        """Initialize the model trainer."""
        self.model = model
        self.tokenizer = tokenizer
        
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        
        self.model.to(self.device)
        
        logger.info(f"Initialized ModelTrainer on device: {self.device}")
    
    def prepare_data_loaders(self, train_df, test_df, batch_size=None):
        """Prepare data loaders for training and evaluation."""
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        
        # Tokenize the data
        train_encodings = self.tokenizer(
            train_df['buggy_code'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=config.MAX_LENGTH,
            return_tensors='pt'
        )
        
        test_encodings = self.tokenizer(
            test_df['buggy_code'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=config.MAX_LENGTH,
            return_tensors='pt'
        )
        
        # Prepare labels
        train_labels = torch.tensor(train_df['has_bug'].tolist())
        test_labels = torch.tensor(test_df['has_bug'].tolist())
        
        # Create tensor datasets
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            train_labels
        )
        
        test_dataset = TensorDataset(
            test_encodings['input_ids'],
            test_encodings['attention_mask'],
            test_labels
        )
        
        # Create data loaders
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=batch_size
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size
        )
        
        logger.info(f"Prepared data loaders: {len(train_dataloader)} training batches, {len(test_dataloader)} testing batches")
        
        return train_dataloader, test_dataloader
    
    def prepare_bug_type_data_loaders(self, train_df, test_df, batch_size=None):
        """Prepare data loaders for bug type classification."""
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        
        # Only use rows with bugs for bug type classification
        train_bug_df = train_df[train_df['has_bug'] == 1].copy()
        test_bug_df = test_df[test_df['has_bug'] == 1].copy()
        
        # Encode bug types
        label_encoder = LabelEncoder()
        
        # Fit on all bug types defined in config
        label_encoder.fit(config.BUG_TYPES)
        
        # Transform the bug types in the data
        train_bug_df['bug_type_encoded'] = label_encoder.transform(train_bug_df['bug_type'])
        test_bug_df['bug_type_encoded'] = label_encoder.transform(test_bug_df['bug_type'])
        
        # Tokenize the data
        train_encodings = self.tokenizer(
            train_bug_df['buggy_code'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=config.MAX_LENGTH,
            return_tensors='pt'
        )
        
        test_encodings = self.tokenizer(
            test_bug_df['buggy_code'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=config.MAX_LENGTH,
            return_tensors='pt'
        )
        
        # Prepare labels
        train_labels = torch.tensor(train_bug_df['bug_type_encoded'].tolist())
        test_labels = torch.tensor(test_bug_df['bug_type_encoded'].tolist())
        
        # Create tensor datasets
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            train_labels
        )
        
        test_dataset = TensorDataset(
            test_encodings['input_ids'],
            test_encodings['attention_mask'],
            test_labels
        )
        
        # Create data loaders
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=batch_size
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size
        )
        
        logger.info(f"Prepared bug type data loaders: {len(train_dataloader)} training batches, {len(test_dataloader)} testing batches")
        
        return train_dataloader, test_dataloader, label_encoder
    
    def train(self, train_dataloader, epochs=None, learning_rate=None, warmup_steps=0, evaluation_dataloader=None):
        """Train the model."""
        if epochs is None:
            epochs = config.EPOCHS
        
        if learning_rate is None:
            learning_rate = config.LEARNING_RATE
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        logger.info(f"Starting training for {epochs} epochs")
        
        global_step = 0
        training_loss = 0.0
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for step, batch in enumerate(epoch_iterator):
                # Get inputs
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                scheduler.step()
                
                # Track loss
                training_loss += loss.item()
                global_step += 1
                
                # Update progress bar
                epoch_iterator.set_postfix({"loss": loss.item()})
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                
                # Evaluate if requested
                if evaluation_dataloader is not None and global_step % 100 == 0:
                    self.evaluate(evaluation_dataloader)
                    self.model.train()
            
            # Log average loss for the epoch
            avg_loss = training_loss / global_step
            logger.info(f"Epoch {epoch+1}/{epochs} - Average loss: {avg_loss:.4f}")
            
            # Evaluate after each epoch if dataloader is provided
            if evaluation_dataloader is not None:
                self.evaluate(evaluation_dataloader)
                self.model.train()
        
        logger.info(f"Training complete after {epochs} epochs, {global_step} steps")
        
        # Save the trained model
        save_model(self.model, self.tokenizer, detector_path)
        logger.info(f"Trained model saved to {detector_path}")
        
        return global_step, training_loss / global_step
    
    def evaluate(self, eval_dataloader):
        """Evaluate the model on the given dataloader."""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        logger.info("Starting evaluation")
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Get inputs
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs["loss"]
                logits = outputs["logits"]
                
                # Track loss
                total_loss += loss.item()
                
                # Convert logits to predictions
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Calculate metrics
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Log results
        metrics = {
            "loss": total_loss / len(eval_dataloader),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        logger.info(f"Evaluation results: {metrics}")
        
        return metrics
    
    def save(self, output_dir=None):
        """Save the model and tokenizer."""
        if output_dir is None:
            output_dir = config.MODEL_SAVE_PATH
        
        save_model(self.model, self.tokenizer, output_dir)


def train_bug_detection_model(train_df=None, test_df=None, force_retrain=False):
    """Train the bug detection model."""
    set_global_seed()
    
    model_save_path = os.path.join(config.MODEL_SAVE_PATH, "detector")
    
    # Check if model already exists
    if os.path.exists(model_save_path) and not force_retrain:
        logger.info(f"Bug detection model already exists at {model_save_path}. Skipping training.")
        return model_save_path
    
    # Load data if not provided
    if train_df is None or test_df is None:
        train_path = os.path.join(config.TRAIN_DATA_PATH, "train_data.csv")
        test_path = os.path.join(config.TEST_DATA_PATH, "test_data.csv")
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            logger.info("Loading training data from disk")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        else:
            logger.error("Training data not found. Please run data_preparation.py first.")
            return None
    
    # Initialize model and tokenizer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, 
        num_labels=2
    )
    
    # Initialize trainer
    trainer = ModelTrainer(model, tokenizer)
    
    # Prepare data loaders
    train_dataloader, test_dataloader = trainer.prepare_data_loaders(train_df, test_df)
    
    # Train model
    trainer.train(
        train_dataloader, 
        epochs=config.EPOCHS,
        evaluation_dataloader=test_dataloader
    )
    
    # Final evaluation
    metrics = trainer.evaluate(test_dataloader)
    
    # Save model
    os.makedirs(model_save_path, exist_ok=True)
    trainer.save(model_save_path)
    
    logger.info(f"Bug detection model trained and saved to {model_save_path}")
    
    return model_save_path


def train_bug_classifier_model(train_df=None, test_df=None, force_retrain=False):
    """Train the bug classifier model."""
    set_global_seed()
    
    model_save_path = os.path.join(config.MODEL_SAVE_PATH, "classifier")
    
    # Check if model already exists
    if os.path.exists(model_save_path) and not force_retrain:
        logger.info(f"Bug classifier model already exists at {model_save_path}. Skipping training.")
        return model_save_path
    
    # Load data if not provided
    if train_df is None or test_df is None:
        train_path = os.path.join(config.TRAIN_DATA_PATH, "train_data.csv")
        test_path = os.path.join(config.TEST_DATA_PATH, "test_data.csv")
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            logger.info("Loading training data from disk")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        else:
            logger.error("Training data not found. Please run data_preparation.py first.")
            return None
    
    # Initialize model and tokenizer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, 
        num_labels=len(config.BUG_TYPES)
    )
    
    # Initialize trainer
    trainer = ModelTrainer(model, tokenizer)
    
    # Prepare data loaders for bug type classification
    train_dataloader, test_dataloader, label_encoder = trainer.prepare_bug_type_data_loaders(train_df, test_df)
    
    # Train model
    trainer.train(
        train_dataloader, 
        epochs=config.EPOCHS,
        evaluation_dataloader=test_dataloader
    )
    
    # Final evaluation
    metrics = trainer.evaluate(test_dataloader)
    
    # Save model
    os.makedirs(model_save_path, exist_ok=True)
    trainer.save(model_save_path)
    
    # Save label encoder
    import pickle
    with open(os.path.join(model_save_path, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    
    logger.info(f"Bug classifier model trained and saved to {model_save_path}")
    
    return model_save_path


if __name__ == "__main__":
    # Load data
    train_path = os.path.join(config.TRAIN_DATA_PATH, "train_data.csv")
    test_path = os.path.join(config.TEST_DATA_PATH, "test_data.csv")
    
    # Check if data exists, otherwise use sample data
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.info("Using sample data for training")
        from data_preparation import create_sample_data
        sample_df = create_sample_data()
        
        # Split sample data
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(sample_df, test_size=0.2, random_state=config.SEED)
    else:
        logger.info("Loading data from disk")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    
    # Train models
    detector_path = train_bug_detection_model(train_df, test_df, force_retrain=True)
    classifier_path = train_bug_classifier_model(train_df, test_df, force_retrain=True)
