"""
Evaluation module for code bug detection and classification models.
"""
import os
import json
import logging
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import get_device, format_metrics, tokenize_code
from src.model import load_pretrained_detector

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Class to evaluate code bug detection and classification models."""
    
    def __init__(self, detector_model, detector_tokenizer, classifier_model=None, classifier_tokenizer=None):
        """Initialize the model evaluator."""
        self.detector_model = detector_model
        self.detector_tokenizer = detector_tokenizer
        self.classifier_model = classifier_model
        self.classifier_tokenizer = classifier_tokenizer
        
        self.device = get_device()
        self.detector_model.to(self.device)
        
        if self.classifier_model is not None:
            self.classifier_model.to(self.device)
        
        # Load bug type label encoder if available
        self.label_encoder = None
        label_encoder_path = os.path.join(config.MODEL_SAVE_PATH, "classifier", "label_encoder.pkl")
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
        
        logger.info(f"Initialized ModelEvaluator on device: {self.device}")
    
    def evaluate_detector(self, test_dataloader):
        """Evaluate the bug detector model."""
        self.detector_model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating bug detector"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch
                
                outputs = self.detector_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm.tolist()
        }
        
        logger.info(f"Bug detector evaluation results: {format_metrics(metrics)}")
        
        return metrics
    
    def evaluate_classifier(self, test_dataloader):
        """Evaluate the bug classifier model."""
        if self.classifier_model is None:
            logger.error("Bug classifier model not provided")
            return None
        
        self.classifier_model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating bug classifier"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch
                
                outputs = self.classifier_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Get class names if label encoder is available
        if self.label_encoder is not None:
            class_names = self.label_encoder.classes_
        else:
            class_names = [f"Class_{i}" for i in range(len(set(all_labels)))]
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Detailed classification report
        report = classification_report(
            all_labels, all_preds, 
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
        
        logger.info(f"Bug classifier evaluation results: {format_metrics(metrics)}")
        
        return metrics
    
    def analyze_code(self, code_snippet):
        """
        Analyze a code snippet for bugs.
        
        Args:
            code_snippet (str): The code snippet to analyze
            
        Returns:
            dict: Results of the analysis, including:
                - has_bug: Boolean indicating if a bug was detected
                - bug_type: Predicted bug type if a bug was detected (if classifier is available)
                - bug_probability: Probability that the code contains a bug
                - metrics: Performance metrics for the model
        """
        # Ensure code is a string
        if not isinstance(code_snippet, str):
            code_snippet = str(code_snippet)
        
        # Tokenize the code
        inputs = tokenize_code(code_snippet, self.detector_tokenizer)
        
        # Move inputs to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Set model to evaluation mode
        self.detector_model.eval()
        
        # Make prediction
        with torch.no_grad():
            outputs = self.detector_model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Extract logits from outputs
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            
            # Get prediction and probability
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()
            bug_prob = probs[0, 1].item()  # Probability of class 1 (has bug)
        
        # Determine if code has a bug
        has_bug = bug_prob >= config.DETECTION_THRESHOLD
        
        # Get borderline cases for recommendations
        borderline = (bug_prob >= config.DETECTION_THRESHOLD - 0.15) & (bug_prob < config.DETECTION_THRESHOLD)
        
        # Initialize result
        result = {
            "has_bug": has_bug or borderline,
            "borderline_case": borderline,
            "bug_probability": bug_prob,
            "metrics": {
                "confidence": bug_prob,
                "threshold": 0.5,
            }
        }
        
        # If a bug was detected and we have a classifier, predict the bug type
        if has_bug and self.classifier_model is not None:
            # Set classifier to evaluation mode
            self.classifier_model.eval()
            
            # Tokenize the code (use the classifier tokenizer if different)
            if self.classifier_tokenizer != self.detector_tokenizer:
                inputs = tokenize_code(code_snippet, self.classifier_tokenizer)
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.classifier_model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Extract logits from outputs
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                
                # Get prediction and probabilities
                bug_type_probs = torch.nn.functional.softmax(logits, dim=1)
                bug_type_idx = torch.argmax(logits, dim=1).item()
            
            # Get the bug type name
            if self.label_encoder is not None:
                bug_type = self.label_encoder.classes_[bug_type_idx]
            else:
                bug_type = f"Type_{bug_type_idx}"
            
            # Add to result
            result["bug_type"] = bug_type
            result["bug_type_probability"] = bug_type_probs[0, bug_type_idx].item()
        
        return result
    
    def plot_confusion_matrix(self, cm, classes=None, normalize=False, title='Confusion Matrix', 
                             cmap=plt.cm.Blues, save_path=None):
        """Plot confusion matrix."""
        if classes is None:
            if len(cm) == 2:
                classes = ['No Bug', 'Has Bug']
            else:
                classes = [f'Type {i}' for i in range(len(cm))]
        
        plt.figure(figsize=(8, 6))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized ' + title
        else:
            fmt = 'd'
        
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def save_metrics(self, detector_metrics, classifier_metrics=None, output_path=None):
        """Save evaluation metrics to a JSON file."""
        if output_path is None:
            output_path = os.path.join(config.MODEL_DIR, "evaluation_metrics.json")
        
        metrics = {
            "detector": format_metrics(detector_metrics)
        }
        
        if classifier_metrics is not None:
            metrics["classifier"] = format_metrics(classifier_metrics)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved evaluation metrics to {output_path}")


def evaluate_models(test_data_path=None):
    """Evaluate bug detection and classification models."""
    # Load test data
    if test_data_path is None:
        test_data_path = os.path.join(config.TEST_DATA_PATH, "test_data.csv")
    
    if not os.path.exists(test_data_path):
        logger.error(f"Test data not found at {test_data_path}")
        return None
    
    test_df = pd.read_csv(test_data_path)
    logger.info(f"Loaded test data with {len(test_df)} samples")
    
    # Load detector model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    detector_path = os.path.join(config.MODEL_SAVE_PATH, "detector")
    if os.path.exists(detector_path):
        detector_model = AutoModelForSequenceClassification.from_pretrained(detector_path)
        detector_tokenizer = AutoTokenizer.from_pretrained(detector_path)
    else:
        logger.warning(f"Detector model not found at {detector_path}, using default model")
        detector_model, detector_tokenizer = load_pretrained_detector()
    
    # Load classifier model
    classifier_path = os.path.join(config.MODEL_SAVE_PATH, "classifier")
    classifier_model = None
    classifier_tokenizer = None
    
    if os.path.exists(classifier_path):
        classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_path)
        classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
    else:
        logger.warning(f"Classifier model not found at {classifier_path}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        detector_model, 
        detector_tokenizer,
        classifier_model,
        classifier_tokenizer
    )
    
    # Prepare data for detector evaluation
    from torch.utils.data import DataLoader, TensorDataset
    
    detector_encodings = detector_tokenizer(
        test_df['buggy_code'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=config.MAX_LENGTH,
        return_tensors='pt'
    )
    
    detector_labels = torch.tensor(test_df['has_bug'].tolist())
    
    detector_dataset = TensorDataset(
        detector_encodings['input_ids'],
        detector_encodings['attention_mask'],
        detector_labels
    )
    
    detector_dataloader = DataLoader(
        detector_dataset,
        batch_size=config.BATCH_SIZE
    )
    
    # Evaluate detector
    detector_metrics = evaluator.evaluate_detector(detector_dataloader)
    
    # Prepare data for classifier evaluation (if available)
    classifier_metrics = None
    if classifier_model is not None:
        # Only use rows with bugs for bug type classification
        bug_test_df = test_df[test_df['has_bug'] == 1].copy()
        
        if len(bug_test_df) > 0:
            # Load label encoder
            label_encoder_path = os.path.join(classifier_path, "label_encoder.pkl")
            label_encoder = None
            
            if os.path.exists(label_encoder_path):
                with open(label_encoder_path, "rb") as f:
                    label_encoder = pickle.load(f)
                
                # Encode bug types
                bug_test_df['bug_type_encoded'] = label_encoder.transform(bug_test_df['bug_type'])
                
                # Prepare data
                classifier_encodings = classifier_tokenizer(
                    bug_test_df['buggy_code'].tolist(),
                    truncation=True,
                    padding='max_length',
                    max_length=config.MAX_LENGTH,
                    return_tensors='pt'
                )
                
                classifier_labels = torch.tensor(bug_test_df['bug_type_encoded'].tolist())
                
                classifier_dataset = TensorDataset(
                    classifier_encodings['input_ids'],
                    classifier_encodings['attention_mask'],
                    classifier_labels
                )
                
                classifier_dataloader = DataLoader(
                    classifier_dataset,
                    batch_size=config.BATCH_SIZE
                )
                
                # Evaluate classifier
                classifier_metrics = evaluator.evaluate_classifier(classifier_dataloader)
    
    # Save metrics
    evaluator.save_metrics(detector_metrics, classifier_metrics)
    
    return detector_metrics, classifier_metrics


if __name__ == "__main__":
    detector_metrics, classifier_metrics = evaluate_models()
    
    # Plot confusion matrices
    if detector_metrics and "confusion_matrix" in detector_metrics:
        cm = np.array(detector_metrics["confusion_matrix"])
        evaluator = ModelEvaluator(None, None)  # Dummy evaluator just for plotting
        evaluator.plot_confusion_matrix(
            cm, 
            classes=['No Bug', 'Has Bug'],
            title='Bug Detector Confusion Matrix',
            save_path=os.path.join(config.MODEL_DIR, "detector_cm.png")
        )
