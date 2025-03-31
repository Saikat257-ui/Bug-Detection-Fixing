"""
Model architecture for code bug detection.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import get_device

logger = logging.getLogger(__name__)

class CodeBugDetector(nn.Module):
    """Model for detecting bugs in code."""
    
    def __init__(self, pretrained_model_name=None, num_labels=2):
        """Initialize the bug detector model."""
        super(CodeBugDetector, self).__init__()
        
        if pretrained_model_name is None:
            pretrained_model_name = config.MODEL_NAME
        
        self.num_labels = num_labels
        self.device = get_device()
        
        logger.info(f"Initializing CodeBugDetector with {pretrained_model_name}")
        
        # Load pre-trained model
        self.code_model = AutoModel.from_pretrained(pretrained_model_name)
        
        # Get the hidden size from the model config
        hidden_size = self.code_model.config.hidden_size
        
        # Classification layers
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Move model to GPU if available
        self.to(self.device)
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.parameters())} parameters")
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """Forward pass through the model."""
        # Pass input through the pre-trained model
        outputs = self.code_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get the [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits
        }


class CodeBugClassifier(nn.Module):
    """Model for classifying bug types in code."""
    
    def __init__(self, pretrained_model_name=None, num_classes=None):
        """Initialize the bug classifier model."""
        super(CodeBugClassifier, self).__init__()
        
        if pretrained_model_name is None:
            pretrained_model_name = config.MODEL_NAME
        
        if num_classes is None:
            num_classes = len(config.BUG_TYPES)
        
        self.device = get_device()
        
        logger.info(f"Initializing CodeBugClassifier with {pretrained_model_name}")
        
        # Load pre-trained model
        self.code_model = AutoModel.from_pretrained(pretrained_model_name)
        
        # Get the hidden size from the model config
        hidden_size = self.code_model.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Move model to GPU if available
        self.to(self.device)
        
        logger.info(f"Bug classifier initialized with {sum(p.numel() for p in self.parameters())} parameters")
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """Forward pass through the model."""
        # Pass input through the pre-trained model
        outputs = self.code_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, len(config.BUG_TYPES)), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits
        }


def load_pretrained_detector(model_path=None):
    """Load a pre-trained bug detector model."""
    if model_path is None:
        model_path = config.MODEL_SAVE_PATH
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info(f"Loaded pre-trained bug detector from {model_path}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading pre-trained model: {e}")
        logger.info(f"Falling back to default model: {config.MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME, 
            num_labels=2
        )
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        return model, tokenizer


if __name__ == "__main__":
    # Simple test to initialize models
    detector = CodeBugDetector()
    classifier = CodeBugClassifier()
    print(f"Detector device: {detector.device}")
    print(f"Classifier device: {classifier.device}")
