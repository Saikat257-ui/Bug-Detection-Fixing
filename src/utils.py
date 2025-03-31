"""
Utility functions for the Code Bug Detection and Fix Recommendation System.
"""
import os
import logging
import random
import numpy as np
import torch
from transformers import set_seed
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_global_seed(seed=None):
    """Set seed for reproducibility across all libraries."""
    if seed is None:
        seed = config.SEED
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    logger.info(f"Global seed set to {seed}")

def get_device():
    """Get the device to use for training."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        config.DATA_DIR,
        config.TRAIN_DATA_PATH,
        config.TEST_DATA_PATH,
        config.SAMPLE_DATA_PATH,
        config.MODEL_DIR
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def save_model(model, tokenizer, path=None):
    """Save model and tokenizer to disk."""
    if path is None:
        path = config.MODEL_SAVE_PATH
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    logger.info(f"Model saved to {path}")

def load_model(model_class, tokenizer_class, path=None):
    """Load model and tokenizer from disk."""
    if path is None:
        path = config.MODEL_SAVE_PATH
    
    model = model_class.from_pretrained(path)
    tokenizer = tokenizer_class.from_pretrained(path)
    logger.info(f"Model loaded from {path}")
    return model, tokenizer

def format_metrics(metrics_dict):
    """Format metrics for display."""
    return {k: round(float(v), 4) if isinstance(v, (int, float)) else v 
            for k, v in metrics_dict.items()}

def tokenize_code(code, tokenizer, max_length=None):
    """Tokenize code snippets using the provided tokenizer."""
    if max_length is None:
        max_length = config.MAX_LENGTH
    
    return tokenizer(
        code,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

def extract_syntax_error_message(error):
    """Extract and format syntax error message."""
    if isinstance(error, SyntaxError):
        return f"Syntax error at line {error.lineno}: {error.msg}"
    return str(error)

def is_valid_python_code(code):
    """Check if code is valid Python syntax."""
    try:
        compile(code, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        return False, extract_syntax_error_message(e)
    except Exception as e:
        return False, str(e)
