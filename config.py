"""
Configuration settings for the Code Bug Detection and Fix Recommendation System.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model configuration
MODEL_NAME = "microsoft/codebert-base"  # Pre-trained model to use
MAX_LENGTH = 512  # Maximum token length for inputs
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5
TRAIN_TEST_SPLIT = 0.2  # 20% for testing
SEED = 42

# Data configuration
PROGRAMMING_LANGUAGES = ["python", "javascript"]  # Languages to support initially
MAX_SAMPLES_PER_LANGUAGE = 10000  # Maximum number of samples to load per language
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test")
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, "samples")

# Model saving/loading
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "bug_detector_model")

# Bug classification
BUG_TYPES = [
    "syntax_error",
    "runtime_error",
    "logic_error",
    "off_by_one_error",
    "infinite_loop",
    "memory_leak",
    "null_pointer",
    "undefined_variable",
    "type_error",
    "index_error",
    "attribute_error",
    "import_error",
    "value_error",
    "key_error",
    "division_by_zero",
    "file_not_found",
    "permission_error",
    "recursion_error",
    "other"
]

# Evaluation threshold
DETECTION_THRESHOLD = 0.45

# Evaluation metrics
METRICS = ["precision", "recall", "f1", "accuracy"]
