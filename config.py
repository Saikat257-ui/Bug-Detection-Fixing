"""
Configuration settings for the Code Bug Detection and Fix Recommendation System.
"""
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "microsoft/codebert-base" 
MAX_LENGTH = 512  
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5
TRAIN_TEST_SPLIT = 0.2  
SEED = 42

# Data configuration
PROGRAMMING_LANGUAGES = ["python"]
MAX_SAMPLES_PER_LANGUAGE = 10000
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test")
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, "samples")


MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "bug_detector_model")


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

DETECTION_THRESHOLD = 0.45

METRICS = ["precision", "recall", "f1", "accuracy"]
