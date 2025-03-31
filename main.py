"""
Main application for Code Bug Detection and Fix Recommendation System.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import torch

import config
from src.utils import set_global_seed, get_device, create_directories, is_valid_python_code
from src.data_preparation import CodeDataProcessor, create_sample_data
from src.train import train_bug_detection_model, train_bug_classifier_model
from src.evaluate import ModelEvaluator
from src.gemini_api import GeminiFixRecommender

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BugDetectionApp:
    """Main application class for code bug detection and fix recommendation."""
    
    def __init__(self):
        """Initialize the application."""
        set_global_seed()
        create_directories()
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        self.detector_model = None
        self.detector_tokenizer = None
        self.classifier_model = None
        self.classifier_tokenizer = None
        self.evaluator = None
        self.fix_recommender = None
        
        self._load_models()
    
    def _load_models(self):
        """Load the trained models if available."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        detector_path = os.path.join(config.MODEL_SAVE_PATH, "detector")
        if os.path.exists(detector_path):
            try:
                self.detector_model = AutoModelForSequenceClassification.from_pretrained(detector_path)
                self.detector_tokenizer = AutoTokenizer.from_pretrained(detector_path)
                logger.info(f"Loaded bug detector model from {detector_path}")
            except Exception as e:
                logger.error(f"Error loading detector model: {e}")
                self._initialize_default_models()
        else:
            logger.warning("Bug detector model not found. Using default model.")
            self._initialize_default_models()
        
        classifier_path = os.path.join(config.MODEL_SAVE_PATH, "classifier")
        if os.path.exists(classifier_path):
            try:
                self.classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_path)
                self.classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
                logger.info(f"Loaded bug classifier model from {classifier_path}")
            except Exception as e:
                logger.error(f"Error loading classifier model: {e}")
                self.classifier_model = None
                self.classifier_tokenizer = None
        
        if self.detector_model is not None:
            self.evaluator = ModelEvaluator(
                self.detector_model,
                self.detector_tokenizer,
                self.classifier_model,
                self.classifier_tokenizer
            )
        
        self.fix_recommender = GeminiFixRecommender()
    
    def _initialize_default_models(self):
        """Initialize with default pre-trained models."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        model_name = config.MODEL_NAME
        
        try:
            self.detector_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=2
            )
            self.detector_tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Initialized default model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading default model: {e}")
            sys.exit(1)
    
    def prepare_data(self, force_recreate=False, languages=None, max_samples_per_language=10000, use_streaming=False):
        """Prepare training and testing data.
        
        Args:
            force_recreate: Whether to force recreation of the dataset even if it exists
            languages: List of programming languages to include (defaults to config.PROGRAMMING_LANGUAGES)
            max_samples_per_language: Maximum number of samples to include per language
            use_streaming: Whether to use streaming download to optimize memory usage and download size
            
        Returns:
            Tuple of (train_data, test_data) containing the training and testing data
        """
        processor = CodeDataProcessor(tokenizer_name=config.MODEL_NAME)
        
        create_sample_data()
        
        if languages is None:
            languages = config.PROGRAMMING_LANGUAGES
        
        logger.info(f"Preparing data for languages: {languages} with streaming={use_streaming}")
        
        train_data, test_data = processor.load_or_create_dataset(
            languages=languages,
            force_recreate=force_recreate,
            max_samples_per_language=max_samples_per_language,
            use_streaming=use_streaming
        )
        
        return train_data, test_data
    
    def train_models(self, force_retrain=False):
        """Train the bug detection and classification models."""
        train_data, test_data = self.prepare_data()
        
        if train_data is None or test_data is None:
            logger.error("Failed to prepare data for training")
            return False
        
        detector_path = train_bug_detection_model(train_data, test_data, force_retrain)
        
        if detector_path is None:
            logger.error("Failed to train bug detector model")
            return False
        
        classifier_path = train_bug_classifier_model(train_data, test_data, force_retrain)
        
        if classifier_path is None:
            logger.warning("Failed to train bug classifier model")
        
        self._load_models()
        
        return True
    
    def analyze_file(self, file_path):
        """Analyze a file for bugs and generate fix recommendations."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {
                "error": f"File not found: {file_path}",
                "has_bug": False
            }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {
                "error": f"Error reading file: {str(e)}",
                "has_bug": False
            }
        
        return self.analyze_code(code)
    
    def analyze_code(self, code):
        """Analyze code for bugs and generate fix recommendations."""
        if not code or not isinstance(code, str):
            return {
                "error": "No code provided or invalid code format",
                "has_bug": False
            }
        
        if self.evaluator is None:
            self._load_models()
            if self.evaluator is None:
                return {
                    "error": "Models not loaded properly",
                    "has_bug": False
                }
        
        analysis_result = self.evaluator.analyze_code(code)
        
        if (analysis_result["has_bug"] or analysis_result.get("borderline_case", False)) \
            and self.fix_recommender is not None:
            bug_type = analysis_result.get("bug_type", None)
            
            fix_result = self.fix_recommender.generate_fix(code, bug_type)
            
            analysis_result["fix_recommendation"] = fix_result
        else:
            analysis_result["fix_recommendation"] = {
                "success": False,
                "explanation": "No potential bugs detected",
            }
        
        return analysis_result
    
    def run_cli(self):
        """Run the command-line interface."""
        parser = argparse.ArgumentParser(
            description="Code Bug Detection and Fix Recommendation System"
        )
        
        parser.add_argument(
            "--train", 
            action="store_true",
            help="Train the bug detection models"
        )
        
        parser.add_argument(
            "--force-retrain", 
            action="store_true",
            help="Force retraining of models even if they already exist"
        )
        
        parser.add_argument(
            "--analyze-file",
            type=str,
            help="Path to a file to analyze for bugs"
        )
        
        parser.add_argument(
            "--analyze-code",
            type=str,
            help="Code snippet to analyze for bugs"
        )
        
        args = parser.parse_args()
        
        if args.train:
            logger.info("Training models...")
            self.train_models(force_retrain=args.force_retrain)
        
        if args.analyze_file:
            logger.info(f"Analyzing file: {args.analyze_file}")
            result = self.analyze_file(args.analyze_file)
            self._display_analysis_result(result, args.analyze_file)
        
        if args.analyze_code:
            logger.info("Analyzing code snippet")
            result = self.analyze_code(args.analyze_code)
            self._display_analysis_result(result, "code_snippet")
        
        if not args.analyze_file and not args.analyze_code and not args.train:
            sample_dir = config.SAMPLE_DATA_PATH
            if os.path.exists(sample_dir):
                sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.py')]
                
                if sample_files:
                    logger.info(f"Found {len(sample_files)} sample files. Analyzing...")
                    
                    for sample_file in sample_files:
                        file_path = os.path.join(sample_dir, sample_file)
                        logger.info(f"Analyzing sample file: {sample_file}")
                        result = self.analyze_file(file_path)
                        self._display_analysis_result(result, file_path)
                else:
                    self._show_help()
            else:
                self._show_help()
    
    def _display_analysis_result(self, result, file_path):
        """Display the analysis result in a user-friendly format."""
        print("\n" + "="*50)
        print(f"ANALYSIS RESULTS FOR: {os.path.basename(file_path)}")
        print("="*50)
        
        if "error" in result and result["error"]:
            print(f"ERROR: {result['error']}")
            return
        
        if result["has_bug"]:
            print(f"BUG DETECTED (Confidence: {result['bug_probability']:.2f})")
            
            if "bug_type" in result:
                print(f"Bug Type: {result['bug_type']}")
            
            if "fix_recommendation" in result and result["fix_recommendation"]["success"]:
                fix = result["fix_recommendation"]
                
                print("\n--- RECOMMENDED FIX ---")
                print(fix["fixed_code"])
                
                print("\n--- EXPLANATION ---")
                print(fix["explanation"])
        elif result.get("borderline_case", False):
            print(f"BORDERLINE CASE DETECTED (Confidence: {result['bug_probability']:.2f})")
            
            if "fix_recommendation" in result and result["fix_recommendation"]["success"]:
                fix = result["fix_recommendation"]
                
                print("\n--- POTENTIAL FIX (REVIEW RECOMMENDED) ---")
                print(fix["fixed_code"])
                
                print("\n--- EXPLANATION ---")
                print(fix["explanation"])
        else:
            print("NO POTENTIAL BUGS DETECTED")
            print(f"Confidence: {1 - result['bug_probability']:.2f}")
        
        print("\n--- METRICS ---")
        for key, value in result.get("metrics", {}).items():
            print(f"{key}: {value}")
        
        print("="*50 + "\n")
    
    def _show_help(self):
        """Show help information."""
        print("\nCode Bug Detection and Fix Recommendation System")
        print("="*50)
        print("Usage examples:")
        print("  Train models:   python main.py --train")
        print("  Analyze file:   python main.py --analyze-file /path/to/file.py")
        print("  Analyze code:   python main.py --analyze-code \"def foo(): return 1/0\"")
        print("\nFor more options, use: python main.py --help")


if __name__ == "__main__":
    print("Starting BugDetectionApp...")
    try:
        app = BugDetectionApp()
        print("Running CLI...")
        app.run_cli()
        print("CLI execution completed")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
