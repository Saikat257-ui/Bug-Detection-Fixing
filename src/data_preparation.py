"""
Data preparation module for collecting and processing code samples.
"""
import os
import json
import pandas as pd
import logging
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
import torch
from transformers import AutoTokenizer
import random
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import set_global_seed, is_valid_python_code

logger = logging.getLogger(__name__)

class CodeDataProcessor:
    """Class to handle code data collection and processing."""
    
    def __init__(self, tokenizer_name=None):
        """Initialize the data processor."""
        if tokenizer_name is None:
            tokenizer_name = config.MODEL_NAME
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        set_global_seed()
        logger.info(f"Initialized CodeDataProcessor with tokenizer: {tokenizer_name}")
    
    def load_code_dataset(self, dataset_name="codeparrot/github-code", language="python", split="train", max_samples=10000, use_streaming=False):
        """Load code dataset from Hugging Face with optimized filtering.
        
        Args:
            dataset_name: Name of the dataset to load from Hugging Face
            language: Programming language to filter for (e.g., 'python', 'javascript')
            split: Dataset split to use ('train', 'test', etc.)
            max_samples: Maximum number of samples to load
            use_streaming: Whether to use streaming mode (may not work with local filesystem cache)
            
        Returns:
            Dataset object filtered by language
        """
        logger.info(f"Loading {language} code dataset: {dataset_name}")
        
        # Define language-specific file extensions for filtering
        extensions = {
            'python': ['.py', '.pyw', '.pyx'],
            'javascript': ['.js', '.jsx', '.mjs']
        }
        
        try:
            # First try non-streaming approach as it's more compatible
            if not use_streaming:
                logger.info(f"Using regular dataset loading for {dataset_name}")
                try:
                    # Load dataset without streaming
                    dataset = load_dataset(dataset_name, split=split, streaming=False)
                    
                    # Filter by language
                    if language in dataset.features:
                        logger.info(f"Filtering dataset for {language} code")
                        dataset = dataset.filter(lambda x: x[language] is not None)
                    elif 'language' in dataset.features:
                        logger.info(f"Filtering by language field for {language} code")
                        dataset = dataset.filter(lambda x: x['language'].lower() == language.lower())
                    elif 'path' in dataset.features and language in extensions:
                        logger.info(f"Filtering by file extension for {language} code")
                        dataset = dataset.filter(lambda x: any(x['path'].lower().endswith(ext) for ext in extensions[language]))
                    
                    # Limit dataset size
                    if max_samples and max_samples < len(dataset):
                        logger.info(f"Limiting dataset to {max_samples} samples")
                        dataset = dataset.select(range(max_samples))
                    
                    logger.info(f"Loaded dataset with {len(dataset)} samples")
                    return dataset
                except Exception as e:
                    logger.warning(f"Regular loading failed: {e}. Trying streaming approach as fallback.")
                    use_streaming = True
            
            # Try streaming approach if requested or if regular approach failed
            if use_streaming and dataset_name == "codeparrot/github-code":
                logger.info(f"Using optimized streaming download for {dataset_name}")
                
                # Define a filter function based on language
                def language_filter(example):
                    # First try direct language field if available
                    if language in example and example[language] is not None:
                        return True
                    
                    # Then try language field if content is available
                    if 'language' in example and 'content' in example:
                        return example['language'].lower() == language.lower()
                    
                    # Finally try path extension
                    if 'path' in example and language in extensions:
                        return any(example['path'].lower().endswith(ext) for ext in extensions[language])
                    
                    return False
                
                try:
                    # Stream the dataset with our filter to avoid downloading everything
                    dataset = load_dataset(dataset_name, split=split, streaming=True)
                    filtered_dataset = dataset.filter(language_filter)
                    
                    # Convert streaming dataset to regular dataset with size limit
                    samples = []
                    sample_count = 0
                    
                    for sample in filtered_dataset:
                        samples.append(sample)
                        sample_count += 1
                        if max_samples and sample_count >= max_samples:
                            break
                    
                    if samples:
                        dataset = Dataset.from_dict({k: [sample.get(k) for sample in samples] 
                                                  for k in samples[0].keys()})
                        logger.info(f"Loaded optimized dataset with {len(dataset)} samples")
                        return dataset
                    else:
                        logger.warning(f"No samples found for {language} with optimized loading")
                except Exception as e:
                    logger.warning(f"Streaming approach failed: {e}. Falling back to regular loading.")
                    # Fall back to regular loading if streaming approach failed
            
            # Regular loading approach as fallback
            logger.info(f"Using regular dataset loading for {dataset_name}")
            dataset = load_dataset(dataset_name, split=split)
            
            # Check if dataset has the language-specific feature
            if language in dataset.features:
                logger.info(f"Filtering dataset for {language} code")
                dataset = dataset.filter(lambda x: x[language] is not None)
            else:
                # Try to find language in content field if available
                if 'content' in dataset.features and 'language' in dataset.features:
                    logger.info(f"Filtering by language field for {language} code")
                    dataset = dataset.filter(lambda x: x['language'].lower() == language.lower())
                else:
                    logger.warning(f"Could not find {language} specific data in dataset")
                    # If we can't find language-specific data, try to infer from file extensions
                    if 'path' in dataset.features:
                        logger.info(f"Attempting to filter by file extension for {language} code")
                        if language in extensions:
                            dataset = dataset.filter(lambda x: any(x['path'].lower().endswith(ext) for ext in extensions[language]))
            
            # Limit dataset size if needed
            if max_samples and max_samples < len(dataset):
                logger.info(f"Limiting dataset to {max_samples} samples")
                dataset = dataset.select(range(max_samples))
            
            logger.info(f"Loaded dataset with {len(dataset)} samples")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    
    def generate_synthetic_bugs(self, code):
        """Generate synthetic bugs in code.
        
        Types of bugs to introduce:
        1. Syntax errors (e.g., missing parentheses)
        2. Variable name errors (undefined or misspelled variables)
        3. Off-by-one errors
        4. Logic errors (e.g., wrong operators)
        5. Type errors
        """
        if not code or not isinstance(code, str):
            return code, "none", code
        
        # Make a copy of the original code
        original_code = code
        
        # Choose a bug type to introduce
        bug_types = [
            "syntax_error", 
            "undefined_variable", 
            "off_by_one_error",
            "logic_error",
            "type_error"
        ]
        
        bug_type = random.choice(bug_types)
        buggy_code = original_code
        
        try:
            if bug_type == "syntax_error":
                # Introduce syntax errors like missing parentheses/brackets
                for char, replacement in [('(', ''), (')', ''), ('{', ''), ('}', ''), (':', '')]:
                    if char in original_code and random.random() < 0.3:
                        pos = random.choice([i for i, c in enumerate(original_code) if c == char])
                        buggy_code = original_code[:pos] + original_code[pos+1:]
                        break
            
            elif bug_type == "undefined_variable":
                # Introduce variable name errors
                import ast
                try:
                    tree = ast.parse(original_code)
                    variables = []
                    
                    # Collect variable names
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                            variables.append(node.id)
                    
                    if variables:
                        var_to_change = random.choice(variables)
                        # Misspell the variable
                        misspelled = var_to_change[:-1] if len(var_to_change) > 1 else var_to_change + "x"
                        buggy_code = original_code.replace(var_to_change, misspelled, 1)
                except:
                    # Fallback if AST parsing fails
                    pass
            
            elif bug_type == "off_by_one_error":
                # Introduce off-by-one errors in range/indexing operations
                if "range(" in original_code:
                    buggy_code = original_code.replace("range(", "range(1 + ", 1)
                elif "] =" in original_code:
                    # Find array indexing and modify
                    parts = original_code.split("] =")
                    if len(parts) > 1 and "[" in parts[0]:
                        idx_start = parts[0].rindex("[")
                        idx_content = parts[0][idx_start+1:]
                        if idx_content.isdigit():
                            new_idx = int(idx_content) + 1
                            buggy_code = parts[0][:idx_start] + f"[{new_idx}] =" + "=".join(parts[1:])
            
            elif bug_type == "logic_error":
                # Change logical operators or comparison operators
                for op, replacement in [('==', '!='), ('!=', '=='), ('>', '<'), ('<', '>'), 
                                        ('and', 'or'), ('or', 'and')]:
                    if op in original_code:
                        buggy_code = original_code.replace(op, replacement, 1)
                        break
            
            elif bug_type == "type_error":
                # Introduce type errors (e.g., add string to int)
                if "+=" in original_code and not ("+ '" in original_code or "+ \"" in original_code):
                    buggy_code = original_code.replace("+=", "+= \"1\" +", 1)
                elif "return" in original_code and not "return \"" in original_code:
                    buggy_code = original_code.replace("return ", "return str(", 1) + ")"
        
        except Exception as e:
            logger.warning(f"Error generating synthetic bug: {e}")
            return original_code, "none", original_code
        
        # Check if we successfully introduced a bug
        if buggy_code == original_code:
            return original_code, "none", original_code
        
        return buggy_code, bug_type, original_code
    
    def prepare_dataset(self, raw_dataset, language="python"):
        """Prepare code dataset for bug detection training."""
        logger.info("Preparing dataset for bug detection training")
        
        prepared_data = []
        
        for idx, sample in enumerate(tqdm(raw_dataset, desc="Preparing data")):
            if language == "python":
                code = sample.get("content", sample.get("python", ""))
            elif language == "javascript":
                code = sample.get("content", sample.get("javascript", ""))
            else:
                code = sample.get("content", "")
            
            if not code or not isinstance(code, str) or len(code) < 50:
                continue
                
            # Generate a version with a bug
            for _ in range(3):  # Try up to 3 times to generate a bug
                buggy_code, bug_type, fixed_code = self.generate_synthetic_bugs(code)
                
                # Skip if no bug was introduced
                if bug_type == "none":
                    continue
                
                prepared_data.append({
                    "buggy_code": buggy_code,
                    "fixed_code": fixed_code,
                    "bug_type": bug_type,
                    "has_bug": 1  # Binary label
                })
                break
            
            # Also include the original code as a non-buggy example 50% of the time
            if random.random() < 0.5:
                is_valid, _ = is_valid_python_code(code)
                if is_valid:
                    prepared_data.append({
                        "buggy_code": code,  # Not actually buggy
                        "fixed_code": code,
                        "bug_type": "none",
                        "has_bug": 0  # Binary label for no bug
                    })
        
        logger.info(f"Created dataset with {len(prepared_data)} samples")
        return pd.DataFrame(prepared_data)
    
    def tokenize_data(self, dataset, max_length=None):
        """Tokenize the code data for the model."""
        if max_length is None:
            max_length = config.MAX_LENGTH
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["buggy_code"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["buggy_code", "fixed_code", "bug_type"]
        )
        
        return tokenized_datasets
    
    def split_and_save_data(self, df, test_size=None):
        """Split data into train and test sets and save to disk."""
        if test_size is None:
            test_size = config.TRAIN_TEST_SPLIT
        
        # Split the data
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=config.SEED
        )
        
        logger.info(f"Split dataset: Train={len(train_df)}, Test={len(test_df)}")
        
        # Save to disk
        os.makedirs(config.TRAIN_DATA_PATH, exist_ok=True)
        os.makedirs(config.TEST_DATA_PATH, exist_ok=True)
        
        train_path = os.path.join(config.TRAIN_DATA_PATH, "train_data.csv")
        test_path = os.path.join(config.TEST_DATA_PATH, "test_data.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Saved train data to {train_path}")
        logger.info(f"Saved test data to {test_path}")
        
        return train_df, test_df
    
    def load_or_create_dataset(self, languages=None, force_recreate=False, max_samples_per_language=None, use_streaming=True, dataset_name="codeparrot/github-code"):
        """Load dataset from disk or create a new one with optimized loading.
        
        Args:
            languages: List of programming languages to include in the dataset
            force_recreate: Whether to force recreation of the dataset even if it exists
            max_samples_per_language: Maximum number of samples to include per language
            use_streaming: Whether to use streaming download to optimize memory usage and download size
            dataset_name: Name of the dataset to load from Hugging Face
            
        Returns:
            Tuple of (train_df, test_df) containing the training and testing data
        """
        if languages is None:
            languages = config.PROGRAMMING_LANGUAGES
        
        if max_samples_per_language is None:
            max_samples_per_language = config.MAX_SAMPLES_PER_LANGUAGE
        
        train_path = os.path.join(config.TRAIN_DATA_PATH, "train_data.csv")
        test_path = os.path.join(config.TEST_DATA_PATH, "test_data.csv")
        
        # Check if datasets already exist
        if os.path.exists(train_path) and os.path.exists(test_path) and not force_recreate:
            logger.info("Loading existing datasets from disk")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Verify that the loaded datasets contain the requested languages
            if 'language' in train_df.columns:
                available_languages = train_df['language'].unique()
                missing_languages = [lang for lang in languages if lang not in available_languages]
                
                if missing_languages:
                    logger.warning(f"Existing dataset is missing languages: {missing_languages}")
                    if not force_recreate:
                        logger.info("Use force_recreate=True to regenerate the dataset with all languages")
            
            return train_df, test_df
        
        logger.info(f"Creating new datasets with max {max_samples_per_language} samples per language")
        all_data = []
        
        # Process each language
        for language in languages:
            logger.info(f"Processing {language} code with max {max_samples_per_language} samples")
            
            # Use optimized loading with streaming if enabled
            raw_dataset = self.load_code_dataset(
                dataset_name=dataset_name,
                language=language, 
                max_samples=max_samples_per_language,
                use_streaming=use_streaming
            )
            
            if raw_dataset and len(raw_dataset) > 0:
                logger.info(f"Preparing dataset for {language} with {len(raw_dataset)} samples")
                language_df = self.prepare_dataset(raw_dataset, language=language)
                
                if language_df is not None and len(language_df) > 0:
                    # Add language column if not already present
                    if 'language' not in language_df.columns:
                        language_df['language'] = language
                    
                    all_data.append(language_df)
                    logger.info(f"Added {len(language_df)} samples for {language}")
                else:
                    logger.warning(f"No valid samples prepared for {language}")
            else:
                logger.warning(f"Failed to load dataset for {language}")
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined dataset has {len(combined_df)} samples across {len(all_data)} languages")
            return self.split_and_save_data(combined_df)
        else:
            logger.error("Failed to create dataset: no valid data for any language")
            return None, None


def create_sample_data():
    """Create sample buggy code examples for testing."""
    samples = [
        {
            "buggy_code": """def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
    
# Example usage
data = [1, 2, 3, 4, 5]
avg = calculate_average(data)
print(f"The average is: {avg}")

# Bug: Division by zero if numbers is empty
""",
            "fixed_code": """def calculate_average(numbers):
    if not numbers:
        return 0
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
    
# Example usage
data = [1, 2, 3, 4, 5]
avg = calculate_average(data)
print(f"The average is: {avg}")
""",
            "bug_type": "logic_error",
            "has_bug": 1
        },
        {
            "buggy_code": """def find_max(arr):
    if not arr:
        return None
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
    return max_val

# No bug in this function
""",
            "fixed_code": """def find_max(arr):
    if not arr:
        return None
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
    return max_val

# No bug in this function
""",
            "bug_type": "none",
            "has_bug": 0
        },
        {
            "buggy_code": """def merge_dicts(dict1, dict2)
    result = dict1.copy()
    result.update(dict2)
    return result

# Bug: Missing colon after function definition
""",
            "fixed_code": """def merge_dicts(dict1, dict2):
    result = dict1.copy()
    result.update(dict2)
    return result
""",
            "bug_type": "syntax_error",
            "has_bug": 1
        },
        {
            "buggy_code": """def search_list(items, target):
    for i in range(0, len(items)-1):
        if items[i] == target:
            return i
    return -1

# Bug: Off-by-one error, should be range(0, len(items))
""",
            "fixed_code": """def search_list(items, target):
    for i in range(0, len(items)):
        if items[i] == target:
            return i
    return -1
""",
            "bug_type": "off_by_one_error",
            "has_bug": 1
        },
        {
            "buggy_code": """def calculate_discounted_price(price, discount_rate):
    discount = price * discount_rate
    final_price = price - discount
    return "The final price is " + final_price

# Bug: Type error - string concatenation with number
""",
            "fixed_code": """def calculate_discounted_price(price, discount_rate):
    discount = price * discount_rate
    final_price = price - discount
    return "The final price is " + str(final_price)
""",
            "bug_type": "type_error",
            "has_bug": 1
        }
    ]
    
    # Create sample data directory
    os.makedirs(config.SAMPLE_DATA_PATH, exist_ok=True)
    
    # Save samples as CSV
    sample_df = pd.DataFrame(samples)
    sample_path = os.path.join(config.SAMPLE_DATA_PATH, "sample_bugs.csv")
    sample_df.to_csv(sample_path, index=False)
    
    # Also save individual samples for testing
    for i, sample in enumerate(samples):
        sample_file = os.path.join(config.SAMPLE_DATA_PATH, f"sample_{i+1}.py")
        with open(sample_file, "w") as f:
            f.write(sample["buggy_code"])
    
    logger.info(f"Created {len(samples)} sample buggy code examples in {config.SAMPLE_DATA_PATH}")
    return sample_df


if __name__ == "__main__":
    processor = CodeDataProcessor()
    # Create sample data for testing
    create_sample_data()
    # Load or create full dataset
    train_data, test_data = processor.load_or_create_dataset(force_recreate=True)
    if train_data is not None:
        logger.info(f"Created dataset with {len(train_data)} training and {len(test_data)} testing samples")
