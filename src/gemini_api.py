"""
Gemini API integration for generating code bug fix recommendations.
"""
import os
import logging
import sys
import time
import google.generativeai as genai
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import is_valid_python_code

logger = logging.getLogger(__name__)

class GeminiFixRecommender:
    """Class to handle Gemini API integration for code fix recommendations."""
    
    def __init__(self, api_key=None):
        """Initialize the Gemini API client."""
        if api_key is None:
            api_key = config.GEMINI_API_KEY
            
        if not api_key:
            error_msg = "Missing Gemini API key. Please set the GEMINI_API_KEY in your .env file."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        genai.configure(
            api_key=api_key,
            transport="rest"
        )
        
        
        model_options = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro-latest"]
        selected_model = None
        
        try:
            models = genai.list_models()
            available_model_names = [model.name for model in models]
            logger.info(f"Available Gemini models: {available_model_names}")
            
            for model_name in model_options:
                if any(model_name in m for m in available_model_names):
                    selected_model = model_name
                    logger.info(f"Selected Gemini model: {selected_model}")
                    break
                    
            if selected_model is None and available_model_names:
                for model_name in available_model_names:
                    if "gemini" in model_name.lower():
                        selected_model = model_name
                        logger.info(f"Using available Gemini model: {selected_model}")
                        break
        
        except Exception as e:
            logger.warning(f"Error listing Gemini models: {str(e)}")
        
        if selected_model is None:
            selected_model = model_options[0]
            logger.warning(f"No Gemini models found, falling back to: {selected_model}")
        
        self.model = genai.GenerativeModel(selected_model)
        logger.info(f"Initialized GeminiFixRecommender with model: {selected_model}")
    
    def generate_fix(self, buggy_code, bug_type=None, max_retries=3, retry_delay=1):
        """
        Generate a fixed version of the buggy code using Gemini API.
        
        Args:
            buggy_code (str): The code with a bug to fix
            bug_type (str, optional): The type of bug detected, if known
            max_retries (int): Maximum number of retries for API calls
            
        Returns:
            dict: Results including the fixed code and explanation
        """
        if not buggy_code:
            return {
                "success": False,
                "error": "No code provided",
                "fixed_code": None,
                "explanation": None
            }
        
        if bug_type and bug_type != "none" and bug_type in config.BUG_TYPES:
            prompt = f"""
            You are a Python code expert. Here's some code that contains a {bug_type}. Please analyze it carefully and provide a detailed fix:
            
            ```python
            {buggy_code}
            ```
            
            Please follow these guidelines:
            1. Identify the specific bug and its potential impact
            2. Provide a complete, working fix that addresses the root cause
            3. Ensure the fix follows Python best practices
            4. Add appropriate error handling if needed
            5. Maintain the original code's functionality while fixing the bug
            6. Consider edge cases and potential failure points
            7. Add input validation where necessary
            8. Include proper exception handling
            9. Add comments explaining complex logic
            10. Ensure all variables are properly initialized
            
            Format your response EXACTLY like this:
            
            FIXED_CODE:
            ```python
            [your complete fixed code here]
            ```
            
            EXPLANATION:
            [detailed explanation of the bug and your fix]
            
            Your explanation must include:
            1. What specific bug was found
            2. Why it's problematic
            3. How your fix addresses the issue
            4. What improvements were made
            5. Any additional safeguards added
            
            Remember to handle edge cases and ensure the fix is robust.
            """
        else:
            prompt = f"""
            Here's some code that may contain a bug. Please analyze it, fix any issues you find, and explain your changes:
            
            ```python
            {buggy_code}
            ```
            
            Provide the fixed code and a clear explanation of what was wrong and how you fixed it.
            If you don't find any bugs, state that the code appears correct.
            
            Format your response like this:
            
            FIXED_CODE:
            ```python
            [your fixed code here]
            ```
            
            EXPLANATION:
            [your explanation here]
            """
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                
                if hasattr(response, 'text'):
                    response_text = response.text
                elif hasattr(response, 'parts'):
                    response_text = ''.join([part.text for part in response.parts])
                else:
                    response_text = str(response)
                    logger.warning(f"Unexpected response format, converted to string: {response_text[:100]}...")
                
                fixed_code, explanation = self._parse_response(response_text, buggy_code)
                
                if not fixed_code or fixed_code.isspace() or fixed_code == buggy_code:
                    logger.warning("Generated fix appears to be empty or unchanged")
                    return {
                        "success": False,
                        "error": "Failed to generate meaningful fix",
                        "fixed_code": None,
                        "explanation": None
                    }
                
                is_valid, error_msg = is_valid_python_code(fixed_code)
                
                if not is_valid:
                    logger.warning(f"Generated fix has syntax errors: {error_msg}")
                    if attempt == max_retries - 1:
                        return {
                            "success": False,
                            "error": f"Generated fix has syntax errors: {error_msg}",
                            "fixed_code": fixed_code,
                            "explanation": explanation
                        }
                    time.sleep(retry_delay)
                    continue  
                
                return {
                    "success": True,
                    "fixed_code": fixed_code,
                    "explanation": explanation,
                    "original_response": response_text
                }
            
            except Exception as e:
                logger.error(f"Error generating fix (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "error": str(e),
                        "fixed_code": None,
                        "explanation": None
                    }
                time.sleep(retry_delay)
        
        return {
            "success": False,
            "error": "Failed to generate fix after multiple attempts",
            "fixed_code": None,
            "explanation": None
        }
    
    def _parse_response(self, response_text, original_code):
        """
        Parse the Gemini API response to extract fixed code and explanation.
        
        Args:
            response_text (str): The response from the Gemini API
            original_code (str): The original buggy code
            
        Returns:
            tuple: (fixed_code, explanation)
        """
        fixed_code = original_code
        explanation = "No explanation provided."
        
        try:
            if "FIXED_CODE:" in response_text and "```python" in response_text:
                code_parts = response_text.split("```python")
                if len(code_parts) > 1:
                    code_block = code_parts[1].split("```")[0].strip()
                    if code_block:
                        fixed_code = code_block
            
            if "EXPLANATION:" in response_text:
                explanation_parts = response_text.split("EXPLANATION:")
                if len(explanation_parts) > 1:
                    explanation = explanation_parts[1].strip()
            
            if fixed_code == original_code and "```python" in response_text:
                code_parts = response_text.split("```python")
                if len(code_parts) > 1:
                    code_block = code_parts[1].split("```")[0].strip()
                    if code_block:
                        fixed_code = code_block
            
            if explanation == "No explanation provided." and fixed_code != original_code:
                explanation = response_text
                for code_block in response_text.split("```"):
                    if "python" in code_block or fixed_code in code_block:
                        explanation = explanation.replace(f"```{code_block}```", "")
                explanation = explanation.replace("FIXED_CODE:", "").replace("EXPLANATION:", "").strip()
        
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return original_code, f"Error parsing response: {str(e)}"
        
        return fixed_code, explanation
        
    def cleanup(self):
        """
        Clean up resources used by the Gemini API client.
        Call this method when you're done using the recommender to prevent gRPC shutdown timeout issues.
        """
        try:
            if getattr(sys, 'meta_path', None) is None:
                logger.info("Skipping cleanup as Python is shutting down")
                return
                
            self.model = None
            
            import gc
            gc.collect()
            
            logger.info("Cleaned up GeminiFixRecommender resources")
        except Exception as e:
            logger.error(f"Error during GeminiFixRecommender cleanup: {str(e)}")
            
    def __del__(self):
        """
        Destructor to ensure resources are cleaned up when the object is garbage collected.
        """
        self.cleanup()


def test_gemini_fix():
    """Test the Gemini API fix recommendation."""
    buggy_code = """
    def divide_numbers(a, b):
        return a / b
    
    result = divide_numbers(10, 0)
    print(f"Result: {result}")
    """
    
    recommender = GeminiFixRecommender()
    
    try:
        result = recommender.generate_fix(buggy_code, bug_type="logic_error")
        
        print("=== Original Code ===")
        print(buggy_code)
        print("\n=== Fixed Code ===")
        print(result["fixed_code"])
        print("\n=== Explanation ===")
        print(result["explanation"])
        
        return result
    finally:
        recommender.cleanup()


if __name__ == "__main__":
    test_result = test_gemini_fix()
    
    if not test_result["success"]:
        logger.error(f"Gemini API test failed: {test_result['error']}")
    else:
        logger.info("Gemini API test successful")
    
    import atexit
    import gc
    
    def cleanup_resources():
        """Force cleanup of any remaining gRPC resources."""
        try:
            if getattr(sys, 'meta_path', None) is None:
                logger.info("Skipping cleanup as Python is shutting down")
                return
                
            gc.collect()
            logger.info("Cleaned up resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    atexit.register(cleanup_resources)
