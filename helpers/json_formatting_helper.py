# *************** IMPORT ***************
import json 
import re 
from setup import LOGGER

# *************** Format json output after invoke 
def format_json_format(output: str) -> str:
    """
    Ensures the model's output always follows the required JSON format.
    
    Args:
        output (str): output llm after invoking

    Returns:
        str: output in a data type str but format like a dict.
    """
    
    if isinstance(output, str):
        # *************** Strip markdown JSON formatting if present
        output = re.sub(r"^```json\n|\n```$", "", output.strip())

        try:
            # *************** Attempt to parse JSON
            output_json = json.loads(output)

            # *************** Ensure required keys exist in the JSON output
            if isinstance(output_json, dict) and "response" in output_json and "is_answered" in output_json:
                return output_json
            
        # *************** If JSON decoding fails, enforce the required format below
        except json.JSONDecodeError:
            pass  
            LOGGER.warning("JSON decoding fails, enforce the required format 'response' and 'is_answered'.")

    # *************** If the output is not valid JSON, wrap it as "response" and force "is_answered": "False"
    return {
        "response": output.strip(),
        "is_answered": "False"
    }