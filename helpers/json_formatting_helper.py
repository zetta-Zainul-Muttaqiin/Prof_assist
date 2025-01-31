# *************** IMPORT ***************
import json 
import re 
from setup import LOGGER

# *************** Format json output after invoke 
def format_json_format(output) -> dict:
    """
    Formats output to return a dictionary with 'response' and 'is_answered' keys.
    Handles cases where the answer might already be in JSON format.
    
    Args:
        output: The LLM output dictionary
        
    Returns:
        dict: A dictionary containing 'response' and 'is_answered' keys
    """
    # *************** Handle dictionary output
    if isinstance(output, dict):
        # *************** Get the answer/response
        answer = output.get('answer', '')
        
        # *************** Check if answer is a string that contains JSON
        if isinstance(answer, str):
            try:
                # # *************** Try to parse the answer as JSON
                parsed_answer = json.loads(answer)
                if isinstance(parsed_answer, dict):
                    # *************** If it's already in the correct format, return it directly
                    if "response" in parsed_answer and "is_answered" in parsed_answer:
                        return parsed_answer
                    # ***************If it has different keys, format it
                    return {
                        "response": parsed_answer.get("response", parsed_answer),
                        "is_answered": parsed_answer.get("is_answered", "True")
                    }
            except json.JSONDecodeError:
                # *************** If not JSON, use the answer as is
                pass
                
        # *************** Return formatted dictionary
        return {
            "response": answer,
            "is_answered": output.get('is_answered', 'False')
        }
    
    # *************** Handle string output
    if isinstance(output, str):
        try:
            # *************** Try to parse as JSON
            parsed_output = json.loads(output)
            if isinstance(parsed_output, dict):
                # *************** If it's already in the correct format, return it directly
                if "response" in parsed_output and "is_answered" in parsed_output:
                    return parsed_output
                # *************** If it has different keys, format it
                return {
                    "response": parsed_output.get("response", parsed_output),
                    "is_answered": parsed_output.get("is_answered", "False")
                }
        except json.JSONDecodeError:
            return {
                "response": output,
                "is_answered": "False"
            }
    
    # *************** If output is neither dict nor string, convert to string
    return {
        "response": output,
        "is_answered": "False"
    }