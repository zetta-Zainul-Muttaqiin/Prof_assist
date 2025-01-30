validator
# *************** IMPORT ***************
from typing import Any
from setup import LOGGER

# *************** Validator function for chek input in data type int
def validate_int_input(input_number: any, input_name: str = "input", is_empty: bool = True) -> bool:
    """
    Validates that the provided input is a int and is not empty or whitespace.

    Args:
        input_number (any): The input to validate.
        input_name (str, optional): The name of the input being validated, used for error messages. Defaults to "input".
        is_empty (bool, optional): Validation for checking empty int or not

    Raises:
        TypeError: If the input is not a int.
        ValueError: If the input is empty or consists only of whitespace.
    
    Retrun:
        bool: Return True if pass checking int type and not empty
    """
    # *************** Check if input is a int
    if not isinstance(input_number, int):
        raise TypeError(f"{input_name} must be a int. but got {type(input_number)} : {input_number}")
    
    # *************** Check if input is not empty or whitespace
    if not input_number and is_empty:
        raise ValueError(f"{input_name} cannot be empty or whitespace.")
    
    return True

# *************** Validator function for chek input in data type string
def validate_string_input(input_text: any, input_name: str = "input", is_empty: bool = True) -> bool:
    """
    Validates that the provided input is a string and is not empty or whitespace.

    Args:
        input_text (any): The input to validate.
        input_name (str, optional): The name of the input being validated, used for error messages. Defaults to "input".
        is_empty (bool, optional): Validation for checking empty string or not

    Raises:
        TypeError: If the input is not a string.
        ValueError: If the input is empty or consists only of whitespace.
    
    Retrun:
        bool: Return True if pass checking string type and not empty
    """
    # *************** Check if input is a string
    if not isinstance(input_text, str):
        raise TypeError(f"{input_name} must be a string. but got {type(input_text)} : {input_text}")
    
    # *************** Check if input is not empty or whitespace
    if not input_text.strip() and is_empty:
        raise ValueError(f"{input_name} cannot be empty or whitespace.")
    
    return True

# *************** Validator function for chek input in data type string
def validate_dict_input(input_dict: any, input_name: str = "Input dict") -> bool:
    """
    Validates that the provided input is a dict and is not empty.

    Args:
        input_dict (any): The input to validate.
        input_name (str, optional): The name of the input being validated, used for error messages. Defaults to "Input dict".

    Raises:
        TypeError: If the input is not a dict.
        ValueError: If the input is an empty dict.
    Retrun:
        bool: Return True if pass checking string type and not empty
    """
    # *************** Check if input is a dict
    if not isinstance(input_dict, dict):
        raise TypeError(f"{input_name} must be a dict. but got {type(input_dict)} : {input_dict}")
    
    # *************** Check if dict is non-empty
    if not input_dict:
        raise ValueError(f"{input_name} cannot be an empty dict.")
    
    return True
    
# *************** Validator function for chek input in data type list
def validate_list_input(input_list: any, input_name: str = "Input list", is_empty: bool = True) -> bool:
    """
    Validates that the provided input is a list and is not empty.

    Args:
        input_list (any): The input to validate.
        input_name (str, optional): The name of the input being validated, used for error messages. Defaults to "Input list".
        is_empty (bool, optional): Validation for checking empty list or not

    Raises:
        TypeError: If the input is not a list.
        ValueError: If the input is an empty list.

    Returns:
        bool: True if the input is a valid, non-empty list.
    """
    # *************** Check if input is a list
    if not isinstance(input_list, list):
        raise TypeError(f"{input_name} must be a list. but got {type(input_list)} : {input_list}")
    
    # *************** Check if list is non-empty
    if not input_list and is_empty:
        raise ValueError(f"{input_name} cannot be an empty list.")
    
    return True

# *************** Validator Function for check dict in correct format and keys
def validate_filter_entry(filter_entry: any, required_keys: dict):
    """
    Validates a single filter entry dictionary.

    Args:
        filter_entry (dict): The filter entry to validate.

    Returns:
        bool: True if valid, Raise an error if the input is not proper dict and keys
    """
    
    if not isinstance(filter_entry, dict):
        raise TypeError(f"Filter entry must be a dictionary. but got {type(filter_entry)} : {filter_entry}")

    for key, expected_type in required_keys.items():
        if key not in filter_entry:
            raise KeyError(f"Missing required key '{key}' in filter entry.")
        
        if not isinstance(filter_entry[key], expected_type):
            raise KeyError(f"Invalid type for key '{key}'. Expected {expected_type}, got {type(filter_entry[key])}.")

    return True

# *************** function to validate response if it is not a string
def validate_message_response(message: Any) -> str:
    """
    Converts message to string if it's not already a string.
    
    Args:
        message: The message to validate (any type)
        
    Returns:
        str: Message as string
    """
    try:
        if isinstance(message, list):
            return ' '.join(str(item) for item in message)
        if not isinstance(message, str):
            return str(message)
        return message
    except Exception as e:
        LOGGER.error(f"Failed to convert message to string: {str(e)}")
        raise