import requests
import json
import re
from astrapy import DataAPIClient

from validator.data_type_validatation import validate_dict_input

from setup import (ASTRADB_TOKEN_KEY, 
                   ASTRADB_API_ENDPOINT, 
                   ASTRA_KEYSPACE_NAME, 
                   ASRA_COLLECTION_NAME_MEMO_STORAGE, 
                   LOGGER)

def send_webhook(url, payload):
    """
    Sends a POST request with a JSON payload to the specified URL.

    Args:
        url (str): The endpoint URL to which the POST request is sent.
        payload (dict): The data to be included in the body of the request.

    Returns:
        None: This function prints the response object and a success message.
    """
    response = requests.post(url, json=payload)
    print(response, "success")

def json_handle_payload(json_str):
    """
    Cleans and parses a JSON string into a Python dictionary.
    
    Args:
        json_str (str): The JSON string to be cleaned and parsed.

    Returns:
        dict: A Python dictionary representation of the JSON string if parsing is successful.
              If parsing fails, returns a dictionary containing an "error" key with the error message.
    """

    # Pre-process to fix common JSON issues like trailing commas
    response_text = re.sub(r',\s*([\]}])', r'\1', json_str)  # Remove trailing commas before closing braces or brackets
    try:
        # Attempt to parse the JSON
        response = json.loads(response_text)
    except json.JSONDecodeError as error:
        error_message = str(error)
        print("Failed to parse JSON response After trying to fix commas:", error_message)
        response = {"error": error_message}
    
    return response

def send_log_data(data:dict):
    """
    Function to send the log data into database
    
    Args:
        data (dict): data dict payload
    """
    if not validate_dict_input(data, "data"):
        print("data to send log must be a dict")
        
    client = DataAPIClient(ASTRADB_TOKEN_KEY)
    database = client.get_database(ASTRADB_API_ENDPOINT)
    coll = database.get_collection(name=ASRA_COLLECTION_NAME_MEMO_STORAGE, namespace=ASTRA_KEYSPACE_NAME)
    
    result = coll.insert_one(data)
    inserted_count = len(result.inserted_id)  
    inserted_count = (f"Inserted {inserted_count} documents successfully.")
    LOGGER.info(f"Inserted {inserted_count} documents successfully.")
        
    