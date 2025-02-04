# *************** IMPORTS ***************
import re
import json
import requests
from requests                        import Response
from datetime                        import datetime
from astrapy                         import DataAPIClient
# *************** IMPORTS HELPERS ***************
from helpers.astradb_connect_helpers import get_document_status_collection

# *************** IMPORTS VALIDATOR ***************
from validator.data_type_validatation import validate_dict_input

# *************** IMPORTS GLOBAL ***************
from setup                           import (LOGGER,
                                             ASTRADB_TOKEN_KEY, 
                                             ASTRADB_API_ENDPOINT,
                                             ASTRA_KEYSPACE_NAME, 
                                             ASRA_COLLECTION_NAME_MEMO_STORAGE)

# *************** Function helper to send payload to urls of webhook target
def send_webhook(url:str, payload:dict) -> Response:
    """
    Sends a POST request with a JSON payload to the specified URL.

    Args:
        url (str): The endpoint URL to which the POST request is sent.
        payload (dict): The data to be included in the body of the request.

    Returns:
        Response: response object and a success message.
    """
    response = requests.post(url, json=payload)
    return response

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

# *************** Function to Call Webhook After Document Processing Success ***************
def log_document_upload(url: str, document_detail: dict, status: str):
    """
    Logs the document upload process by calling a webhook and saving the status.

    This function prepares a webhook payload by setting the document's processing status and timestamp,
    then sends the data to the specified webhook URL. It logs the response and updates the document 
    status in the database.

    Args:
        url (str): The webhook endpoint to which the document details will be sent.
        document_detail (dict): The dictionary containing document metadata.
        status (str): The processing status of the document (e.g., 'success', 'failed').

    Returns:
        None
    """
    # *************** Set the current timestamp in ISO format for serialization
    time = datetime.now()
    current_time_serializable = time.isoformat()
    
    # *************** Update document details with the processing status and timestamp
    document_detail['status'] = status
    document_detail['time'] = current_time_serializable
    
    # *************** Send the updated document details to the webhook target
    response = send_webhook(url, document_detail)  
    LOGGER.info(f"Webhook response: {response} \nDocument Detail: {document_detail}")

    # *************** Save the document processing status in the database
    upload_document_status(document_detail)

# *************** Function to Save Webhook Call Record to AstraDB ***************
def upload_document_status(document_detail: dict):
    """
    Stores the document processing status in the AstraDB collection.

    This function inserts the document processing record into the database for logging and tracking.

    Args:
        document_detail (dict): The dictionary containing document metadata and status.
    """
    # *************** Retrieve the database collection for document statuses
    collection = get_document_status_collection()
    
    # *************** Insert the document processing details into the database
    insert_status = collection.insert_one(document_detail)

    LOGGER.info(f"Puhsed {len(insert_status.raw_results)} chunks to AstraDB {collection.name}")

# *************** function to send log response to astradb
def send_log_data(data:dict):
    """
    Function to send the log data into database
    
    Args:
        data (dict): data dict payload
    """
    if not validate_dict_input(data, "data"):
        print("data to send log must be a dict")
    
    # *************** Set astradb client
    client = DataAPIClient(ASTRADB_TOKEN_KEY)
    database = client.get_database(ASTRADB_API_ENDPOINT)
    coll = database.get_collection(name=ASRA_COLLECTION_NAME_MEMO_STORAGE, namespace=ASTRA_KEYSPACE_NAME)
    
    # *************** Send the data into astradb
    result = coll.insert_one(data)
    inserted_count = len(result.inserted_id)  
    inserted_count = (f"Inserted {inserted_count} documents successfully.")
    LOGGER.info(f"Inserted {inserted_count} documents successfully.")
        