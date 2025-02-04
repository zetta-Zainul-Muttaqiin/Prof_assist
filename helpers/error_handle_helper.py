# ***************** IMPORT LIBRARY *****************
import traceback
from setup import LOGGER

# ***************** Function to Error handling for try-except detail 
def handle_error(error: Exception, function: str) -> None:
    """
    Prints detailed information about an exception error, including type, message, and traceback.

    Args:
        error (Exception): The caught exception object.
        function (str): The name of the function where the error occurred.
    """
    # ****** Initialize variables for capturing error details 
    error_type = type(error).__name__  # Get the error type (e.g., ValueError)
    error_message = str(error)  # Get the error message as a string
    tb = traceback.format_exc()  #  Capture the traceback details 
    
    # ****** Print error details 
    LOGGER.error(f"Error at ({function})")  # Function where the error occurred
    LOGGER.error(f"Error Type: {error_type}")  # Print the error type
    LOGGER.error(f"Error Message: {error_message}")  # Print the error message
    
    # ****** Complex logic: printing traceback 
    LOGGER.error("Traceback details:")  # Inform the user about traceback
    LOGGER.error(tb)  # Print the full traceback for detailed error analysis