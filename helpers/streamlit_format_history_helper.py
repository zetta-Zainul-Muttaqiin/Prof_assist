# ********** function format chat history to pass in function ask_to_memory
def format_chat_history(chat_history: list) -> list:
    """
    Formats a chat history list into a standardized format.

    Args:
        chat_history (list): A list of messages, where each message is either:
            - A dictionary with keys "type", "content", and optionally "header_ref".
            - An object with attributes `type`, `content`, and optionally `header_ref`.

    Returns:
        list: A list of formatted dictionaries ensuring 'header_ref' is only present for AI messages.
    """
    formatted_chat_history = []

    for msg in chat_history:
        if isinstance(msg, dict):
            # ********** Process message as a dictionary
            msg_dict = {
                "type": msg.get("type"),
                "content": msg.get("content"),
            }
            if msg_dict["type"] == "ai" and "header_ref" in msg:
                msg_dict["header_ref"] = msg["header_ref"]  

            formatted_chat_history.append(msg_dict)

        elif hasattr(msg, 'type') and hasattr(msg, 'content'):
            # ********** Process message as an object
            msg_dict = {
                "type": getattr(msg, 'type', None),
                "content": getattr(msg, 'content', None),
            }
            if msg_dict["type"] == "ai" and hasattr(msg, 'header_ref'):
                msg_dict["header_ref"] = getattr(msg, 'header_ref')

            formatted_chat_history.append(msg_dict)
        else:
            raise ValueError(f"Invalid message format: {msg}")

    return formatted_chat_history

# ********** format chat history from history returned function ask_to_memory
def format_and_extract_header_returned(chat_history: list) -> tuple:
    """
    Formats a chat history list and extracts the `header_ref` for the last AI message.

    Args:
        chat_history (list): A list of messages, where each message is either:
            - An object with attributes `type`, `content`, and optionally `header_ref`.
            - A dictionary with keys "type", "content", and optionally "header_ref".

    Returns:
        tuple: A tuple containing:
            - list: A list of formatted dictionaries with keys:
                - `type` (str): The type of the message (e.g., 'ai', 'human').
                - `content` (str): The content of the message.
                - `header_ref` (str): The header reference for AI messages (optional, default to an empty string).
            - str: The `header_ref` of the last AI message, or an empty string if no AI messages are present.

    """
    formatted_returned_history = []
    header_ref_extracted = ""

    for msg in chat_history:
        if isinstance(msg, dict):
            # ********** Process as a dictionary
            msg_dict = {
                "type": msg.get("type"),
                "content": msg.get("content"),
            }
            if msg_dict["type"] == "ai":
                msg_dict["header_ref"] = msg.get("header_ref", "")
                header_ref_extracted = msg.get("header_ref", "")
            formatted_returned_history.append(msg_dict)
        elif hasattr(msg, 'type') and hasattr(msg, 'content'):
            # ********** Process as an object
            msg_dict = {
                "type": getattr(msg, 'type', None),
                "content": getattr(msg, 'content', None),
            }
            if msg_dict["type"] == "ai":
                msg_dict["header_ref"] = getattr(msg, 'header_ref', "")
                header_ref_extracted = getattr(msg, 'header_ref', "")
            formatted_returned_history.append(msg_dict)
        else:
            raise ValueError(f"Invalid message format: {msg}")

    return formatted_returned_history, header_ref_extracted
