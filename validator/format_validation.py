import re

# *************** Function to Validate URL Format
def is_url(text: str) -> bool:
    """
    Validates whether a given text is a properly formatted URL.

    The function checks if the input string follows a standard URL format, including:
    - HTTP, HTTPS, or FTP protocols.
    - Domain names with valid subdomains.
    - Optional ports and paths.

    If the input does not match a valid URL pattern, it raises a ValueError.

    Args:
        text (str): The input string to be validated as a URL.

    Returns:
        bool: True if the input is a valid URL; otherwise, raises a ValueError.

    Raises:
        ValueError: If the input text is not a valid URL.
    """
    # *************** Protocol: HTTP, HTTPS, or FTP
    # *************** Domain name with optional port
    # *************** Optional path
    url_pattern = re.compile(
        r'^(https?|ftp)://'
        r'([A-Z0-9][A-Z0-9_-]*(?:\.[A-Z0-9][A-Z0-9_-]*)+)\.?(:\d+)?'
        r'(/\S*)?$',
        re.IGNORECASE
    )
    
    # *************** Match the input text against the compiled regex pattern
    url_match = re.match(url_pattern, text) is not None

    # *************** If the input does not match a valid URL format, raise an error
    if not url_match:
        raise ValueError(f"Expected a URL input, got '{text}'")
    
    # *************** Return True if the input is a valid URL
    return True
