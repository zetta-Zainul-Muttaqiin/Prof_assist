from models.lingua  import LinguaModel
from setup          import LOGGER

# *************** Helper Function to detect langauge used form a text 
def get_language_used(text: str) -> str:
    """
    Get language used by sentence or text.
    Supported with Lingua language detector.

    Args:
        text (str): a string inputted need to know the language used

    Returns:
        string: a language name. (e.g. english or french)
    """
    try:
        lang_detected = LinguaModel().lingua.detect_language_of(text)
        lang_result = lang_detected.name.lower()

        LOGGER.info(f"Language used: {lang_result}")
        
        return lang_result
    
    except Exception as error_lang:
        print("An error occurred when detect langugae with Lingua:", error_lang)
        return text

