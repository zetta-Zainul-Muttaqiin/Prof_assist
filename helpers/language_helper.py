from deep_translator import GoogleTranslator
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.lingua import LinguaModel


def translate_language(text, target_lang):
    # ********* translate language to English
    try:
        query_lang = LinguaModel().lingua.detect_language_of(text).name.lower()
        translated_query = GoogleTranslator(source=query_lang, target=target_lang).translate(text)
        return translated_query
    except Exception as error_lang:
        print("An error occurred:", error_lang)
        return text

translate_language("halo disana", "en")
