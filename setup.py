# *************** IMPORTS ***************
from dotenv import load_dotenv
import logging
import os

logging.basicConfig(
    filename='error.log', # Set a file for save logger output 
    level=logging.INFO, # Set the logging level
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )

LOGGER = logging.getLogger(__name__)
LOGGER.info("Init Global Variable")

# ********* load .env content
load_dotenv(override=True)
# ********* set token for openai and datastax
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
ASTRADB_TOKEN_KEY = os.getenv("ASTRADB_TOKEN_KEY")
ASTRADB_API_ENDPOINT = os.getenv("ASTRADB_API_ENDPOINT")
ASTRADB_COLLECTION_NAME = os.getenv("ASTRADB_COLLECTION_NAME")

GREETINGS_EN = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'hi, how is it going?', 'greetings!', 'how are you doing?', 'how do you do?', 'what`s up?']
GREETINGS_FR = ['bonjour', 'salut', 'coucou', 'bonsoir', 'bonjour à tous', 'comment allez-vous ce matin ?', 'bonne journée', 'bonne soirée', 'bonne nuit', 'À bientôt', 'À plus tard', 'À tout à lheure', 'À demain', 'Ça va?', 'enchanté']


LOGGER.info("Setup Done")