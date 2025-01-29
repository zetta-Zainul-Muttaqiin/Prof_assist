# *************** IMPORTS ***************
from dotenv import load_dotenv
import logging
import os
import streamlit as st

logging.basicConfig(
    filename='error.log', # Set a file for save logger output 
    level=logging.INFO, # Set the logging level
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )

LOGGER = logging.getLogger(__name__)
LOGGER.info("Init Global Variable")

# ********* set token for openai and datastax
api_key = st.secrets["api"]
OPENAI_API_KEY = api_key["OPENAI_API_KEY"]

ASTRADB_TOKEN_KEY = api_key["ASTRADB_TOKEN_KEY"]
ASTRADB_API_ENDPOINT = api_key["ASTRADB_API_ENDPOINT"]
ASTRADB_COLLECTION_NAME = api_key["ASTRADB_COLLECTION_NAME"]
ASTRADB_COLLECTION_NAME_UPLOAD_DOC = api_key["ASTRADB_COLLECTION_NAME_UPLOAD_DOC"]

DB_FILE = api_key['DATABASE_CHAT_TOPIC']
LIST_DOC_FILE = api_key["DATA_LIST_DOC"]

ASTRA_KEYSPACE_NAME = api_key["ASTRA_KEYSPACE_NAME"]
ASRA_COLLECTION_NAME_MEMO_STORAGE = api_key["ASRA_COLLECTION_NAME_MEMO_STORAGE"]

URL_WEBHOOK = api_key['URL_WEBHOOK']

GREETINGS_EN = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'hi, how is it going?', 'greetings!', 'how are you doing?', 'how do you do?', 'what`s up?']
GREETINGS_FR = ['bonjour', 'salut', 'coucou', 'bonsoir', 'bonjour à tous', 'comment allez-vous ce matin ?', 'bonne journée', 'bonne soirée', 'bonne nuit', 'À bientôt', 'À plus tard', 'À tout à lheure', 'À demain', 'Ça va?', 'enchanté']


LOGGER.info("Setup Done")