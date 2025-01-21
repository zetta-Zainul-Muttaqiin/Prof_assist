from langchain_astradb.vectorstores import AstraDBVectorStore
from astrapy.db import AstraDB as AstraDBPy
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import logging
import os

logging.basicConfig(
    filename='error.log', # Set a file for save logger output 
    level=logging.INFO, # Set the logging level
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )

logger = logging.getLogger(__name__)
logger.info("Init Global Variable")

# ********* load .env content

load_dotenv()
# ********* set token for openai and datastax

openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
astradb_token_key = os.getenv("ASTRADB_TOKEN_KEY")
astradb_api_endpoint = os.getenv("ASTRADB_API_ENDPOINT")
astradb_collection_name = os.getenv("ASTRADB_COLLECTION_NAME")

vstore = AstraDBVectorStore(
embedding=OpenAIEmbeddings(
    max_retries=5, 
    retry_min_seconds=20, 
    retry_max_seconds=60,
    model='text-embedding-3-large'
    ),
collection_name=astradb_collection_name,
api_endpoint=astradb_api_endpoint,
token=astradb_token_key,
namespace="default_keyspace"
)

astra_db_get_topics = AstraDBPy(token=astradb_token_key,
api_endpoint=astradb_api_endpoint)
topics_collection = astra_db_get_topics.collection(collection_name=astradb_collection_name)

logger.info("Connected to DataStax")