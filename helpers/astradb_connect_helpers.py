
from astrapy                            import DataAPIClient
from langchain_astradb.vectorstores     import AstraDBVectorStore
# *************** LOAD ENVIRONMENT ***************
from setup                              import (
                                            ASTRADB_TOKEN_KEY,
                                            ASTRADB_API_ENDPOINT,
                                            ASTRADB_COLLECTION_NAME,
                                            ASTRADB_COLLECTION_NAME_UPLOAD_DOC
                                        )

# *************** Function to get AstraDB table Collection with vectorizer
def get_vector_collection() -> AstraDBVectorStore:
    """
    Initialize the AstraDBVectorStore and get the AstraDB vector of document collection.
    This function become RAG engine for get context with as_retriever()

    Returns:
        AstraDBVectorStore (object): The AstraDB vector collection object.
    """

    try:

        vector_store_integrated = AstraDBVectorStore(
            collection_name=ASTRADB_COLLECTION_NAME,
            api_endpoint=ASTRADB_API_ENDPOINT,
            token=ASTRADB_TOKEN_KEY,
            namespace="default_keyspace",
            autodetect_collection=True,
        )

        return vector_store_integrated
    except Exception as e:
        # *************** Handle exceptions
        raise ConnectionError(f"Failed to connect to AstraDB {ASTRADB_COLLECTION_NAME}: {str(e)}")

# *************** Function to get AstraDB Document Status collection with Astra Client
def get_document_status_collection():
    """
    Initialize the DataAPIClient and get the AstraDB collection for upload docuemnt status.

    Returns:
        astrapy_collection: The AstraDB collection object.
    """
    try:
        # ********************* Initialize the client and get a "Database" object
        client = DataAPIClient(ASTRADB_TOKEN_KEY)
        database = client.get_database(ASTRADB_API_ENDPOINT)

        # *************** Get collection target to upload docuemnt status collection
        astrapy_collection = database.get_collection(
            ASTRADB_COLLECTION_NAME_UPLOAD_DOC, 
            namespace="default_keyspace"
        )
        return astrapy_collection
    except Exception as e:
        # *************** Handle exceptions
        raise ConnectionError(f"Failed to connect to AstraDB: {str(e)}")
    
# *************** Function to get AstraDB Document collection with Astra Client
def get_document_collection():
    """
    Initialize the DataAPIClient and get the AstraDB collection of document uploaded.

    Returns:
        astrapy_collection: The AstraDB collection object.
    """
    try:
        # ****** Initialize the client and get a "Database" object ******
        client = DataAPIClient(ASTRADB_TOKEN_KEY)
        database = client.get_database(ASTRADB_API_ENDPOINT)

        # *************** Get collection target to document uploaded collection
        astrapy_collection = database.get_collection(
            ASTRADB_COLLECTION_NAME, 
            namespace="default_keyspace"
        )
        return astrapy_collection
    except Exception as e:
        # ****** Handle exceptions ******
        raise ConnectionError(f"Failed to connect to AstraDB: {str(e)}")