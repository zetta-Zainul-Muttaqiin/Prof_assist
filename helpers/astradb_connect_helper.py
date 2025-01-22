
from langchain_astradb.vectorstores     import AstraDBVectorStore

# *************** LOAD ENVIRONMENT ***************
from setup                              import (
                                            ASTRADB_TOKEN_KEY,
                                            ASTRADB_API_ENDPOINT,
                                            ASTRADB_COLLECTION_NAME,
                                        )

# ************ Function to get AstraDB table Collection ************
def get_vector_collection() -> AstraDBVectorStore:
    """
    Initialize the AstraDBVectorStore and get the AstraDB vector of document collection.
    This function become RAG engine for get context with as_retriever()

    Returns:
        AstraDBVectorStore (object): The AstraDB vector collection object.
    """

    try:

        print("COLL:", ASTRADB_COLLECTION_NAME)

        vector_store_integrated = AstraDBVectorStore(
            collection_name=ASTRADB_COLLECTION_NAME,
            api_endpoint=ASTRADB_API_ENDPOINT,
            token=ASTRADB_TOKEN_KEY,
            namespace="default_keyspace",
            autodetect_collection=True,
        )

        return vector_store_integrated
    except Exception as e:
        # ****** Handle exceptions ******
        raise ConnectionError(f"Failed to connect to AstraDB {ASTRADB_COLLECTION_NAME}: {str(e)}")
