# *************** IMPORT FRAMEWORK *************** 
from langchain_openai import OpenAIEmbeddings
from setup import OPENAI_API_KEY

class EmbeddingsModels():
    def __init__(self):
        self.embedding_large_openai = self.create_embedding_large_openai()

    def create_embedding_large_openai(self):
        embedding = OpenAIEmbeddings(
            model="text-embedding-3-large",
            show_progress_bar=True,
            api_key=OPENAI_API_KEY,
            tiktoken_enabled=True,
            tiktoken_model_name='cl100k_base',
        )
        return embedding