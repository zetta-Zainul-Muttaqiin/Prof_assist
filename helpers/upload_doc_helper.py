# ************ IMPORT ************
from typing import Optional
from werkzeug.local import Local
import tiktoken 

# ************ Token counter setup
class TokenCounter:
    """
    Manages token counting for semantic chunking
    """
    _local = Local()
    _encoding = tiktoken.encoding_for_model("text-embedding-3-large")

    @classmethod
    def set_semantic_chunker_token(cls, token: int) -> None:
        cls._local.semantic_chunker_token = token

    @classmethod
    def get_semantic_chunker_token(cls) -> Optional[int]:
        return getattr(cls._local, "semantic_chunker_token", None)

    @classmethod
    def count_tokens(cls, text: str) -> int:
        """Count tokens in a text string"""
        return len(cls._encoding.encode(text))
    
    
    