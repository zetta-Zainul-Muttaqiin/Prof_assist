# *************** IMPORT FRAMEWORK *************** 
from langchain_openai import ChatOpenAI
from setup import OPENAI_API_KEY
from typing import Optional

class LLMModels:
    """
    Define LLM models with OpenAI
    """
    def __init__(
            self, 
            temperature: Optional[float] = 0.1, 
            max_tokens: Optional[int] = 4096
            ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_cv = self.create_llm_cv(self.temperature, self.max_tokens)
    # ***** Define a function for intialize OpenAI LLM
    def create_llm_cv(
            self, 
            temperature: Optional[float], 
            max_tokens: Optional[int]
            ):
        llm_model = ChatOpenAI(
            temperature=temperature,
            max_tokens=max_tokens,
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY, 
        )

        return llm_model