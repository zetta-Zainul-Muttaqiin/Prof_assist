# *************** IMPORTS FRAMEWORKS ***************
from langchain.callbacks.base   import BaseCallbackHandler

# *************** IMPORTS LIBRARY ***************
import time
from typing                     import Any, Dict, List
from datetime                   import datetime

# *************** IMPORTS GLOBAL ***************
from setup                      import LOGGER

class CustomCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for monitoring agent, tools, retriever, and LLM operations"""
    
    def __init__(self):
        self.start_time = None
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts running."""
        LOGGER.info(f"\n🤖 LLM Started at {datetime.now()}")
        LOGGER.info(f"Prompt length: {len(prompts[0])} characters")
        self.start_time = time.time()

    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM ends running."""
        duration = time.time() - self.start_time
        LOGGER.info(f"🤖 LLM Finished. Duration: {duration:.2f} seconds")
        LOGGER.info(f"Output length: {len(str(response))} characters")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool starts running."""
        LOGGER.info(f"\n🔧 Tool '{serialized.get('name', 'unknown')}' Started")
        LOGGER.info(f"Input: {input_str[:100]}...")  # Truncate long inputs
        
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool ends running."""
        LOGGER.info(f"🔧 Tool Finished")
        LOGGER.info(f"Output: {str(output)[:100]}...")  # Truncate long outputs

    def on_retriever_start(self, query: str, **kwargs) -> None:
        """Called when retriever starts running."""
        LOGGER.info(f"\n📚 Retriever Started")
        LOGGER.info(f"Query: {query}")

    def on_retriever_end(self, documents: List[Any], **kwargs) -> None:
        """Called when retriever ends running."""
        LOGGER.info(f"📚 Retriever Finished")
        LOGGER.info(f"Retrieved {len(documents)} documents")

    def on_agent_action(self, action, **kwargs) -> Any:
        """Called when agent takes an action."""
        LOGGER.info(f"\n👤 Agent Action: {action.tool}")
        LOGGER.info(f"Input: {action.tool_input}")

    def on_agent_finish(self, finish, **kwargs) -> None:
        """Called when agent finishes running."""
        LOGGER.info(f"\n👤 Agent Finished")
        LOGGER.info(f"Output: {finish.return_values.get('output', '')}")