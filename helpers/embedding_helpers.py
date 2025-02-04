# *************** IMPORTS ***************
import tiktoken

# *************** Function to count tokens using openai embedding tokenizer
def count_token_embedding_openai(text_input: str, model_name: str = "text-embedding-3-large") -> int:
    """
    Count the number of tokens in a given text input using OpenAI's embedding tokenizer.

    Args:
        text_input (str): The input text to be tokenized.
        model_name (str): Model named for sync the encoding of tiktoken.

    Returns:
        int: The number of tokens in the text input.
    """
    # *************** Get the embedding tokenizer for the specified OpenAI model
    embedding_tiktoken = tiktoken.encoding_for_model(model_name)

    # *************** Encode the input text and count the number of tokens
    num_tokens = len(embedding_tiktoken.encode(text_input))

    # *************** Return the token count
    return num_tokens