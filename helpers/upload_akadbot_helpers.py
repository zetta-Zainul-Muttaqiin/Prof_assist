
# *************** IMPORTS ***************
import re
from bs4                                import BeautifulSoup

# ************ IMPORT FRAMEWORKS ************** 
from langchain.docstore.document        import Document

# *************** IMPORTS HELPERS ***************
from helpers.embedding_helpers          import count_token_embedding_openai

# *************** IMPORTS VALIDATORS ***************
from validator.chunks_validation        import validate_document_input
from validator.data_type_validatation   import validate_string_input, validate_dict_keys

# *************** IMPORTS GLOBAL ***************
from setup import LOGGER

# *************** Function to remove HTML tags
def clean_html(text: str) -> str:
    """
    Removes all HTML tags from the given text.

    Args:
        text (str): The input text containing HTML.

    Returns:
        str: Cleaned text without HTML tags.
    """
    # *************** Validate input of text in string
    if not validate_string_input(text, 'text_clean_html', False):
        LOGGER.error("'text' must be a string")

    # *************** Compile a regex pattern to match HTML tags
    clean_html_tags = re.compile('<.*?>')

    # *************** Remove HTML tags using regex and strip surrounding spaces
    return re.sub(clean_html_tags, '', text).strip()


# *************** Function to extract headers from a chunk
def extract_headers_from_chunk(chunk: Document) -> dict:
    """
    Extracts headers (<h1>-<h4>) from an HTML document chunk.

    Args:
        chunk (Document): A document containing HTML content.

    Returns:
        dict: Dictionary with extracted headers.
    """
    # *************** Validate input Documents
    if not validate_document_input(chunk, 'chunk_header'):
        LOGGER.error("'chunk' must be a Document based")

    # *************** Initialize header variables
    h1, h2, h3, h4 = None, None, None, None

    # *************** Parse the chunk content using BeautifulSoup
    search_tag = BeautifulSoup(chunk.page_content, "html.parser")

    # *************** Find all headers in the document chunk (h1-h4)
    for line in search_tag.find_all(["h1", "h2", "h3", "h4"]):
        if hasattr(line, "name") and line.text.strip():
            header = line.text.strip()

            # *************** Assign headers to respective variables based on their tags
            if line.name == "h1":
                h1, h2, h3, h4 = header, None, None, None
            elif line.name == "h2":
                h2, h3, h4 = header, None, None
            elif line.name == "h3":
                h3, h4 = header, None
            elif line.name == "h4":
                h4 = header

    # *************** Return extracted headers as a dictionary
    return {"header1": h1, "header2": h2, "header3": h3, "header4": h4}

# *************** Function to split text into sentences
def split_into_sentences(text: str) -> list:
    """
    Splits text into individual sentences based on '.', '?', or '!' delimiters.

    Args:
        text (str): The input text to be split.

    Returns:
        list: A list of dictionaries containing sentences and their original indices.
    """
    
    # *************** Validate input text
    if not validate_string_input(text, 'text_split_semantic'):
        LOGGER.error("'text' must be a valid string input")
        raise ValueError("'text' must be a valid string input")
    
    # *************** Split text using regex
    sentence_list = re.split(r'(?<=[.?!])\s+', text)
    
    # *************** Return sentence with index ordering based on the regex order
    return [{'sentence': sentence, 'index': idx} for idx, sentence in enumerate(sentence_list)]

# *************** Function to generate contextual triplets
def generate_contextual_triplets(sentences: list, buffer_size: int) -> list:
    """
    Generates contextual triplets by combining each sentence with its neighboring sentences.

    Args:
        sentences (list): A list of dictionaries containing individual sentences.
        buffer_size (int): The number of adjacent sentences to include.

    Returns:
        list: Updated list with combined sentences for context.
    """
    
    # *************** Validate input list
    if not isinstance(sentences, list) or not all(isinstance(item, dict) and 'sentence' in item for item in sentences):
        LOGGER.error("'sentences' must be a list of dictionaries containing 'sentence' keys")
        raise ValueError("'sentences' must be a valid list of dictionaries containing 'sentence' keys")
    
    if not isinstance(buffer_size, int) or buffer_size < 0:
        LOGGER.error("'buffer_size' must be a non-negative integer")
        raise ValueError("'buffer_size' must be a non-negative integer")
    
    # *************** Generate combined sentences
    for index, sentence_data in enumerate(sentences):
        start_idx = max(0, index - buffer_size)
        end_idx = min(len(sentences), index + buffer_size + 1)
        
        # *************** Combined the sentece by the neighboring sentences indexing 
        sentence_data['combined_sentence'] = " ".join(sentences[idx]['sentence'] for idx in range(start_idx, end_idx))
    
    return sentences

# *************** Function to add metadata to a document chunk
def add_metadata(doc: Document, document_processed: dict) -> Document:
    """
    Adds metadata information to a document chunk.

    This function updates the document's metadata with course and document details. 
    It also ensures that essential metadata fields exist and calculates the token 
    embedding count for the document content.

    Args:
        doc (Document): The document chunk that requires metadata addition.
        document_processed (dict): A dictionary containing document metadata with required keys:
            - course_id (str): The unique identifier of the course.
            - course_name (str): The name of the course.
            - document_name (str): The name of the document.
            - document_id (str): The unique identifier of the document.

    Returns:
        Document: The updated document chunk with metadata.
    
    Raises:
        ValueError: If the document or metadata dictionary is invalid.
    """
    # *************** Validate input document
    if not validate_document_input(doc, 'doc_metadata'):
        LOGGER.error("'doc' must be in Document format")
        raise ValueError("'doc' must be a valid Document object")

    # *************** Validate metadata dictionary
    field_required = {"course_id": str, "course_name": str, "document_id": str, "document_name": str}
    if not validate_dict_keys(document_processed, field_required):
        LOGGER.error(f"'document_processed' is missing required keys: {field_required}")
        raise ValueError(f"'document_processed' must contain keys: {field_required}")

    # *************** Extract metadata fields
    course_id = document_processed["course_id"]
    course_name = document_processed["course_name"]
    doc_id = document_processed["document_id"]
    doc_name = document_processed["document_name"]

    # *************** Ensure all metadata fields exist
    metadata_fields = ["header1", "header2", "header3", "header4"]
    for field in metadata_fields:
        doc.metadata[field] = doc.metadata.get(field, "")

    # *************** Update metadata and compute token embeddings
    doc.metadata.update({
        "course_id": str(course_id),
        "course_name": str(course_name),
        "document_name": str(doc_name),
        "document_id": str(doc_id),
        "tokens_embbed": count_token_embedding_openai(doc.page_content)
    })

    return doc