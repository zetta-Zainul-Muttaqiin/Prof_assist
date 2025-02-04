# *************** IMPORTS ***************
import re
import fitz
import openai
import requests
from bs4                                    import BeautifulSoup

# ************ IMPORT FRAMEWORKS ************** 
from langchain.text_splitter                import RecursiveCharacterTextSplitter
from langchain.docstore.document            import Document
from langchain_experimental.text_splitter   import SemanticChunker

# *************** IMPORTS MODELS ***************
from models.embeddings                      import EmbeddingsModels

# *************** IMPORTS HELPERS ***************
from helpers.sending_payload                import log_document_upload
from helpers.embedding_helpers              import count_token_embedding_openai
from helpers.upload_akadbot_helpers         import (
                                                clean_html,
                                                add_metadata,
                                                split_into_sentences,
                                                extract_headers_from_chunk, 
                                                generate_contextual_triplets,
                                            )
from helpers.astradb_connect_helpers        import (
                                                get_vector_collection, 
                                                get_document_collection
                                            )

# *************** IMPORTS VALIDATORS ***************
from validator.chunks_validation            import validate_document_input
from validator.format_validation            import is_url
from validator.data_type_validatation       import (
                                                validate_dict_keys,
                                                validate_list_input,
                                                validate_string_input,
                                            )

# *************** IMPORTS GLOBAL ***************
from setup                                  import (
                                                LOGGER, 
                                                URL_WEBHOOK,
                                            )

# *************** Function to load PDF from URL
def load_doc(pdf_path: str) -> fitz.Document:
    """
    Load a PDF document from a given URL.
    This function for read an URL PDF using request and fitz.

    Args:
        pdf_path (str): URL of the PDF file.

    Returns:
        fitz.Document: The loaded PDF document.
    """
    # *************** Validate pdf_path in URL string
    if not is_url(pdf_path):
        LOGGER.error("'pdf_path' must be a valid URL string.")

    # *************** Get usage for access to the PDF link
    req = requests.get(pdf_path)
    # *************** Raise an error for failed requests
    req.raise_for_status()

    # *************** Open the PDF requested with Fitz in streamable mode
    pdf = fitz.open(stream=req.content, filetype="pdf")

    LOGGER.info(f"Loader is done. Total pages: {len(pdf)}")
    return pdf
    
# *************** Function to get tag header HTML for each page
def parse_pdf_headers(pdf: fitz.Document) -> str:
    """
    Extracts and joins all header tags (<h1>-<h4>) from each page of a PDF, 
    retaining header structure in the output string.

    Args:
        pdf (fitz.Document): The PDF document to parse.

    Returns:
        str: A joined string containing all extracted header tags.
    """
    # *************** Validate fitz document input
    if not pdf or len(pdf) == 0:
        LOGGER.error("The provided PDF document is empty or invalid.")
        raise ValueError("The provided PDF document is empty or invalid.")
    
    # *************** Use a list for efficient string concatenation
    extracted_text = []

    # *************** parsing each page in pdf to HTML tag
    for page in pdf:
        page_html = page.get_textpage().extractXHTML() 
        search_html = BeautifulSoup(page_html, "html.parser")

        # *************** Check div tag is exist in the pdf html
        if not search_html.div:
            LOGGER.warning(f"<div> Not Found on page {pdf.index(page)}")
            continue

        # *************** Get into lines inside the div tag
        for line in search_html.div:
            
            # *************** Retain all headers tag needed and not in Italic format
            if line.name in ("h1", "h2", "h3", "h4") and not line.find("i"):
                extracted_text.append(str(line))
            
            # *************** Except headers will reader as text or page string
            else:
                extracted_text.append(line.text.strip())
                
        # *************** Extract tables from the page 
        tables = page.find_tables()
        
        # *************** if table found loop and convert to dataframe
        if tables.tables: 
            for table in tables:
                df = table.to_pandas()
                
                # *************** if df not empty append it to extracted_text and convert df to string 
                if not df.empty:
                    extracted_text.append("\n Data: " + df.to_string(index=False) + "\n")
    
    # *************** Join extracted text with spaces and return
    return " ".join(extracted_text).strip()

# *************** Function to split document based on semantic meaning of the document with breakpoint threshold 75 percentile
def create_document_by_semantic(chunks_text: str, threshold_amount: int=75) -> list[Document]:
    """
    Splits a document into smaller chunks based on semantic meaning using a percentile-based breakpoint threshold.

    This function takes a large text input, validates it, and applies a semantic chunking method 
    using OpenAI embeddings. The chunking process identifies logical breakpoints based on 
    meaning rather than arbitrary lengths. The threshold is set to the 75th percentile.

    Args:
        chunks_text (str): The input document text to be split into semantic chunks.

    Returns:
        tuple[list[Document], int]: 
            - A list of `Document` objects representing the segmented portions of the text.
            - The total number of chunks created.
    """
    
    # *************** Validate input
    if not validate_string_input(chunks_text, 'chunks_text'):
        LOGGER.error("'chunks_text' must be a string input")
    
    # *************** Initialize semantic chunking with percentile and 75 
    text_splitter = SemanticChunker(
        EmbeddingsModels().embedding_large_openai, 
        breakpoint_threshold_type="percentile", 
        breakpoint_threshold_amount=threshold_amount
    )
    
    # *************** Perform document chunking
    docs = text_splitter.create_documents([chunks_text])
    
    # *************** Log chunking process
    LOGGER.info(f"Chunking is done. Chunks created: {len(docs)}")
    
    return docs

# *************** Function to process all document chunks
def extract_headers(document: list[Document]) -> list[Document]:
    """
    Extracts headers from multiple document chunks and assigns them as metadata.

    Args:
        document (list[Document]): A list of document chunks.

    Returns:
        list[Document]: A list of documents with extracted headers as metadata.
    """

    # *************** Validate the input list of documents
    if not validate_list_input(document, 'document_header'):
        LOGGER.error("'document' must be a list of Document based")

    # *************** Initialize an empty list to store processed documents
    docs = []

    # *************** Iterate through each document chunk
    for chunk in document:
        # *************** Extract headers from each chunk
        headers = extract_headers_from_chunk(chunk)

        # *************** Clean the HTML content of the chunk
        clean_content = clean_html(chunk.page_content)

        # *************** Create a new Document with cleaned content
        temp_doc = Document(page_content=clean_content)

        # *************** Assign the extracted headers as metadata to the document
        temp_doc.metadata = headers

        # *************** Append the processed document to the docs list
        docs.append(temp_doc)

    # *************** Log the number of processed chunks
    LOGGER.info(f"Chunks Listed with Headers: {len(docs)}")

    # *************** Return the list of documents with headers as metadata
    return docs

# *************** Function to track token usage of semantic chunker process
def tokens_semantic_chunker(text_chunks: str) -> int:
    """
    Processes a text by splitting it into sentences, creating contextual triplets, 
    and counting tokens for semantic chunking.

    Args:
        text_chunks (str): The input text to be chunked.

    Returns:
        int: Total number of tokens counted after semantic chunking.
    """
    
    # *************** Validate input text
    if not validate_string_input(text_chunks, 'text_chunk_semantic'):
        LOGGER.error("'text_chunks' must be a valid string input")
        raise ValueError("'text_chunks' must be a valid string input")
    
    # *************** Split text into sentences
    sentences = split_into_sentences(text_chunks)
    
    # *************** Generate contextual triplets
    sentences = generate_contextual_triplets(sentences, buffer_size=1)
    
    # *************** Count tokens for each combined sentence
    total_tokens = sum(count_token_embedding_openai(sentence['combined_sentence']) for sentence in sentences)
    
    LOGGER.info(f"Tokens for semantic chunking estimation: {total_tokens}")
    
    return total_tokens

# *************** Function to handle large document chunks by resplitting
def handle_large_chunk(large_chunk: Document, document_processed: dict) -> tuple[list[Document], int]:
    """
    Handles large document chunks by resplitting them and adding metadata.

    This function detects if a document chunk exceeds a length limit, 
    resplits it into smaller chunks, and updates the metadata for each chunk. 
    It also calculates the total number of tokens across all resplit chunks.

    Args:
        large_chunk (Document): The document chunk that needs to be split.
        document_processed (dict): A dictionary containing document metadata with required keys:
            - course_id (str): The unique identifier of the course.
            - course_name (str): The name of the course.
            - document_name (str): The name of the document.
            - document_id (str): The unique identifier of the document.

    Returns:
        tuple[list[Document], int]: 
            - A list of resplit document chunks with metadata.
            - The total token count after resplitting.
    """

    # *************** Validate input document
    if not validate_document_input(large_chunk, 'large_chunk'):
        LOGGER.error("'large_chunk' must be in Document format")
        raise ValueError("'large_chunk' must be a valid Document object")

    # *************** Validate metadata dictionary
    field_required = {"course_id": str, "course_name": str, "document_id": str, "document_name": str}
    if not validate_dict_keys(document_processed, field_required):
        LOGGER.error(f"'document_processed' is missing required keys: {field_required}")
        raise ValueError(f"'document_processed' must contain keys: {field_required}")

    # *************** Resplit large document into smaller chunks
    resplit_docs, tokens_for_semantic_chunker = resplit_chunk(large_chunk)
    total_tokens = tokens_for_semantic_chunker

    # *************** Process each resplit chunk by adding metadata
    for resplit_doc in resplit_docs:
        resplit_doc = add_metadata(resplit_doc, document_processed)
        total_tokens += resplit_doc.metadata["tokens_embbed"]

    return resplit_docs, total_tokens

# *************** Function to process document chunks with metadata and size handling 
def process_chunks(chunks: list[Document], document_processed: dict) -> tuple[list[Document], int]:
    """
    Processes document chunks by adding metadata and handling oversized chunks.

    This function iterates through a list of document chunks, checking their length. 
    If a chunk exceeds the predefined size limit (7000 characters), it is resplit 
    into smaller chunks. All processed chunks have metadata added, and the total 
    token count is calculated.

    Args:
        chunks (list[Document]): A list of document chunks to process.
        document_processed (dict): A dictionary containing document metadata with required keys:
            - course_id (str): The unique identifier of the course.
            - course_name (str): The name of the course.
            - document_name (str): The name of the document.
            - document_id (str): The unique identifier of the document.

    Returns:
        tuple[list[Document], int]: 
            - A list of processed document chunks with metadata.
            - The total token count across all processed chunks.
    """

    # *************** Validate input chunks
    if not validate_list_input(chunks, 'chunks'):
        LOGGER.error("'chunks' must be a list of Document objects")
        raise ValueError("'chunks' must be a valid list of Document objects")

    # *************** Validate metadata dictionary
    field_required = {"course_id": str, "course_name": str, "document_id": str, "document_name": str}
    if not validate_dict_keys(document_processed, field_required):
        LOGGER.error(f"'document_processed' is missing required keys: {field_required}")
        raise ValueError(f"'document_processed' must contain keys: {field_required}")

    chunks_tokens = 0
    new_chunks = []

    # *************** Iterate through document chunks
    for doc in chunks:
        if len(doc.page_content) > 7000:
            # *************** If document is too large, resplit and add metadata
            resplit_docs, tokens = handle_large_chunk(doc, document_processed)
            new_chunks.extend(resplit_docs)
            chunks_tokens += tokens
        else:
            # *************** Directly add metadata to smaller documents
            doc = add_metadata(doc, document_processed)
            chunks_tokens += doc.metadata["tokens_embbed"]
            new_chunks.append(doc)

    return new_chunks, chunks_tokens

# *************** Function to Run OpenAI Embeddings and Push to Vector DB
def embed_document_openai(chunks: list[Document], document_details: dict) -> None:
    """
    Runs OpenAI embeddings on document chunks and pushes them to the vector database.
    Calls a webhook for success or failure.

    Args:
        chunks (List[Document]): The list of document chunks to be embedded.
        document_details (Dict): Detailed document metadata.
    """
    # *************** if success, call webhook success
    try:
        # *************** set embedding and vector store
        vector_coll = get_vector_collection()
        
        # *************** Push chunks to AstraDB
        vector_coll.add_documents(chunks)
        
        LOGGER.info("Calling Webhook for success uploading")
        log_document_upload(URL_WEBHOOK, document_details, "Success Uploading")

    # *************** if hits ratelimit error, call error webhook
    except openai.RateLimitError as er:
        LOGGER.error("Fail rate limit error. Call error webhook")
        log_document_upload(URL_WEBHOOK, document_details, er.message)

# *************** Function to resplit oversize chunk (>8000 tokens) by reducing the threshold
def resplit_chunk(chunk: Document) -> tuple[list[Document], int]:
    """
    Resplits an oversized chunk into smaller, semantically meaningful sub-chunks.
    If semantic splitting does not work within the defined threshold, falls back to recursive splitting.

    Args:
        chunk (Document): The input document chunk to be resplit.

    Returns:
        Tuple[List[Document], int]: A tuple containing the resplit document chunks and total token count.
    """
    # *************** Validate input chunks
    if not validate_document_input(chunk, 'chunks_resplit'):
        LOGGER.error("'chunks_resplit' must be a Document and page content not empty")

    LOGGER.info("ENTERING SEMANTIC RESPLIT")
    # *************** Count tokens semantic
    tokens_for_semantic_chunker = tokens_semantic_chunker(chunk.page_content)
    documents = process_semantic_resplit(chunk)

    if not is_valid_chunk_split(documents, max_length=7000):
        LOGGER.info("ENTERING RECURSIVE RESPLIT")
        documents = process_recursive_resplit(chunk)

    return documents, tokens_for_semantic_chunker

# *************** Function for Semantic-Based Resplitting
def process_semantic_resplit(chunk: Document) -> list[Document]:
    """
    Attempts to resplit a document chunk using semantic chunking.
    
    The function iteratively reduces the threshold if the initial resplit produces oversized chunks. 
    If a valid split is achieved within the defined maximum iterations, it returns the resplit documents. 
    Otherwise, it returns an empty list, indicating failure.

    Args:
        chunk (Document): The document to be resplit.

    Returns:
        list[Document]: A list of resplit document chunks. If unsuccessful, returns an empty list.
    """
     # *************** Validate input
    if not validate_document_input(chunk, 'chunk_resplit'):
        LOGGER.error("'chunk_resplit' must be a document based")

    # *************** Initial threshold value for semantic splitting
    threshold_amount = 70       
    # *************** Maximum number of attempts to resplit
    max_iterations = 2          
    # *************** Track the number of attempts
    current_iteration = 0       

    while current_iteration < max_iterations:
        LOGGER.info(f"Semantic Resplit Attempt {current_iteration + 1} with Threshold: {threshold_amount}")

        # *************** Generate new document chunks using semantic chunking
        documents = create_document_by_semantic(chunk.page_content, threshold_amount)

        # *************** Validate if all chunks are within the allowed length
        if is_valid_chunk_split(documents, max_length=7000):
            # *************** Successfully resplit, return the documents
            return documents  

        # *************** Reduce threshold and retry
        threshold_amount -= 5
        current_iteration += 1

    # *************** Indicate failure to resplit properly
    return []  

# *************** Function to Validate Chunk Length After Splitting
def is_valid_chunk_split(documents: list[Document], max_length: int) -> bool:
    """
    Validates whether all document chunks adhere to the allowed length constraint.

    The function iterates over all generated chunks and verifies if their length is within 
    the defined `max_length`. It ensures that no chunk exceeds the maximum token length.

    Args:
        documents (list[Document]): The list of document chunks to validate.
        max_length (int): The maximum allowed token length for each chunk.

    Returns:
        bool: True if all document chunks are within the limit, otherwise False.
    """
    return all(len(doc.page_content) <= max_length for doc in documents)

# *************** Function for Recursive Resplitting as a Fallback from resplitting with semantic
def process_recursive_resplit(chunk: Document) -> list[Document]:
    """
    Performs recursive resplitting of a document chunk as a fallback approach.

    RecursiveCharacterTextSplitter to split the text into smaller chunks with a 
    predefined chunk size and overlap.

    Args:
        chunk (Document): The document to be resplit.

    Returns:
        list[Document]: A list of recursively resplit document chunks.
    """
    # *************** Validate input
    if not validate_document_input(chunk, 'chunk_recursive'):
        LOGGER.error("'chunk_recursive' must be a document based")

    # *************** Set chunk size to 512 characters
    # *************** Define an overlap of 50 characters for context retention
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,      
        chunk_overlap=50     
    )

    # *************** Create and return the resplit document chunks
    return text_splitter.create_documents(texts=[chunk.page_content])

# *************** MAIN Function to be called for processing document to vectorized and pushed to Astra
def upload_akadbot_document(url: str, course_id: str, course_name: str, doc_name: str, doc_id: str):
    """
    Processes a document by extracting its text, performing semantic chunking, and embedding it
    before pushing it to AstraDB. This function ensures validation, token tracking, and logging
    throughout the process.

    Args:
        url (str): The URL of the PDF document to be processed.
        course_id (str): The unique identifier for the course.
        course_name (str): The name of the course.
        doc_name (str): The name of the document.
        doc_id (str): The unique identifier for the document.

    Returns:
        None: Logs information and processes document embedding accordingly.

    Raises:
        ValueError: If any input parameter is invalid.
    """
    # *************** Validate input parameters
    if not (validate_string_input(url, 'url') and is_url(url)):
        LOGGER.error("'url' must be a string input and in correct URL format of PDF")
    
    if not validate_string_input(course_id, 'course_id'):
        LOGGER.error("'course_id' must be a string input")
    
    if not validate_string_input(course_name, 'course_name'):
        LOGGER.error("'course_name' must be a string input")
    
    if not validate_string_input(doc_name, 'doc_name'):
        LOGGER.error("'doc_name' must be a string input")
    
    if not validate_string_input(doc_id, 'doc_id'):
        LOGGER.error("'doc_id' must be a string input")
    
    # *************** Initialize document metadata and token tracking
    doc_tokens = 0
    document_processed = {
        "course_id": course_id, 
        "course_name": course_name,  
        "document_id": doc_id,
        "document_name": doc_name
    }

    # *************** Load document from URL
    pdf_page = load_doc(url)
    
    # *************** Extract text content from PDF
    pdf_page_parsed = parse_pdf_headers(pdf_page)
    LOGGER.info(f"Parsed: {pdf_page_parsed}")
    
    # *************** Create document structure for semantic processing
    doc = create_document_by_semantic(pdf_page_parsed)
    
    # *************** Compute semantic chunking token usage
    semantic_tokens = tokens_semantic_chunker(pdf_page_parsed)
    # *************** Update token usage with semantic tokens used
    doc_tokens += semantic_tokens  
    
    # *************** Extract headers and segment document into chunks
    chunks = extract_headers(doc)
    
    # *************** Process document chunks and add metadata
    chunks, chunks_tokens = process_chunks(chunks, document_processed)
    
    # *************** Log token usage
    LOGGER.info(f"Token Usage for uploading: {chunks_tokens}")
    # *************** Update tokens with chunking tokens estimation
    doc_tokens += chunks_tokens
    LOGGER.info(f"Token Usage for uploading + semantic chunking: {doc_tokens}")
    LOGGER.info(f"Chunks Created: {len(chunks)}")
    
    # *************** Store total token usage in metadata
    document_processed["doc_embedd"] = doc_tokens
    
    # *************** Proceed with embedding if token count is within limit
    if doc_tokens < 1_000_000:
        embed_document_openai(chunks, document_processed)  # Embed document
        LOGGER.info(f"Embeddings done for Document: {document_processed}")
    
    # *************** Handle large document error logging
    else:
        error = "PDF Too Large"
        log_document_upload(URL_WEBHOOK, document_processed, error)  # Log error
        LOGGER.error(f"{error} for Document: {document_processed}")

# *************** MAIN Function to delete based on a single document ID
def delete_documents_id(document_id: str) -> str:  
    """
    Deletes a document from the collection based on the provided document ID.
    
    This function interacts with the document collection and removes the document(s) 
    matching the specified document ID. If deletion is paginated and more data remains, 
    the function calls itself recursively until all matching documents are deleted.

    Args:
        document_id (str): The unique identifier of the document to be deleted.
    
    Returns:
        str: Status message of the delete operation.
    """
    # *************** Validate input for list delete documents
    if not validate_string_input(document_id, 'document_id'):
        LOGGER.error("'document_id' must be a string of document id")

    # *************** Get collection target
    collection = get_document_collection()

    # *************** Process document to delete by akadbot
    col_response = collection.delete_many(filter={"metadata.document_id": document_id})
    LOGGER.info(f"Delete Status: {col_response}")
    
    # *************** Warning Not found any data with that document_id
    if col_response.deleted_count == 0:
        status_message = f"Document with ID '{document_id}' not found."
        LOGGER.warning(status_message)

    # *************** Success fully delete document
    else:
        status_message = f"Document ID '{document_id}' deleted successfully!"
        LOGGER.info(status_message)
    
    return status_message

# *************** MAIN Function to delete multiple document IDs from a list
def delete_list_document(list_document: list[str]) -> str:  
    """
    Deletes multiple documents from the collection based on the provided list of document IDs.

    This function removes all documents whose document IDs are present in the provided list. 
    If deletion is paginated and more data remains, it recursively calls the function to 
    continue deleting until all matching documents are removed.

    Args:
        list_document (list): A list of document IDs to be deleted.

    Returns:
        str: Status message of the delete operation.
    """
    # *************** Validate input for list delete documents
    if not validate_list_input(list_document, 'list_document'):
        LOGGER.error("'list_document' must be a list of document id")
    
    # *************** Get collection target
    collection = get_document_collection()

    # *************** Process list document to delete by akadbot
    col_response = collection.delete_many(filter={"metadata.document_id": {'$in': list_document}})
    LOGGER.info(f"Delete Status: {col_response}")
    
    # *************** Warning Not found any data with that list_document
    if col_response.deleted_count == 0:
        status_message = f"Documents with IDs '{list_document}' not found."
        LOGGER.warning(status_message)

    # *************** Success fully delete documents
    else:
        status_message = f"Documents with IDs '{list_document}' deleted successfully!"
        LOGGER.info(status_message)
    
    return status_message


if __name__ == "__main__":
    upload_akadbot_document(
        "https://api.features-v2.zetta-demo.space/fileuploads/MDC-2026---Charte-du-Centre-Partenaire-a53eee77-6fcb-4e1d-99a5-468e0da64ba4-fd69bff7-f17f-43f7-822f-1048ec9b4be1-55584a77-e2c9-4aa7-80f5-1d4368d399cb.pdf",
        "testing_002", 
        "Sea Course", 
        "Blue Ocean", 
        "testing_doc_002"
        )# This is a sample Python script.

    # delete_documents_id('testing_doc_001')