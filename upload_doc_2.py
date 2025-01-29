# ************ IMPORT FRAMEWORK ************
from langchain_experimental.text_splitter   import SemanticChunker
from langchain.text_splitter                import RecursiveCharacterTextSplitter
from langchain_community.callbacks          import get_openai_callback
from langchain.docstore.document            import Document

from setup import (LOGGER, 
                    OPENAI_API_KEY, 
                    ASTRADB_API_ENDPOINT, 
                    URL_WEBHOOK,
                    URL_ERROR_WEBHOOK,
                    ASTRADB_COLLECTION_NAME, 
                    ASTRADB_COLLECTION_NAME_UPLOAD_DOC, 
                    ASTRADB_TOKEN_KEY)

# ************ IMPORT ************
from astrapy.db import AstraDB
from dataclasses import dataclass
import datetime
from werkzeug.local import Local
import openai
from bs4 import BeautifulSoup
import re
import requests
from dotenv import load_dotenv
import os
import tiktoken
import fitz
from typing import Optional, List
from models.embeddings import EmbeddingsModels

# ************ IMPORT HELPER ************
from helpers.astradb_connect_helper import get_vector_collection
from helpers.upload_doc_helper import TokenCounter
# ************ IMPORT VALIDATOR ************
from validator.data_type_validatation import (validate_dict_input, 
                                              validate_filter_entry, 
                                              validate_int_input, 
                                              validate_list_input, 
                                              validate_string_input)

load_dotenv()

# *************** config data number
@dataclass 
class Config:
    """
    Configuration container for environment variables and settings
    """
    MAX_DOCUMENT_TOKENS: int = 1_000_000
    MAX_CHUNK_CHARS: int = 7000
    BREAKPOINT_THRESHOLD = 75

# ************ set config as a global variable
CONFIG = Config()

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
    
# ************ Tokens semantic chunker
def tokens_semantic_chunker(data):
    """

    Args:
        data (_type_): _description_
    """
    
    LOGGER.info("Counting tokens for semantic chunking ...")
    #*************** split document based on '.', '?', or '!'
    single_sentences_list = re.split(r'(?<=[.?!])\s+', data)
    sentences = [{'sentence':x, 'index':i} for i, x in enumerate(single_sentences_list)]
    buffer_size = 1 
    #*************** combining sentences in triplet
    for sentence in range(len(sentences)):
        combined_sentence = ''
        for index in range(sentence - buffer_size, sentence):
            if index >= 0:
                combined_sentence += sentences[index]['sentence'] + ''
        combined_sentence += sentences[sentence]['sentence']
        for index in range(sentence + 1, sentence + 1 + buffer_size):
            if index < len(sentences):
                combined_sentence += ' ' + sentences[index]['sentence']
        sentences[sentence]['combined_sentence'] = combined_sentence
    #*************** end of combining sentences in triplet
    token_semantic_chunker = 0
    for index in sentences:
        token_semantic_chunker += TokenCounter.count_tokens(index['combined_sentence'])

    LOGGER.info(f"Done counting tokens for semantic chunking. The amount of tokens: {token_semantic_chunker}")
    return(token_semantic_chunker)

# *************** function to load PDF from URL 
def load_pdf(pdf_url: str) -> fitz.Document:
    """
    Load PDF from URL with error handling.
    """
    # *************** validate input data type and end with .pdf
    if not validate_string_input(pdf_url, "pdf_url") or pdf_url.endswith(".pdf"):
        LOGGER.error("pdf_url must be a string and end with .pdf")
    
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        return fitz.open(stream=response.content, filetype="pdf")
    except requests.exceptions.RequestException as e:
        LOGGER.error(f"Failed to download PDF from {pdf_url}: {str(e)}")
        raise ValueError(f"Failed to load PDF: {str(e)}")

#*************** function to parse loaded doc into html
def parse_pdf_html(pdf: fitz.Document) -> str:
    """
    Extract and parse HTML content from PDF.
    """
    
    try:
        content = []
        # ************ parsing each page in pdf to HTML tag
        for page in pdf:
            page_html = page.get_textpage().extractXHTML()
            soup = BeautifulSoup(page_html, "html.parser")
            
            # *************** find all header tag <h> inside div tag
            for line in soup.div:
                if line.name in ["h1", "h2", "h3", "h4"] and not line.find("i"):
                    content.append(str(line))
                else:
                    content.append(line.text)
                    
        return " ".join(content).strip()
    except Exception as e:
        LOGGER.error(f"Failed to parse PDF: {str(e)}")
        raise ValueError(f"PDF parsing failed: {str(e)}")

#*************** function to add \n\n breakspace (remove_break_add_whitespace)
def process_text(text: str) -> str:
    """
    Process text by handling line breaks and whitespace.
    """
    # *************** validate input 
    if not validate_string_input(text, "string"):
        LOGGER.error("text to clean must be a string")
    
    try:
        text = re.sub(r"(\n)([a-z])", r" \2", text)
        return re.sub(r"\n", r"\n\n", text)
    except Exception as e:
        LOGGER.error(f"Text processing failed: {str(e)}")
        raise ValueError(f"Text processing error: {str(e)}")

#*************** split document based on semantic meaning of the document with breakpoint threshold 75 percentile
def create_document_by_splitting(page: str):
    """

    """
    
    if not validate_string_input(page, "page"):
        LOGGER.error("page for doc splitting must be a string")
    
    tokens_for_semantic_chunker = tokens_semantic_chunker(page)
    TokenCounter.set_semantic_chunker_token(tokens_for_semantic_chunker)

    LOGGER.info("Chunking Procces...")
    
    with get_openai_callback() as cb:
        text_splitter = SemanticChunker(
            EmbeddingsModels().embedding_large_openai, 
            breakpoint_threshold_type="percentile", 
            breakpoint_threshold_amount=75
            )
        docs = text_splitter.create_documents([page])
        LOGGER.info(f"Chunking is done... {len(docs)}")
    return docs

#**************** function to retrieve header based on tag '<h>' and save it to metadata header
def extract_headers(document: List[Document]) -> List[Document]:
    """
    Extract headers from documents and add to metadata.
    """
    try:
        clean = re.compile('<.*?>')
        docs = []
        h1 = h2 = h3 = h4 = None

        for chunk in document:
            clean_content = re.sub(clean, '', chunk.page_content)
            temp_doc = Document(page_content=clean_content)
            
            soup = BeautifulSoup(chunk.page_content, "html.parser")
            for line in soup:
                header = line.text
                if len(header) > 1 and not re.match(r'^\W', header):
                    if line.name == "h1":
                        h1, h2, h3, h4 = header, None, None, None
                    elif line.name == "h2":
                        h2, h3, h4 = header, None, None
                    elif line.name == "h3":
                        h3, h4 = header, None
                    elif line.name == "h4":
                        h4 = header

            temp_doc.metadata = {
                "header1": h1,
                "header2": h2,
                "header3": h3,
                "header4": h4
            }
            docs.append(temp_doc)
        LOGGER.info("header is done")
        return docs
    
    except Exception as e:
        LOGGER.error(f"Header extraction failed: {str(e)}")
        raise ValueError(f"Failed to extract headers: {str(e)}")


# ************ function to call webhook to BE once document processing is succcess
def send_webhook(url: str, course_id: str, doc_id: str, tokens: int, error: Optional[str] = None) -> None:
    """
    Send webhook notification with processing results.
    """
    
    inputs = [url, course_id, doc_id]
    for input in inputs:
        if not validate_string_input(input, f"{input}"):
            LOGGER.error(f"{input} must be a string")
    
    if not validate_int_input(tokens, "tokens"):
        LOGGER.error("tokens must be a int")
    
    try:
        payload = {
            'course_id': course_id,
            'document_id': doc_id,
            'tokens_embbed': tokens,
            'status': 'Failed Uploading' if error else 'Success Uploading',
            'time': datetime.now().isoformat()
        }
        if error:
            payload['Error'] = error

        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # ************ Store webhook record in AstraDB
        db = AstraDB(token=ASTRADB_TOKEN_KEY,
                    api_endpoint=ASTRADB_API_ENDPOINT)
        collection = db.collection(ASTRADB_COLLECTION_NAME_UPLOAD_DOC)
        collection.insert_one(payload)
        
    except Exception as e:
        LOGGER.error(f"Webhook call failed: {str(e)}")
        raise ValueError(f"Webhook notification failed: {str(e)}")

#*************** function to call webhook to BE once document processing hits rate limit error  
# def callErrorWebhook(url_error, course, course_document_id, doc_tokens, error):

    # LOGGER.info('Webhook called... Course ID: ', course, ' || Document ID: ', course_document_id, ' || Tokens Embbed: ', doc_tokens)

    # payload = {'course_id': course, 'document_id': course_document_id, 'tokens_embbed': doc_tokens, 'Error': error}  # Replace with your JSON payload

    # response = requests.post(url_error, json=payload)
    # LOGGER.info('Webhook response: ', response,'Course ID: ', course, ' || Document ID: ', course_document_id, ' || Tokens Embbed: ', doc_tokens, ' || Error Status: ', error)

    # #*************** Save webhook call record to AstraDB 
    # status = 'Failed Uploading'
    # time = datetime.datetime.now()
    # current_time_serializable = time.isoformat()
    # payload['status'] = status
    # payload['time'] = current_time_serializable
    # db = AstraDB(
    #     token=astradb_token_key,
    #     api_endpoint=astradb_api_endpoint
    # )
        
    # collection = db.collection(collection_name=astradb_collection_name_upload_doc)
    # collection.insert_one(payload)

#*************** main function to be called for processing document
def callRequest(URL: str, course_id: str, course_name:str,  doc_name: str, doc_id: str):

    pdf_page = load_doc(URL)
    if(pdf_page):
        LOGGER.info('Document downloaded... Course ID: ', course_id, ' || Document ID: ', doc_id)

    pdf_page_parsed = parsing_pdf_html(pdf_page)
    removed_break = remove_break_add_whitespace(pdf_page_parsed)
    doc = create_document_by_splitting(removed_break)
    chunks = extract_headers(doc)
    doc_tokens = 0
    print("insert metedata information")
    #*************** add metadata course ID, document name, and document ID to chunks
    new_chunks = []
    for doc in chunks:
        #*************** Check if doc.page_content character count is more than 7000. If yes, resplit the chunk
        if len(doc.page_content) > 7000:
            resplit = resplit_chunk(doc)
            for resplit_doc in resplit:
                resplit_doc.metadata['header1'] = doc.metadata['header1']
                resplit_doc.metadata['header2'] = doc.metadata['header2']
                resplit_doc.metadata['header3'] = doc.metadata['header3']
                resplit_doc.metadata['header4'] = doc.metadata['header4']
                resplit_doc.metadata["course_id"] = f"{course_id}"
                resplit_doc.metadata["course_name"] = f"{course_name}"
                resplit_doc.metadata["document_name"] = f"{doc_name}"
                resplit_doc.metadata["document_id"] = f"{doc_id}"
                x = tokens_embbeding(doc.page_content)
                resplit_doc.metadata["tokens_embbed"] = x
                doc_tokens += x
            new_chunks.extend(resplit)
        #*************** end of Check if doc.page_content character count is more than 7000. If yes, resplit the chunk
        else:
            doc.metadata["course_id"] = f"{course_id}"
            doc.metadata["course_name"] = f"{course_name}"
            doc.metadata["document_name"] = f"{doc_name}"
            doc.metadata["document_id"] = f"{doc_id}"
            x = tokens_embbeding(doc.page_content)
            doc.metadata["tokens_embbed"] = x
            doc_tokens += x
            new_chunks.append(doc)

    chunks = new_chunks
    #*************** end of add metadata course ID, document name, and document ID to chunks
    print(f"Token Usage for uploading: {doc_tokens}")
    doc_tokens += get_semantic_chunker_token()
    print(f"Token Usage for uploading + semantic chunking: {doc_tokens}")
    print(f"chunks: {len(chunks)}")
    print(f"token usage : {doc_tokens}")
    if doc_tokens < 1000000:
        print("Embedding Process")
        Embbed_openaAI(chunks, course_id, doc_id, doc_tokens)
        print("Embeddings done")
    else:
        error = "PDF Too Large"
        callErrorWebhook(url_error_webhook, course_id, doc_id, doc_tokens, error)
#*************** end of main function to be called for processing document

#*************** function to run the chunk embeddings and pushing to vector DB
def embed_documents(chunks: List[Document], course_id: str, doc_id: str, 
                   tokens: int) -> None:
    """Embed documents and store in vector database."""
    try:
        with get_openai_callback() as cb:
            vector_coll = get_vector_collection()
            vector_coll.add_documents(chunks)
            send_webhook(CONFIG.URL_WEBHOOK, course_id, doc_id, tokens)
    except openai.RateLimitError as e:
        LOGGER.error(f"Rate limit exceeded: {str(e)}")
        send_webhook(CONFIG.URL_ERROR_WEBHOOK, course_id, doc_id, tokens, str(e))
    except Exception as e:
        LOGGER.error(f"Document embedding failed: {str(e)}")
        send_webhook(CONFIG.URL_ERROR_WEBHOOK, course_id, doc_id, tokens, str(e))




def resplit_chunk(chunk):
    threshold_amount = 70  # Starting threshold amount
    max_iterations = 2
    current_iteration = 0
    is_semantic_successful = False

    while current_iteration < max_iterations:
        print("ENTERING SEMANTIC RESPLIT")
        docs = None
        print('RESPLIT CHUNK: ', chunk)
        tokens_for_semantic_chunker = tokens_semantic_chunker(chunk.page_content)
        set_semantic_chunker_token(tokens_for_semantic_chunker + get_semantic_chunker_token())
        text_splitter = SemanticChunker(
            EmbeddingsModels().embedding_large_openai, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=threshold_amount
        )
        docs = text_splitter.create_documents(texts=[chunk.page_content])
        all_docs_below_threshold = all(len(doc.page_content) <= 7000 for doc in docs)
        if all_docs_below_threshold:
            is_semantic_successful = True
            break  # Exit the loop if all documents satisfy the condition

        # Decrease the threshold amount by 5 for the next iteration
        threshold_amount -= 5
        current_iteration += 1

    if(is_semantic_successful is False):
        print("ENTERING RECURSIVE RESPLIT")
        docs = None
        print('RESPLIT CHUNK: ', chunk)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        docs = text_splitter.create_documents(texts=[chunk.page_content])
    print('RESULT RESPLIT: ', docs)
    return docs
#*************** end of function to resplit oversize chunk

#*************** function to delete based on document ID
def delete(document_id):
    from astrapy.db import AstraDB, AstraDBCollection

    astradb_token_key = ASTRADB_TOKEN_KEY
    astradb_api_endpoint = ASTRADB_API_ENDPOINT
    astradb_collection_name = ASTRADB_COLLECTION_NAME
    
    astra_db = AstraDB(token=astradb_token_key,
                       api_endpoint=astradb_api_endpoint)
    collection = AstraDBCollection(collection_name=astradb_collection_name, astra_db=astra_db)
    col_response = collection.delete_many(filter={"metadata.document_id": document_id})
    print(col_response)
    #************** deleting is paginated per 20 data. Recursive if status is still 'moreData'
    if 'moreData' in col_response['status']:
        print("deleting again")
        delete(document_id)
    else:
        print("delete is done!")
    #************** end of deleting is paginated per 20 data. Recursive if status is still 'moreData'
#*************** end of function to delete based on document ID

if __name__ == "__main__":
    callRequest("https://www2.ed.gov/documents/ai-report/ai-report.pdf","65c46edc510f69a052a47119", "ai", "65c46f04510f69a052a47148")# This is a sample Python script.