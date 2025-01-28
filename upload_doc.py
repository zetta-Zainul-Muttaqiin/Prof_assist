from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from astrapy.db import AstraDB
import datetime
from langchain_community.callbacks          import get_openai_callback

from langchain.docstore.document import Document
from werkzeug.local import Local
import openai
from bs4 import BeautifulSoup
import re
import requests
from dotenv import load_dotenv
import os
import tiktoken
import fitz

from models.embeddings import EmbeddingsModels
from helpers.astradb_connect_helper import get_vector_collection

from setup import (LOGGER,
                   ASTRADB_COLLECTION_NAME_UPLOAD_DOC, 
                   ASTRADB_API_ENDPOINT, 
                   ASTRADB_COLLECTION_NAME, 
                   ASTRADB_TOKEN_KEY, 
                   OPENAI_API_KEY,
                   URL_WEBHOOK)

load_dotenv()
# *************** to save the result of token count for semantic chunking in non-shared global variable 
upload_doc = Local()
def set_semantic_chunker_token(token):
    upload_doc.semantic_chunker_token = token
        
def get_semantic_chunker_token():
    return getattr(upload_doc, "semantic_chunker_token", None)
# *************** end of to save the result of token count for semantic chunking in non-shared global variable 

#*************** retrieve keys from env file and set it to env variable
openai_api_key = OPENAI_API_KEY
astradb_token_key = ASTRADB_TOKEN_KEY
astradb_api_endpoint = ASTRADB_API_ENDPOINT
astradb_collection_name = ASTRADB_COLLECTION_NAME
astradb_collection_name_upload_doc = ASTRADB_COLLECTION_NAME_UPLOAD_DOC

url_webhook = URL_WEBHOOK
url_error_webhook = URL_WEBHOOK
encoding = tiktoken.encoding_for_model("text-embedding-3-large")

#*************** end of retrieve keys from env file and set it to env variable

def tokens_semantic_chunker(data):
    print("Counting tokens for semantic chunking ...")
    # split document based on '.', '?', or '!'
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
        token_semantic_chunker += tokens_embbeding(index['combined_sentence'])
    print(f"Done counting tokens for semantic chunking. The amount of tokens: {token_semantic_chunker}")
    return(token_semantic_chunker)


def tokens_embbeding(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens

#*************** function to load PDF from URL 
def load_doc(pdf_path):
    print("Load pdf...")
    req = requests.get(pdf_path) 
    pdf = fitz.open(stream = req.content, filetype="pdf")
    print("Loader is done!...")
    return pdf

#*************** function to get tag header HTML for each page
def parsing_pdf_html(pdf):
    print("Parsing pdf...")
    join_page = ''
    #*************** parsing each page in pdf to HTML tag
    for page in pdf:
        page_html = page.get_textpage().extractXHTML() 
        search_html = BeautifulSoup(page_html, "html.parser")

        #*************** find all header tag <h> inside div tag
        for line in search_html.div:
            if line.name == "h1" and not line.find("i"):
                join_page += f"{line}" + " "

            elif line.name == "h2" and not line.find("i"):
                join_page += f"{line}" + " "

            elif line.name == "h3" and not line.find("i"):
                join_page += f"{line}" + " "

            elif line.name == "h4" and not line.find("i"):
                join_page += f"{line}" + " "
            
            else:
                join_page += f"{line.text}" + " "
        #*************** end parsing each page in pdf to HTML tag
                
    join_page = join_page.strip()
    print("Parsing is done!...")
    return join_page
#*************** end of get tag header HTML for each page

#*************** function to add \n\n breakspace
def remove_break_add_whitespace(text):
    print("Retaining whitespace...")
    pattern = r"(\n)([a-z])"
    pattern2 = r"\n"

    replacement = r" \2"
    replacement2 = r"\n\n"

    replace_break = re.sub(pattern, replacement, text)
    replace_break = re.sub(pattern2, replacement2, replace_break)
    print("WHite space is retain...")
    return replace_break
#*************** end of function to add \n\n breakspace

#*************** split document based on semantic meaning of the document with breakpoint threshold 75 percentile
def create_document_by_splitting(data):
  tokens_for_semantic_chunker = tokens_semantic_chunker(data)
  set_semantic_chunker_token(tokens_for_semantic_chunker)
  print("Chunking Process...")
  with get_openai_callback() as cb:
    text_splitter = SemanticChunker(
        EmbeddingsModels().embedding_large_openai, 
        breakpoint_threshold_type="percentile", 
        breakpoint_threshold_amount=75
        )
    docs = text_splitter.create_documents([data])
    print(f"Chunking is done... {len(docs)}")
    print(cb)
  return docs
#*************** end of split document based on semantic meaning of the document with breakpoint threshold 75 percentile

#**************** function to retrieve header based on tag '<h>' and save it to metadata header
def extract_headers(document):
    print("extracting header...")
    
    h1 = None # first header 1 is null
    h2 = None # first header 2 is null
    h3 = None # first header 3 is null
    h4 = None # first header 4 is null
    docs = [] # final chunks will append to this array
    clean = re.compile('<.*?>') # for clean all type of HTML

    #**************** set metadata header from each chunks based on tag <h>
    for chunk in document:
        clean_content = re.sub(clean, '', chunk.page_content)
        temp_doc = Document(page_content=clean_content)


        search_tag = BeautifulSoup(chunk.page_content, "html.parser")
        #**************** check each start of sentences with <h> will set as header based on the header level
        for line in search_tag:
            header =  line.text
            if (len(header) > 1) & (re.match(r'^\W' , header) is None):
                if line.name == "h1":
                    h1 = header
                    h2 = None
                    h3 = None
                    h4 = None
                elif line.name == "h2":
                    h2 = header
                    h3 = None
                    h4 = None
                elif line.name == "h3":
                    h3 = header
                    h4 = None
                elif line.name == "h4":
                    h4 = header
        #**************** end of check each start of sentences with <h> will set as header based on the header level 
            #*************** add metadata header to chunk
            temp_doc.metadata = {
                    "header1": h1,
                    "header2": h2,
                    "header3": h3,
                    "header4": h4
                    }
        #**************** end of check sentences start with tag <h>
        docs.append(temp_doc)
    #*************** end of set metadata header based on tag <h>
    print("Header is ready...")
    return docs
#**************** end of function to retrieve header based on tag <h> and save it to metadata header

#*************** function to call webhook to BE once document processing is succcess
def callWebhook(url, course, course_document_id, doc_tokens):

    LOGGER.info('Webhook called... Course ID: ', course, ' || Document ID: ', course_document_id, ' || Tokens Embbed: ', doc_tokens)

    payload = {'course_id': course, 'document_id': course_document_id, 'tokens_embbed': doc_tokens}  # Replace with your JSON payload

    response = requests.post(url, json=payload)
    LOGGER.info('Webhook response: ', response,'Course ID: ', course, ' || Document ID: ', course_document_id, ' || Tokens Embbed: ', doc_tokens)

    #*************** Save webhook call record to AstraDB 
    status = 'Success Uploading'
    time = datetime.datetime.now()
    current_time_serializable = time.isoformat()
    payload['status'] = status
    payload['time'] = current_time_serializable
    db = AstraDB(
        token=astradb_token_key,
        api_endpoint=astradb_api_endpoint
    )
    collection = db.collection(collection_name=astradb_collection_name_upload_doc)
    collection.insert_one(payload)
    #*************** end of save webhook call record to AstraDB 
#*************** end of function to call webhook to BE once document processing is done

#*************** function to call webhook to BE once document processing hits rate limit error  
def callErrorWebhook(url_error, course, course_document_id, doc_tokens, error):

    LOGGER.info('Webhook called... Course ID: ', course, ' || Document ID: ', course_document_id, ' || Tokens Embbed: ', doc_tokens)

    payload = {'course_id': course, 'document_id': course_document_id, 'tokens_embbed': doc_tokens, 'Error': error}  # Replace with your JSON payload

    response = requests.post(url_error, json=payload)
    LOGGER.info('Webhook response: ', response,'Course ID: ', course, ' || Document ID: ', course_document_id, ' || Tokens Embbed: ', doc_tokens, ' || Error Status: ', error)

    #*************** Save webhook call record to AstraDB 
    status = 'Failed Uploading'
    time = datetime.datetime.now()
    current_time_serializable = time.isoformat()
    payload['status'] = status
    payload['time'] = current_time_serializable
    db = AstraDB(
        token=astradb_token_key,
        api_endpoint=astradb_api_endpoint
    )
        
    collection = db.collection(collection_name=astradb_collection_name_upload_doc)
    collection.insert_one(payload)
    #*************** end of save webhook call record to AstraDB 
#*************** end of function to call webhook to BE once document processing hits rate limit error 

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
def Embbed_openaAI(chunks, course_id, course_document_id, doc_tokens):

    with get_openai_callback() as cb:
        #*************** if success, call webhook success
        try:
           #*************** set embedding and vector store
            vector_coll = get_vector_collection()
           #*************** end of set embedding and vector store
            vector_coll.add_documents(chunks)
            print(cb)
            print("Calling Webhook for success uploading")
            return callWebhook(url_webhook, course_id, course_document_id, doc_tokens)
        #*************** end of if success, call webhook success

        #*************** if hits ratelimit error, call error webhook
        except openai.RateLimitError as er:
            print("Fail rate limit error. Call error webhook")
            callErrorWebhook(url_error_webhook, course_id, course_document_id, doc_tokens, er.message)
        #*************** end of if hits ratelimit error, call error webhook
#*************** end of function to run the chunk embeddings and pushing to vector DB

#*************** function to resplit oversize chunk
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