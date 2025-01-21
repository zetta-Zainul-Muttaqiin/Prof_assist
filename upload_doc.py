# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from langchain.document_loaders import (
        PyPDFLoader,
        UnstructuredMarkdownLoader,
        PDFMinerLoader,
        PDFMinerPDFasHTMLLoader,
    )
from langchain.text_splitter import (
        CharacterTextSplitter,
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import AstraDB
from langchain.callbacks import get_openai_callback
from langchain.docstore.document import Document
import openai
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import requests
import markdown
from dotenv import load_dotenv
import os
import logging
import tiktoken
import time

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] =  openai_api_key
astradb_token_key = os.getenv("ASTRADB_TOKEN_KEY")
astradb_api_endpoint = os.getenv("ASTRADB_API_ENDPOINT")
astradb_collection_name = os.getenv("ASTRADB_COLLECTION_NAME")
url_webhook = os.getenv("URL_WEBHOOK")
url_error_webhook = os.getenv("URL_ERROR_WEBHOOK")

def tokens_embbeding(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def load_doc(pdf_path):
    print("Load pdf...")
    loader = PDFMinerPDFasHTMLLoader(pdf_path)
    data = loader.load()[0]
    print("Loader is done!...")
    return data

def parsing_pdf_html(pages):
    print("Parsing process...")
    soup = BeautifulSoup(pages.page_content, "html.parser")
    content = soup.find_all("div")

    cur_fs = None
    cur_text = ""
    snippets = []  # first collect all snippets that have the same font size
    for c in content:
        sp = c.find("span")
        if not sp:
            continue
        st = sp.get("style")
        if not st:
            continue
        fs = re.findall("font-size:(\d+)px", st)
        if not fs:
            continue
        fs = int(fs[0])
        if not cur_fs:
            cur_fs = fs
        if fs == cur_fs:
            cur_text += c.text
        else:
            snippets.append((cur_text, cur_fs))
            cur_fs = fs
            cur_text = c.text
    snippets.append((cur_text, cur_fs))
    print("parsing is done!...")
    return snippets

def to_markdown(snippet):
    print("Converting to Markdown...")
    df = pd.DataFrame(snippet, columns=["text", "font_size"])
    # set header condition
    size1 = df["font_size"] >= 30
    size2 = (df["font_size"] < 30) & (df["font_size"] >= 18)
    size3 = (df["font_size"] < 17) & (df["font_size"] >= 14)
    size4 = (df["font_size"] < 14) & (df["font_size"] >= 10)

    size_is = [size1, size2, size3, size4]
    mark_is = ["#", "##", "###", "####"]

    # set header and join text and header
    df["header"] = np.select(size_is, mark_is, default="")
    df["markdown"] = df["header"] + " " + df["text"]
    # one string
    mark = "".join(df["markdown"].tolist())
    print("Markdown is ready...")
    return mark


def remove_break_add_whitespace(mark):
    print("Retaining whitespace...")
    pattern = r"(\n)([a-z])"
    pattern2 = r"\n"

    replacement = r" \2"
    replacement2 = r"\n\n"

    replace_break = re.sub(pattern, replacement, mark)
    replace_break = re.sub(pattern2, replacement2, replace_break)
    print("WHite space is retain...")
    return replace_break


def text_to_markdown_head(string, filename):
    md = markdown.markdown(string)
    with open("/content/" + filename, "w", encoding="utf-8") as f:
        f.write(md)


def create_document_by_splitting(data):
    print("Chungking Process...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=25,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.create_documents([data])
    split = text_splitter.split_documents(docs)
    print("Chungking is done...")
    return docs


def extract_headers(document):
    print("extracting header...")
    docs = []
    h1 = None
    h2 = None
    h3 = None
    h4 = None

    for doc in document:
        page_content = doc.page_content
        lines = page_content.split("\n\n")
        temp_doc = Document(page_content=doc.page_content)

        for line in lines:
            if line.startswith("#"):
                header_level = line.count("#")
                header_text = line.lstrip("#").strip()

                if header_level == 1:
                    h1 = header_text
                    h2 = None
                    h3 = None
                    h4 = None
                    temp_doc.metadata = {
                        "header1": h1,
                        "header2": h2,
                        "header3": h3,
                        "header4": h4,
                    }
                    # print('detect H1')
                elif header_level == 2:
                    h2 = header_text
                    h3 = None
                    h4 = None
                    temp_doc.metadata = {
                        "header1": h1,
                        "header2": h2,
                        "header3": h3,
                        "header4": h4,
                    }
                    # print('detect H2')
                elif header_level == 3:
                    h3 = header_text
                    h4 = None
                    temp_doc.metadata = {
                        "header1": h1,
                        "header2": h2,
                        "header3": h3,
                        "header4": h4,
                    }
                    # print('detect H3')
                elif header_level == 4:
                    h4 = header_text
                    temp_doc.metadata = {
                        "header1": h1,
                        "header2": h2,
                        "header3": h3,
                        "header4": h4,
                    }
                    # print('detect H4')
            else:
                temp_doc.metadata = {
                    "header1": h1,
                    "header2": h2,
                    "header3": h3,
                    "header4": h4,
                }
        docs.append(temp_doc)
    print("Header is ready...")
    return docs


def callWebhook(url, course, course_document_id, doc_tokens):
    logging.basicConfig(level=logging.INFO,  # Set the logging level
                        format='%(asctime)s [%(levelname)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger(__name__)

    logger.info('Webhook called... Course ID: ', course, ' || Document ID: ', course_document_id, ' || Tokens Embbed: ', doc_tokens)

    payload = {'course_id': course, 'document_id': course_document_id, 'tokens_embbed': doc_tokens}  # Replace with your JSON payload

    response = requests.post(url, json=payload)
    logger.info('Webhook response: ', response,'Course ID: ', course, ' || Document ID: ', course_document_id, ' || Tokens Embbed: ', doc_tokens)

def callErrorWebhook(url_error, course, course_document_id, doc_tokens, error):
    logging.basicConfig(level=logging.INFO,  # Set the logging level
                        format='%(asctime)s [%(levelname)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger(__name__)

    logger.info('Webhook called... Course ID: ', course, ' || Document ID: ', course_document_id, ' || Tokens Embbed: ', doc_tokens)

    payload = {'course_id': course, 'document_id': course_document_id, 'tokens_embbed': doc_tokens, 'Error': error}  # Replace with your JSON payload

    response = requests.post(url_error, json=payload)
    logger.info('Webhook response: ', response,'Course ID: ', course, ' || Document ID: ', course_document_id, ' || Tokens Embbed: ', doc_tokens, ' || Error Status: ', error)

def callRequest(URL, course_id, file_name, course_document_id):

    logging.basicConfig(level=logging.INFO,  # Set the logging level
                        format='%(asctime)s [%(levelname)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger(__name__)

    pdf_page = load_doc(URL)
    if(pdf_page):
        logger.info('Document downloaded... Course ID: ', course_id, ' || Document ID: ', course_document_id)
    snippets = parsing_pdf_html(pdf_page)
    mark = to_markdown(
            snippets
        )  # << use the output to chunk splitter by header markdown
    mark_white = remove_break_add_whitespace(mark)
    doc = create_document_by_splitting(mark_white)

    chunks = extract_headers(doc)
    doc_tokens = 0
    print("insert metedata information")
    for doc in chunks:
        doc.metadata["source"] = f"{course_id}"
        doc.metadata["document_name"] = f"{file_name}"
        doc.metadata["document_id"] = f"{course_document_id}"
        x = tokens_embbeding(doc.page_content)
        doc.metadata["tokens_embbed"] = x
        doc_tokens += x
    print(f"chunks: {len(chunks)}")
    # for i, _ in enumerate(chunks):
    #     print(f"chunk #{i}, size: {chunks[i]}")
    #     print(" - " * 100)
    print(f"token usage : {doc_tokens}")
    # Commented to remove validation for maximum of 150.000 tokens.
    # if doc_tokens < 150000:
    #     print("Embedding Process")
    #     Embbed_openaAI(chunks, course_id, course_document_id, doc_tokens)
    #     print("Embeddings done")
    # else:
    #     print("PDF is too large")
    #     error = "Tokens too large for LLM model OpenAI"
    #     callErrorWebhook(url_error_webhook, course_id, course_document_id, doc_tokens, error)
    print("Embedding Process")
    Embbed_openaAI(chunks, course_id, course_document_id, doc_tokens)
    print("Embeddings done")

def Embbed_openaAI(chunks, course_id, course_document_id, doc_tokens):
    with get_openai_callback() as cb:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", max_retries=3, retry_min_seconds=30,
                                      retry_max_seconds=60, show_progress_bar=True)
        vstore = AstraDB(
            embedding=embeddings,
            collection_name=astradb_collection_name,
            api_endpoint=astradb_api_endpoint,
            token=astradb_token_key,
            )
        try:
            vstore.add_documents(chunks)
            print(cb)
            print("Calling Webhook for success uploading")
            return callWebhook(url_webhook, course_id, course_document_id, doc_tokens)
        except openai.RateLimitError as er:
            print("Fail rate limit error. Call error webhook")
            callErrorWebhook(url_error_webhook, course_id, course_document_id, doc_tokens, er.message)

def delete(course_id):
    from astrapy.db import AstraDB, AstraDBCollection
    astradb_token_key = os.getenv("ASTRADB_TOKEN_KEY")
    astradb_api_endpoint = os.getenv("ASTRADB_API_ENDPOINT")
    astradb_collection_name = os.getenv("ASTRADB_COLLECTION_NAME")

    astra_db = AstraDB(token=astradb_token_key,
                       api_endpoint=astradb_api_endpoint)
    collection = AstraDBCollection(
        collection_name=astradb_collection_name, astra_db=astra_db)

    col_response = collection.delete_many(filter={"metadata.source": course_id})
    print(col_response)
    if 'moreData' in col_response['status']:
        print("deleting again")
        delete(course_id)
    else:
        print("delete is done!")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    callRequest("https://www2.ed.gov/documents/ai-report/ai-report.pdf","65c46edc510f69a052a47119", "ai", "65c46f04510f69a052a47148")# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from langchain.document_loaders import (
        PyPDFLoader,
        UnstructuredMarkdownLoader,
        PDFMinerLoader,
        PDFMinerPDFasHTMLLoader,
    )
from langchain.text_splitter import (
        CharacterTextSplitter,
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import AstraDB
from langchain.callbacks import get_openai_callback
from langchain.docstore.document import Document
import openai
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import requests
import markdown
from dotenv import load_dotenv
import os
import logging
import tiktoken
import time

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] =  openai_api_key
astradb_token_key = os.getenv("ASTRADB_TOKEN_KEY")
astradb_api_endpoint = os.getenv("ASTRADB_API_ENDPOINT")
astradb_collection_name = os.getenv("ASTRADB_COLLECTION_NAME")
url_webhook = os.getenv("URL_WEBHOOK")
url_error_webhook = os.getenv("URL_ERROR_WEBHOOK")

def tokens_embbeding(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def load_doc(pdf_path):
    print("Load pdf...")
    loader = PDFMinerPDFasHTMLLoader(pdf_path)
    data = loader.load()[0]
    print("Loader is done!...")
    return data

def parsing_pdf_html(pages):
    print("Parsing process...")
    soup = BeautifulSoup(pages.page_content, "html.parser")
    content = soup.find_all("div")

    cur_fs = None
    cur_text = ""
    snippets = []  # first collect all snippets that have the same font size
    for c in content:
        sp = c.find("span")
        if not sp:
            continue
        st = sp.get("style")
        if not st:
            continue
        fs = re.findall("font-size:(\d+)px", st)
        if not fs:
            continue
        fs = int(fs[0])
        if not cur_fs:
            cur_fs = fs
        if fs == cur_fs:
            cur_text += c.text
        else:
            snippets.append((cur_text, cur_fs))
            cur_fs = fs
            cur_text = c.text
    snippets.append((cur_text, cur_fs))
    print("parsing is done!...")
    return snippets

def to_markdown(snippet):
    print("Converting to Markdown...")
    df = pd.DataFrame(snippet, columns=["text", "font_size"])
    # set header condition
    size1 = df["font_size"] >= 30
    size2 = (df["font_size"] < 30) & (df["font_size"] >= 18)
    size3 = (df["font_size"] < 17) & (df["font_size"] >= 14)
    size4 = (df["font_size"] < 14) & (df["font_size"] >= 10)

    size_is = [size1, size2, size3, size4]
    mark_is = ["#", "##", "###", "####"]

    # set header and join text and header
    df["header"] = np.select(size_is, mark_is, default="")
    df["markdown"] = df["header"] + " " + df["text"]
    # one string
    mark = "".join(df["markdown"].tolist())
    print("Markdown is ready...")
    return mark


def remove_break_add_whitespace(mark):
    print("Retaining whitespace...")
    pattern = r"(\n)([a-z])"
    pattern2 = r"\n"

    replacement = r" \2"
    replacement2 = r"\n\n"

    replace_break = re.sub(pattern, replacement, mark)
    replace_break = re.sub(pattern2, replacement2, replace_break)
    print("WHite space is retain...")
    return replace_break


def text_to_markdown_head(string, filename):
    md = markdown.markdown(string)
    with open("/content/" + filename, "w", encoding="utf-8") as f:
        f.write(md)


def create_document_by_splitting(data):
    print("Chungking Process...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=25,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.create_documents([data])
    split = text_splitter.split_documents(docs)
    print("Chungking is done...")
    return docs


def extract_headers(document):
    print("extracting header...")
    docs = []
    h1 = None
    h2 = None
    h3 = None
    h4 = None

    for doc in document:
        page_content = doc.page_content
        lines = page_content.split("\n\n")
        temp_doc = Document(page_content=doc.page_content)

        for line in lines:
            if line.startswith("#"):
                header_level = line.count("#")
                header_text = line.lstrip("#").strip()

                if header_level == 1:
                    h1 = header_text
                    h2 = None
                    h3 = None
                    h4 = None
                    temp_doc.metadata = {
                        "header1": h1,
                        "header2": h2,
                        "header3": h3,
                        "header4": h4,
                    }
                    # print('detect H1')
                elif header_level == 2:
                    h2 = header_text
                    h3 = None
                    h4 = None
                    temp_doc.metadata = {
                        "header1": h1,
                        "header2": h2,
                        "header3": h3,
                        "header4": h4,
                    }
                    # print('detect H2')
                elif header_level == 3:
                    h3 = header_text
                    h4 = None
                    temp_doc.metadata = {
                        "header1": h1,
                        "header2": h2,
                        "header3": h3,
                        "header4": h4,
                    }
                    # print('detect H3')
                elif header_level == 4:
                    h4 = header_text
                    temp_doc.metadata = {
                        "header1": h1,
                        "header2": h2,
                        "header3": h3,
                        "header4": h4,
                    }
                    # print('detect H4')
            else:
                temp_doc.metadata = {
                    "header1": h1,
                    "header2": h2,
                    "header3": h3,
                    "header4": h4,
                }
        docs.append(temp_doc)
    print("Header is ready...")
    return docs


def callWebhook(url, course, course_document_id, doc_tokens):
    logging.basicConfig(level=logging.INFO,  # Set the logging level
                        format='%(asctime)s [%(levelname)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger(__name__)

    logger.info('Webhook called... Course ID: ', course, ' || Document ID: ', course_document_id, ' || Tokens Embbed: ', doc_tokens)

    payload = {'course_id': course, 'document_id': course_document_id, 'tokens_embbed': doc_tokens}  # Replace with your JSON payload

    response = requests.post(url, json=payload)
    logger.info('Webhook response: ', response,'Course ID: ', course, ' || Document ID: ', course_document_id, ' || Tokens Embbed: ', doc_tokens)

def callErrorWebhook(url_error, course, course_document_id, doc_tokens, error):
    logging.basicConfig(level=logging.INFO,  # Set the logging level
                        format='%(asctime)s [%(levelname)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger(__name__)

    logger.info('Webhook called... Course ID: ', course, ' || Document ID: ', course_document_id, ' || Tokens Embbed: ', doc_tokens)

    payload = {'course_id': course, 'document_id': course_document_id, 'tokens_embbed': doc_tokens, 'Error': error}  # Replace with your JSON payload

    response = requests.post(url_error, json=payload)
    logger.info('Webhook response: ', response,'Course ID: ', course, ' || Document ID: ', course_document_id, ' || Tokens Embbed: ', doc_tokens, ' || Error Status: ', error)


def Embbed_openaAI(chunks, course_id, course_document_id, doc_tokens):
    with get_openai_callback() as cb:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", max_retries=3, retry_min_seconds=30,
                                      retry_max_seconds=60, show_progress_bar=True)
        vstore = AstraDB(
            embedding=embeddings,
            collection_name=astradb_collection_name,
            api_endpoint=astradb_api_endpoint,
            token=astradb_token_key,
            )
        try:
            vstore.add_documents(chunks)
            print(cb)
            print("Calling Webhook for success uploading")
            return callWebhook(url_webhook, course_id, course_document_id, doc_tokens)
        except openai.RateLimitError as er:
            print("Fail rate limit error. Call error webhook")
            callErrorWebhook(url_error_webhook, course_id, course_document_id, doc_tokens, er.message)

def delete(course_id):
    from astrapy.db import AstraDB, AstraDBCollection
    astradb_token_key = os.getenv("ASTRADB_TOKEN_KEY")
    astradb_api_endpoint = os.getenv("ASTRADB_API_ENDPOINT")
    astradb_collection_name = os.getenv("ASTRADB_COLLECTION_NAME")

    astra_db = AstraDB(token=astradb_token_key,
                       api_endpoint=astradb_api_endpoint)
    collection = AstraDBCollection(
        collection_name=astradb_collection_name, astra_db=astra_db)

    col_response = collection.delete_many(filter={"metadata.source": course_id})
    print(col_response)
    if 'moreData' in col_response['status']:
        print("deleting again")
        delete(course_id)
    else:
        print("delete is done!")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    callRequest("https://www2.ed.gov/documents/ai-report/ai-report.pdf","65c46edc510f69a052a47119", "ai", "65c46f04510f69a052a47148")