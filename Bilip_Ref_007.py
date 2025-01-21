from langchain import hub
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate
)
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
import nltk
import os
import re
import time
from dotenv import load_dotenv
from langdetect import detect
import spacy
import tiktoken

from langchain.callbacks import get_openai_callback
from langchain.schema.messages import AIMessage, HumanMessage

from langchain.vectorstores import AstraDB

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
print(f'api: {openai_api_key}')

nltk.download("punkt")

def tokens_embbeding(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def tokens_llm(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def retry_with_openAI(
    func,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                print("delay for a minute")
                print(f"Error message: {e}")
                # Increment the delay
                delay = 60.0

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper

def setup():
    astradb_token_key = os.getenv("ASTRADB_TOKEN_KEY")
    astradb_api_endpoint = os.getenv("ASTRADB_API_ENDPOINT")
    astradb_collection_name = os.getenv("ASTRADB_COLLECTION_NAME")
    print(f'token_db: {astradb_token_key}')
    print(f'endpoint: {astradb_api_endpoint}')
    print(f'collection: {astradb_collection_name}')

    embeddings = OpenAIEmbeddings()

    vstore = AstraDB(
        embedding=embeddings,
        collection_name=astradb_collection_name,
        api_endpoint=astradb_api_endpoint,
        token=astradb_token_key,
    )
    return vstore


def calculate_similarity(question, document):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([question, document])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity[0, 0]


def check_similarity(answer, chunks):
    nlp = spacy.load("en_core_web_sm")
    question_tokens = nlp(answer)
    question2_tokens = nlp(chunks)
    similarity_score = calculate_similarity(question_tokens.text, question2_tokens.text)

    # Set a threshold for similarity
    threshold = 0.23

    # Check if the similarity score is above the threshold
    if similarity_score > threshold:
        print("Similarity score: ", similarity_score)
        return True
    else:
        print("Similarity score: ", similarity_score)
        return False


def historyTomemory(chat_history, memory):
    print('Into historyTomemory process...')
    input = []
    output = []
    header_ref_array = []
    for i in range(len(chat_history)):
        print('CHAT HISTORY')
        if chat_history[i]['type'] == 'human':
            input.append(chat_history[i]['content'])
        else:
            output.append(chat_history[i]['content'])
            header_ref_array.append(chat_history[i]['header_ref'])
    print('HEADER_REF_ARRAY INPUT: ', header_ref_array)
    for idx in range(len(input)):
        memory.save_context({"input": input[idx]}, {"output": output[idx]})
    return header_ref_array


def ragChain(vstore, source, chat_history):
    print("RAG Chain creating...")
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", temperature=0.3, max_tokens=512
    )
    retriever = vstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 2, "filter": {"course_id": source}}
    )
    memory = ConversationBufferMemory(return_messages=True)
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
    #add history to the memory
    header_ref_array = []
    if chat_history == []:
        print('no history')
    else:
        header_ref_array = historyTomemory(chat_history, memory)

    condense_q_system_prompt = """
  You are an expert of the document. Generate answer only based on the given context. Do not make up the answer.
    """
    condense_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    condense_q_chain = condense_q_prompt | llm | StrOutputParser()

    qa_system_prompt = """
  You are an expert of the document. Generate answer only based on the given context. Do not make up the answer.
  The context is {context} and the question is {question}
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    def format_docs(docs):
        global retrieved_docs
        retrieved_docs = []
        for doc in docs:
           retrieved_docs.append(doc)
        return "\n\n".join(doc.page_content for doc in docs)

    def condense_question(input: dict):
        if input.get("chat_history"):
            return condense_q_chain
        else:
            return input["question"]

    rag_chain = (
        RunnablePassthrough.assign(
            context=condense_question | retriever_from_llm | format_docs
        ) 
        | qa_prompt
        | llm
    )
    return rag_chain, memory, header_ref_array


def topic_creation(chat_history):
  
  temp_hist = []
  print(f'Into topic_creation : {chat_history}')
  for hist in chat_history:
    if hist.type == "human":
        list_hist = {"type": "human", "content": hist.content}
    elif hist.type == "ai":
        list_hist = {"type": "ai", "content": hist.content}
    temp_hist.append(list_hist)

  conv = temp_hist[0]['content']

  def detect_language_langdetect(text):
    try:
        detected_language = detect(text)
        return detected_language
    except Exception as e:
        print("An error occurred:", e)
        return "en"
    
  result = detect_language_langdetect(conv)
  custom_template = """
    Create a title header from the text below. use the laguage code {lang}.
    {question}
    """

  CUSTOM_QUESTION_PROMPT = PromptTemplate(input_variables=["lang", "question"], template=custom_template)
  llm_topic = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0.5, max_tokens=50
        )
  print(f"lang: {result}")
  print(f'question: {conv}')
  llmchain = LLMChain(llm=llm_topic, prompt=CUSTOM_QUESTION_PROMPT)
  with get_openai_callback() as cost_topic:
    output = llmchain.invoke({"lang":result, "question": conv})
    output = re.sub(r'Title:\s*', '', output['text'])
    output = re.sub(r'Title Header:\s*', '', output)
    output = re.sub(r'\n\n', ' ', output)
    output_token_cb = cost_topic.completion_tokens
    input_token_cb = cost_topic.prompt_tokens
    print("Topic Usage")
    print(f'cb in: {input_token_cb}')
    print(f'cb out: {output_token_cb}')
  return output, output_token_cb, input_token_cb


def ask_with_memory(question, source, chat_history=[], topics=''):
    # setup()
    import time
    vstore = setup()
    custom_template = """
    This is conversation with a human. Answer the questions you get based on the knowledge you have.
    If you don't know the answer, just say that you don't, don't try to make up an answer.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    """
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

    rag_chain, memory, header_ref_array = ragChain(vstore, source, chat_history)
    message = ''
    header_ref = ''
    # chat history is memory from same with database request
    print(f"inside memory: {memory.buffer}")
    if chat_history != []:
        chat_history = memory.buffer
    else:
        chat_history = memory.buffer
    print(f"current history: {chat_history}")
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", temperature=0.5, max_tokens=512
    )
    retriever = vstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3, "filter": {"source": source}}
    )
    with get_openai_callback() as cost_ask:
        # Answering question with rag_chain
        start_time = time.time()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever,
            condense_question_prompt=CUSTOM_QUESTION_PROMPT,
            memory=memory,
        )
        # ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
        result = qa({"question": question})
        print(result['answer'])
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"TIME TO INVOKE: {elapsed_time} seconds")
        tokens_embbed = 30
        print(f"ai answer: {result['answer']}")
        chunks_content = ''
        source_documents = retriever.get_relevant_documents(question)
        print('SOURCE DOCUMENTS: ', source_documents[0].page_content)
        retrieved_docs = source_documents
        retrieved_docs_vector_checked = []
        start_time_check_similarity = time.time()
        # chek similarity from chunk and answer for relevant chun above 0.3 similarity
        for i in range(len(retrieved_docs)):
            if check_similarity(result['answer'], retrieved_docs[i].page_content):
                chunks_content += retrieved_docs[i].page_content
                retrieved_docs_vector_checked.extend(retrieved_docs[i])
        if (check_similarity(result['answer'], chunks_content)):
            message = result['answer']
            # Adding Header, chat_history,
            for x in range(len(retrieved_docs)):
              if (check_similarity(result['answer'], retrieved_docs[x].page_content)):
                header_ref = f"{header_ref} - {retrieved_docs[x].metadata['document_name']}"
                if ('header1' in retrieved_docs[x].metadata) & (retrieved_docs[x].metadata['header1'] != None):
                    header_ref = f"{header_ref} > {retrieved_docs[x].metadata['header1']}"
                if ('header2' in retrieved_docs[x].metadata) & (retrieved_docs[x].metadata['header2'] != None):
                    header_ref = f"{header_ref} > {retrieved_docs[x].metadata['header2']}"
                if ('header3' in retrieved_docs[x].metadata) & (retrieved_docs[x].metadata['header3'] != None):
                    header_ref = f"{header_ref} > {retrieved_docs[x].metadata['header3']}"
                if ('header4' in retrieved_docs[x].metadata) & (retrieved_docs[x].metadata['header4'] != None):
                    header_ref = f"{header_ref} > {retrieved_docs[x].metadata['header4']}"
                header_ref += "\n"
            # print(header_ref)
            end_time_check_similarity = time.time()
            elapsed_time_check_similarity = end_time_check_similarity - start_time_check_similarity
            print(f"Elapsed Time Check Similarity: {elapsed_time_check_similarity} seconds")
            chat_history.extend([HumanMessage(content=question), AIMessage(content=result['answer'])])
            header_ref_array.append(header_ref)
            print('HEADER REF ARRAY: ', header_ref_array)
            # message += "\nYou can see more detail explanation in the document at:\n" + header_ref + "\n"
            print(f"{'REFERENCE: ', chunks_content}")
            print('')
            print('-' * 250)
            print(message)

            # Check if topic already exist and create summary as Topic
            if (topics == ''):
                print(f"no topics")
                topics, topic_out, topic_in = topic_creation(chat_history)
                print(f'top in: {topic_in}')
                print(f'top out: {topic_out}')

        else:
            print(f"{'REFERENCE: ', chunks_content}")
            print(f"Sorry but I have no knowledge to your question")
            message = "Sorry but I have no knowledge to your question"
            chat_history.extend([HumanMessage(content=question), AIMessage(content=result['answer'])])
            header_ref_array.append('')
            # Check if topic already exist and create summary as Topic
            if (topics == ''):
                print(f"no topics")
                temp_history = [HumanMessage(content=question), AIMessage(content=message)]
                topics, topic_out, topic_in = topic_creation(temp_history)
    output_token_cb = cost_ask.completion_tokens
    input_token_cb = cost_ask.prompt_tokens
    print(f'cb in: {input_token_cb}')
    print(f'cb out: {output_token_cb}')
    print(f'request: {cost_ask.successful_requests}')
    print(f'insert-an: ')
    print([HumanMessage(content=question), AIMessage(content=result['answer'])])
    # Add current question and answer into chat history
    # *******************RESPONSE
    print(f"message: {message}")
    print(f"history: {chat_history}")
    print(f"topics: {topics}")
    print(f"tokens_embbed: {tokens_embbed}")
    # *******************RESPONSE
    print(f"tokens usage: {cost_ask.total_tokens}")

    memory.clear()
    return message, chat_history, topics, output_token_cb, input_token_cb, tokens_embbed, header_ref_array


def main():
    chat_history = []
    topic = ""
    ask_with_memory("what is this document about?", '65bcdc83def141632c3fba7d', chat_history, topic)
    # ask_with_memory('tell me more about it', 'emarketing_textbook_download_chunk_separator', chat_history, topic)


if __name__ == "__main__":
    main()

#     """{
#     "question": "tell me about fact in software",
#     "source": "777",
#     "chat_history":[
#         {
#             "content": "what this document about?",
#             "type": "human"
#         },
#         {
#             "content": "The document is about a book that presents the Python language in a linear fashion. It provides an overview of Python and its major language features, such as types and operations. The book also includes exercises, quizzes, and summaries to help readers review and test their understanding of the material. The document mentions that the third edition of the book reflects the changes in Python 2.5 and incorporates structural changes.",
#             "type": "ai"
#         }
#     ],
#     "topic": "This document is written by Robert L. Glass"
# }"""

