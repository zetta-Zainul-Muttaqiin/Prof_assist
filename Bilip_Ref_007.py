from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate
)
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.chat_models.openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from flask import Flask,g
import nltk
from nltk.tokenize import word_tokenize
import os
import re
import time
from dotenv import load_dotenv
import spacy
from langchain.callbacks.manager import get_openai_callback
from langchain.schema.messages import AIMessage, HumanMessage
from lingua import Language, LanguageDetectorBuilder
from setup import vstore

load_dotenv()
app_akadbot = Flask(__name__)


with app_akadbot.app_context():
    def set_retrieved_docs_akadbot(docs):
        g.retrieved_docs_akadbot = docs
        
    def get_retrieved_docs_akadbot():
        return g.retrieved_docs_akadbot

# ********* set token for openai and datastax
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

# nltk.download("punkt")
nlp = spacy.load("en_core_web_lg")
greetings_en = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'hi, how is it going?', 'greetings!', 'how are you doing?', 'how do you do?', 'what`s up?']
greetings_fr = ['bonjour', 'salut', 'coucou', 'bonsoir', 'bonjour à tous', 'comment allez-vous ce matin ?', 'bonne journée', 'bonne soirée', 'bonne nuit', 'À bientôt', 'À plus tard', 'À tout à lheure', 'À demain', 'Ça va?', 'enchanté']
language_data = [Language.ENGLISH, Language.FRENCH]
detector = LanguageDetectorBuilder.from_languages(*language_data).build()
llm = ChatOpenAI(
        model="gpt-4o-mini", temperature=0
    )
llm_topic = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.5, max_tokens=50
        )

def check_similarity(answer, chunks):
    answer_tokens = nlp(answer)
    chunks_tokens = nlp(chunks)
    similarity_score = answer_tokens.similarity(chunks_tokens)

    # ********* Set a threshold for similarity
    threshold = 0.75

    # ********* Check if the similarity score is above the threshold
    if similarity_score > threshold:
        print("Similarity score: ", similarity_score)
        return True
    else:
        print("Similarity score: ", similarity_score)
        return False

def historyTomemory(chat_history, add_memory):
    print('Into historyTomemory process...')
    input = []
    output = []
    header_ref_array = []
    store_chat_history = []
    index_chat = 0

    # ********* convert chat_history object from BE
    # ********* looping until all chat_histoy index is done, from 0 to same length with chat_histoy
    while index_chat < len(chat_history)-1:
        
        # ********* Checking every first chat_history is human_message and the next chat is ai_message 
        if chat_history[index_chat]['type'] == 'human':
            input.append(chat_history[index_chat]['content'])

            # ********* Check the next chat is ai_message
            if chat_history[index_chat+1]['type'] == 'ai':
                output.append(chat_history[index_chat+1]['content'])
                header_ref_array.append(chat_history[index_chat+1]['header_ref'])

            # ********* if after human_message is not ai_message will add empty ai_message 
            else:
                output.append('')
                header_ref_array.append('')
        index_chat += 1
    print('HEADER_REF_ARRAY INPUT: ', header_ref_array)
    
    # ********* inject human_message and ai_message from chat_history to our engine memory
    for index_input in range(len(input)):
        add_memory.add_user_message(input[index_input])
        add_memory.add_ai_message(output[index_input])
        store_chat_history.extend([HumanMessage(content=input[index_input]), AIMessage(content=output[index_input])])
    return header_ref_array, add_memory, store_chat_history


def detect_language_langdetect(text):
    # ********* detect language used from a sentence use langdetect
    try:
        detected_language = detector.detect_language_of(text).name.lower()
        return detected_language
    # ********* set default to english if langdetection is error
    except Exception as error_lang:
        print("An error occurred:", error_lang)
        return "english"
    
def detect_greetings(text):
    # ********* detect question input
    lang = detect_language_langdetect(text)
    print(f'lang: {lang}')
    # ********* word_tokenize into english if question is french
    if lang == 'french':
        # ********* tokenizing question input
        words = word_tokenize(text.lower(), language=lang)
        greetings = greetings_fr
    # ********* default to english
    else:
        words = word_tokenize(text.lower())
        greetings = greetings_en
    # ********* check each word
    for word in words:
        # ********* True if find greetings word from question input
        if word in greetings:
            return True
    return False

def topic_creation(chat_history):
  # ********* create topic based on first question
  start_time = time.time()
  temporary_hist = []
  print(f'Into topic_creation : {chat_history}')
  
  # ********* loop each chat_history 
  for history in chat_history:

    # ********* get question that human chat
    if history.type == "human":
        list_history = {"type": "human", "content": history.content}
    
    # ********* get answer that ai chat
    elif history.type == "ai":
        list_history = {"type": "ai", "content": history.content}
    temporary_hist.append(list_history)
  
  # ********* get first question
  print(f"temp_hist: {temporary_hist}")
  question = temporary_hist[0]['content']
  
  # ********* detect language used from first question
  lang_topic = detect_language_langdetect(question)
  
  # ********* set prompt so engine can understand question and lang when create topic
  if lang_topic == 'french':
    custom_template = """
        Créer un en-tête de titre à partir du texte ci-dessous. Ne pas inventer la réponse. Créer en langue française.
        {question}
        """
  else:
    custom_template = """
        Create a title header from the text below. Do not make up the answer. Create in English Language.
        {question}
        """
  CUSTOM_QUESTION_PROMPT = PromptTemplate(input_variables=["question"], template=custom_template)

  print(f"lang: {lang_topic}")
  print(f'question: {question}')
  llmchain = LLMChain(llm=llm_topic, prompt=CUSTOM_QUESTION_PROMPT)
  
  # ********* invoke topic
  with get_openai_callback() as cost_topic:
    output = llmchain.invoke({"question": question})

    # ********* clean ai answer
    output = re.sub(r'Title:\s*', '', output['text'])
    output = re.sub(r'Title Header:\s*', '', output)
    output = re.sub(r'\n\n', ' ', output)
    output_token_cb = cost_topic.completion_tokens
    input_token_cb = cost_topic.prompt_tokens
    print("Topic Usage")
    print(f'cb in: {input_token_cb}')
    print(f'cb out: {output_token_cb}')

  # ********* calculate time for topic created
  end_time = time.time()
  print(f'topic() TIME usage: {end_time-start_time} seconds')
  return output

def ragChain(source, chat_history):
    retriever = vstore.as_retriever(

        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6, "filter": {"source": source}}
    )
    memory = ConversationBufferWindowMemory(k=1, return_messages=True)

    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
    
    
    # ********* add history to the memory
    add_memory = ChatMessageHistory()
    header_ref_array = []
    if chat_history == []:
        print('no history')
    else:
        print('is history')
        # ********* set current memory and get header reference from history
        header_ref_array, add_memory, chat_history = historyTomemory(chat_history, add_memory)
    
    memory = ConversationBufferWindowMemory(k=1, chat_memory=add_memory, return_messages=True)
    # ********* set system prompt for bot can understanding the main job
    condense_q_system_prompt = """
  You are an expert of the document. Generate answer only based on the given '''context'''. Do not make up the answer.
    """
    condense_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    condense_q_chain = condense_q_prompt | llm | StrOutputParser()

    # ********* set QA chain prompt for bot can understand context
    qa_system_prompt = """
  You are an expert of the document. Generate answer only based on the given context. Do not make up the answer.
  The context is '''{context}''' and the question is '''{question}'''
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    # ********* looping all chunks to one context
    def format_docs(docs):
        retrieved_docs = []
        for doc in docs:
            retrieved_docs.append(doc)
        set_retrieved_docs_akadbot(retrieved_docs)
        return "\n\n".join(doc.page_content for doc in docs)

    # ********* get question only or get question with history context
    def condense_question(input: dict):
        if input.get("chat_history"):
            return condense_q_chain
        else:
            return input["question"]
    # ********* build chain with retriever context, the context it self, prompt and llm
    rag_chain = (
        RunnablePassthrough.assign(
            context=condense_question | retriever_from_llm | format_docs
        )
        | qa_prompt
        | llm
    )
    return rag_chain, memory, header_ref_array, chat_history

def ask_with_memory(question, source, chat_history=[], topics=''):

    rag_chain, memory, header_ref_array, chat_history = ragChain(source, chat_history)
    message = ''
    header_ref = ''

    with get_openai_callback() as cb:
        print(f"memory: {memory.chat_memory.messages}")
        # ********* get langauge used from question
        lang_question = detect_language_langdetect(question)
        print(f'input lang used: {lang_question}')
        start_time = time.time()
        if detect_greetings(question.lower()):
            # ********* invoke answer for greetings
            ai_msg = llm.invoke(question)
            # ********* save as message response
            message = ai_msg.content

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"TIME TO INVOKE: {elapsed_time} seconds")
            # ********* Add current question and answer into chat history, also header reference but null
            chat_history.extend([HumanMessage(content=question), ai_msg])
            header_ref_array.append('')
        else:
            # ********* Answering question with rag_chain
            ai_msg = rag_chain.invoke({"question": question, "chat_history": memory.chat_memory.messages})
            message = ai_msg.content
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"TIME TO INVOKE: {elapsed_time} seconds")
            retrieved_docs = get_retrieved_docs_akadbot()
            print(f"chunks: {len(retrieved_docs)}")
            # chek similarity from chunk and answer for relevant chun above 0.3 similarity
            start_time_check_similarity = time.time()
            for index_docs in range(len(retrieved_docs)):
                    # ********* compare ansewr and chunks context distance similarity, if similarity > 0.23, add header
                    if (check_similarity(message, retrieved_docs[index_docs].page_content)):

                        # ********* set first reference is document name
                        header_ref = f"{header_ref} - {retrieved_docs[index_docs].metadata['document_name']}"
                        # ********* if there is header1 on metadata and set as next reference
                        if ('header1' in retrieved_docs[index_docs].metadata) & (retrieved_docs[index_docs].metadata['header1'] != None):
                            header_ref = f"{header_ref} > {retrieved_docs[index_docs].metadata['header1']}"
                        # ********* if there is header2 on metadata and set as next reference
                        if ('header2' in retrieved_docs[index_docs].metadata) & (retrieved_docs[index_docs].metadata['header2'] != None):
                            header_ref = f"{header_ref} > {retrieved_docs[index_docs].metadata['header2']}"
                        # ********* if there is header3 on metadata and set as next reference
                        if ('header3' in retrieved_docs[index_docs].metadata) & (retrieved_docs[index_docs].metadata['header3'] != None):
                            header_ref = f"{header_ref} > {retrieved_docs[index_docs].metadata['header3']}"
                        # ********* if there is header4 on metadata and set as next reference
                        if ('header4' in retrieved_docs[index_docs].metadata) & (retrieved_docs[index_docs].metadata['header4'] != None):
                            header_ref = f"{header_ref} > {retrieved_docs[index_docs].metadata['header4']}"
                        header_ref += "\n"
                        print(f"header chuhks :{header_ref}")

                        # ********* calculate time for cosine_similarity check
                        end_time_check_similarity = time.time()
                        elapsed_time_check_similarity = end_time_check_similarity - start_time_check_similarity
                        print(f"Elapsed Time Check Similarity: {elapsed_time_check_similarity} seconds")
                
            # ********* add all header_reference and removing the duplicated reference
            unique_lines = set(header_ref.split('\n'))
            header_ref = '\n'.join(unique_lines)
            header_ref_array.append(header_ref)
            # ********* end of add all header_reference and removing the duplicated reference
            print('HEADER REF ARRAY: ', header_ref_array)
            print('')
            print('-' * 250)
                # ********* Add current question and answer into chat history, also header reference but null
            chat_history.extend([HumanMessage(content=question), AIMessage(content=message)])

                # Chek if topic already exist and create summary as Topic
        print(f'history: {chat_history}')
        if (topics == ''):
            print(f"no topics")
            topics = topic_creation(chat_history)
        print(cb)

    print(f"ai answ : {ai_msg}")
    print(f"history: {chat_history}")
    print(f"topics: {topics}")
    tokens_out = cb.completion_tokens
    tokens_in = cb.prompt_tokens
    tokens_embbed = 50

    memory.clear()
    return message, chat_history, topics, tokens_out, tokens_in, tokens_embbed, header_ref_array

def main():
    chat_history = []
    topic = ""
    ask_with_memory("what is customer service?", '001', chat_history, topic)
    

if __name__ == "__main__":
    main()