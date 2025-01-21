from langchain.vectorstores import AstraDB
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.prompts.chat import (
    ChatPromptTemplate
)
from langchain.callbacks import get_openai_callback
from langchain.schema.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import os
import openai
import requests
import time
import tiktoken
import logging
import re

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
print(f'api: {openai_api_key}')

astradb_token_key = os.getenv("ASTRADB_TOKEN_KEY")
astradb_api_endpoint = os.getenv("ASTRADB_API_ENDPOINT")
astradb_collection_name = os.getenv("ASTRADB_COLLECTION_NAME")


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
    # logging.basicConfig(level=logging.INFO,  # Set the logging level
    #                     format='%(asctime)s [%(levelname)s] - %(message)s',
    #                     datefmt='%Y-%m-%d %H:%M:%S')

    # logger = logging.getLogger(__name__)

    # logger.info('AstraDB Token Key: ', astradb_token_key)
    # logger.info('AstraDB API Endpoint: ', astradb_api_endpoint)
    # logger.info('AstraDB Collection Name: ', astradb_collection_name)
    embeddings = OpenAIEmbeddings()
    vstore = AstraDB(
        embedding=embeddings,
        collection_name=astradb_collection_name,
        api_endpoint=astradb_api_endpoint,
        token=astradb_token_key,
    )
    return vstore


def get_prompt(quiz_type, option_amount, lang):
    part_prompt = """"""
    final_prompt = """"""
    if (quiz_type == 'multiple'):
        options = ['A) ...', '   B) ...', '   C) ...', '   D) ...', '   E) ...']
        for i in range(option_amount):
            part_prompt += options[i] + '\n'
        final_prompt = f"""
   You are an expert of the document to create exam for student. Avoid 'all of the above' answer.
   When creating multiple choices question, set the {option_amount} choices in bullet points with only ONE right answer and put the right answer below it with explanation. Here is the format you MUST follow:

   Question: What is the purpose of the document "General Standardization Development Guideline"?
   Options:
   {part_prompt}
   Answer: C) ...
   Explanation: ...
  """
    elif (quiz_type == 'essay_long' and lang=="en"):
        final_prompt = """
  You are an expert of the document to create essay questions with long answer for students. Create the question and then the answer below. Here is the format you MUST follow:

  Question: ...
  Answer: ...
  """

    elif (quiz_type == 'essay_long' and lang == "fr"):
        final_prompt = """
    Vous ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Âªtes un expert du document pour crÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©er des questions d'essai avec rÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©ponse longue pour les ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©tudiants. CrÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©ez la question puis la rÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©ponse ci-dessous. Voici le format que vous DEVEZ suivre:

    Question: ...
    Answer: ...
    """

    elif (quiz_type == 'essay_short' and lang == "en"):
        final_prompt = """
    You are an expert of the document to create essay questions with short answer for students. Create the question and then the answer below. Here is the format you MUST follow:

    Question: ...
    Answer: ...
    """

    elif (quiz_type == 'essay_short' and lang=="fr"):
        final_prompt = """
  Vous ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Âªtes un expert du document pour crÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©er des questions d'essai avec rÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©ponse courte pour les ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©tudiants.CrÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©ez la question puis la rÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©ponse ci-dessous. Voici le format que vous DEVEZ suivre:

  Question: ...
  Answer: ...
  """
    return (final_prompt)


def remove_duplicate_quizzes(quizzes):
    seen_questions = set()
    unique_quizzes = []

    for quiz in quizzes:
        question = quiz.get("question", "")

        # Check if the question is not duplicated
        if question not in seen_questions:
            seen_questions.add(question)
            unique_quizzes.append(quiz)

    return unique_quizzes


def quiz_extract(quiz_text, quiz_type, lang):
    quiz_pattern = re.compile(r"Quiz (\d+):(.*?)Options:(.*?)(?:Answer: (.*?)\s*)?Explanation:(.*?)\s*(?=Quiz|$)",
                              re.DOTALL)
    if (quiz_type == 'multiple'):
        quiz_pattern = re.compile(
            r"Question(?::? (\d+))?: (.*?)(?:\nOptions:\n(.*?))?\n(?:Answer|R[eÃƒÆ’Ã‚Â©]ponse): (.*?)\n(?:Explanation|Explication): (.*?)(?=\n\n|$)",
            re.DOTALL)
    elif (quiz_type == 'essay_long' or quiz_type == 'essay_short'):
        quiz_pattern = re.compile(
            r"Question(?:\s*(\d+))?:\s*(.*?)(?=\n(?:Answer|R[eÃƒÆ’Ã‚Â©]ponse)(?:\s*(\d+))?:|$)\s*(?:Answer|R[eÃƒÆ’Ã‚Â©]ponse)(?:\s*(\d+))?:\s*(.*?)(?=\n(?:Question|R[ÃƒÆ’Ã‚Â©e]ponse)(?:\s*(\d+))?:|$)",
            re.DOTALL)

    # Find all matches in the text
    matches = quiz_pattern.findall(quiz_text)
    # Extracted quiz data
    quizzes = []
    if (quiz_type == 'multiple'):
        for match in matches:
            question_number, question, options, correct_answer, explanation = match
            question_text = question if not question_number else question.lstrip('0123456789.: ')
            options = [opt.strip() for opt in options.split('\n') if opt.strip()] if options else []

            quiz_data = {
                "question": question_text.strip(),
                "options": options,
                "correct_answer": correct_answer.strip(),
                "explanation": explanation.strip() if explanation else None
            }

            quizzes.append(quiz_data)
    elif (quiz_type == 'essay_long' or quiz_type == 'essay_short'):
        quizzes = [{"question": question.strip(), "answer": answer.strip()} for
                   question_number, question, _, answer_number, answer, _ in matches]
        # quizzes.append(quiz_data)
    return quizzes

def getChunksData(document_id, amount_of_quiz):
 print('ENTERING getChunksData')
 from astrapy.db import AstraDB, AstraDBCollection
 import random

 array_header = set()
 astra_db = AstraDB(token=astradb_token_key,
                   api_endpoint=astradb_api_endpoint)
 collection = astra_db.collection(collection_name=astradb_collection_name)
 print('ENTER HERE')
 for i in range(len(document_id)):
   generator = collection.paginated_find(
    filter={"metadata.document_id": document_id[i]},
    options={"limit": amount_of_quiz*4 + 8}
)
   print(generator)
   for doc in generator:
     if 'metadata' in doc:
        for i in range(1, 6):
            header_key = f'header{i}'
            if header_key in doc['metadata']:
                header_value = doc['metadata'][header_key]
                if header_value is not None and any(c.isdigit() for c in header_value) == False and 5 <= len(header_value) <= 40:
                    header_value = header_value.replace('\t', ' ').replace('\n', ' ')
                    array_header.add(header_value)

 shuffled_array_header = list(array_header)
 random.shuffle(shuffled_array_header)
 print('DATA LENGTH: ', len(shuffled_array_header))
 return(shuffled_array_header)

def detect_and_create_quizzes(quiz_id, text, source, quiz_type, multiple_option_amount, document_id, lang, quiz_generated=[]):
    tokens_in = 0
    tokens_out = 0
    tokens_embbed = 0
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1, max_tokens=512, model_kwargs={"top_p": 0.5})
    vstore = setup()
    retriever = vstore.as_retriever(search_type='similarity', search_kwargs={"k": 10, 'filter': {'course_id': source,
                                                                                                 'course_document_id':
                                                                                                     document_id[0]}})
    memory = ConversationBufferMemory(return_messages=True)
    condense_q_system_prompt = f"""
        {get_prompt(quiz_type, multiple_option_amount, lang)}
      """
    print(condense_q_system_prompt)
    tokens_in = tokens_llm(condense_q_system_prompt)
    condense_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    condense_q_chain = condense_q_prompt | llm | StrOutputParser()
    context_for_prompt = """{context}"""
    qa_system_prompt = f"""
       {get_prompt(quiz_type, multiple_option_amount, lang)}
       {context_for_prompt}
      """
    print(qa_system_prompt)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    tokens_in += tokens_llm(qa_system_prompt)

    def format_docs(docs):
        global tokens_docs
        tokens_docs = 0
        for doc in docs:
            tokens_docs += tokens_llm(doc.page_content)
        return "\n\n".join(doc.page_content for doc in docs)

    def condense_question(input: dict):
        if input.get("chat_history"):
            return condense_q_chain
        else:
            return input["question"]

    rag_chain = (
            RunnablePassthrough.assign(context=condense_question | retriever | format_docs)
            | qa_prompt
            | llm
    )
    chat_history = []
    content_iteration = 0
    quiz_removed_duplicate = quiz_generated
    number_of_quiz_per_iteration = 4
    keywords = ['generate', 'create', 'quiz', 'question', 'crÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©er']
    number_match = re.search(r'\b\d+\b', text)
    content = getChunksData(document_id, int(number_match.group()))

    if any(keyword in text.lower() for keyword in keywords) and number_match:
        original_number = int(number_match.group())
        fix_original_number = original_number
        print(f"Original Number: {original_number}")
        isFirst = True
        quiz_text = ''
        while original_number > 0:
            if isFirst:
                if(lang=="en"):
                    question = re.sub(r'\b\d+\b', str(min(original_number, number_of_quiz_per_iteration)),
                                  f"{text} about {content[content_iteration]}. Create in English language.")
                elif (lang == "fr"):
                    question = re.sub(r'\b\d+\b', str(min(original_number, number_of_quiz_per_iteration)),
                                      f"{text} about {content[content_iteration]}. Create in French language.")
                content_iteration += 1
                print('QUESTION: ', question)
                ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
                # ******************************count
                tokens_out += tokens_llm(ai_msg.content)
                tokens_in += tokens_llm(question)
                print(f'first generate usage: {tokens_out}')
                # ******************************
                quiz_text += ai_msg.content
                #  chat_history.extend([HumanMessage(content=question), ai_msg])
                isFirst = False
            else:
                token_quiz_next = 0
                if(lang=="en"):
                    question = f"Create {min(original_number, number_of_quiz_per_iteration)} questions about {content[content_iteration]}. Create in English language."
                elif(lang=="fr"):
                    question = f"Create {min(original_number, number_of_quiz_per_iteration)} questions about {content[content_iteration]}. Create in French language."
                content_iteration += 1
                print('QUESTION: ', question)
                ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
                # ****************************count
                token_quiz_next = tokens_llm(ai_msg.content)
                tokens_in += tokens_llm(question)
                print(f'next generate usage: {token_quiz_next}')
                tokens_out += token_quiz_next
                print(f'current usage: {tokens_out}')
                # ****************************
                quiz_text += ai_msg.content
            #  chat_history.extend([HumanMessage(content=question), ai_msg])
            original_number -= number_of_quiz_per_iteration
        print(quiz_text)
        # print('-' * 250)
        # print(len(quiz_result))
        # print(quiz_result)
        quiz_result = quiz_extract(quiz_text, quiz_type, lang)
        quiz_removed_duplicate = remove_duplicate_quizzes(quiz_result)
        while (len(quiz_removed_duplicate) < fix_original_number):
            print(len(quiz_removed_duplicate))
            token_quiz_dup = 0
            if(lang=="en"):
              question = f"Create {(fix_original_number - len(quiz_removed_duplicate))} questions about {content[content_iteration]}. Create in English language."
            elif(lang=="fr"):
              question = f"Create {(fix_original_number - len(quiz_removed_duplicate))} questions about {content[content_iteration]}. Create in French language."

            content_iteration += 1
            print(question)
            ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
            print(ai_msg.content)
            # ***************************count
            token_quiz_dup = tokens_llm(ai_msg.content)
            tokens_in += tokens_llm(question)
            print(f'change dup generate usage: {token_quiz_dup}')
            tokens_out += token_quiz_dup
            print(f'current usage: {tokens_out}')
            # ***************************
            more_quiz_result = quiz_extract(ai_msg.content, quiz_type, lang)
            quiz_removed_duplicate.extend(more_quiz_result)
            quiz_removed_duplicate = remove_duplicate_quizzes(quiz_removed_duplicate)
            # chat_history.extend([HumanMessage(content=question), ai_msg])

        url_webhook = os.getenv("URL_WEBHOOK_QUIZ_BUILDER")
        callWebhook(quiz_id, url_webhook, quiz_result, tokens_in, tokens_out, tokens_embbed)
        return quiz_removed_duplicate, tokens_in, tokens_out, tokens_embbed
    else:
        return False


@retry_with_openAI
def quiz_builder(quiz_id, number_of_quiz, source, type_of_quiz, multiple_option_amount, document_id, lang):
    # rag_chain()
    quiz_word = ''
    if (type_of_quiz == 'multiple'):
        quiz_word = 'quizzes'
    else:
        quiz_word = 'questions'

    text = f"Create {number_of_quiz} {quiz_word}"
    string_quiz, tokens_in, tokens_out, tokens_embbed = detect_and_create_quizzes(quiz_id, text, source, type_of_quiz, multiple_option_amount, document_id, lang)
    print('quiz result: ', string_quiz)
    print(f"tokens_docs: {tokens_docs}")
    tokens_in += tokens_docs
    print(f"tokens_in: {tokens_in}")
    print(f"tokens_out: {tokens_out}")
    print(f"tokens usage: {tokens_in + tokens_out}")
    return string_quiz, tokens_in, tokens_out, tokens_embbed


@retry_with_openAI
def quiz_editor(source, type_of_quiz, multiple_option_amount, quiz_generated, document_id, lang):
    tokens_refresh_out = 0
    tokens_refresh_in = 0
    tokens_refresh_embbed = 0
    content = getChunksData(document_id, 10)
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1, max_tokens=512)
    vstore = setup()
    retriever = vstore.as_retriever(search_type='similarity', search_kwargs={"k": 10, 'filter': {'course_id': source,
                                                                                                 'course_document_id':
                                                                                                     document_id[0]}})
    condense_q_system_prompt = f"""
    {get_prompt(type_of_quiz, multiple_option_amount, lang)}
  """
    print(condense_q_system_prompt)
    condense_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    tokens_refresh_in = tokens_llm(condense_q_system_prompt)
    condense_q_chain = condense_q_prompt | llm | StrOutputParser()
    context_for_prompt = """{context}"""
    qa_system_prompt = f"""
   {get_prompt(type_of_quiz, multiple_option_amount, lang)}
 {context_for_prompt}
  """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    tokens_refresh_in += tokens_llm(qa_system_prompt)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def condense_question(input: dict):
        if input.get("chat_history"):
            return condense_q_chain
        else:
            return input["question"]

    rag_chain = (
            RunnablePassthrough.assign(context=condense_question | retriever | format_docs)
            | qa_prompt
            | llm
    )
    i=0
    chat_history = []
    quiz_removed_duplicate = []
    quiz_removed_duplicate.extend(quiz_generated)
    with get_openai_callback() as cb_refresh:
        while (len(quiz_removed_duplicate) < len(quiz_generated) + 2):

            if(lang=='en'):
             question = f"create 2 questions about {content[i]} from the document. Create in English language."
            elif(lang=='fr'):
             question = f"create 2 questions about {content[i]} from the document. Create in French language."

            ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
            # **********************count
            retrieved_docs = retriever.get_relevant_documents(question)
            for i in range(len(retrieved_docs)):
                tokens_refresh_in += tokens_llm(retrieved_docs[i].page_content)
            tokens_refresh_out += tokens_llm(ai_msg.content)
            print("CALLBACK: ", cb_refresh)
            print(f'refresh generate usage: {tokens_refresh_out}')
            print(f'current generate usage: {tokens_refresh_out} {tokens_refresh_in}')
            tokens_refresh_in += tokens_llm(question)
            # **********************
            # chat_history.extend([HumanMessage(content=question), ai_msg])
            quiz_result = quiz_extract(ai_msg.content, type_of_quiz, lang)
            quiz_removed_duplicate.extend(quiz_result)
            quiz_removed_duplicate = remove_duplicate_quizzes(quiz_removed_duplicate)
            i+=1
        print(f'cb usage: {cb_refresh.total_tokens}')
        tokens_refresh_in = cb_refresh.prompt_tokens
        tokens_refresh_out = cb_refresh.completion_tokens

    print(len(quiz_result))
    print(print(f"in: {tokens_refresh_in}"))
    print(print(f"out: {tokens_refresh_out}"))
    print(print(f"embbed: {tokens_refresh_embbed}"))
    print(quiz_result)
    quiz_send = quiz_result[0]
    return quiz_send, tokens_refresh_in, tokens_refresh_out, tokens_refresh_embbed

def callWebhook(quiz_id, url, quiz_result, tokens_in, tokens_out, tokens_embbed):
    logging.basicConfig(level=logging.INFO,  # Set the logging level
                        format='%(asctime)s [%(levelname)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger(__name__)

    payload = {'quiz_builder_id': quiz_id, 'quiz_result': quiz_result, 'tokens_in': tokens_in, 'tokens_out': tokens_out, 'tokens_embbed': tokens_embbed}  # Replace with your JSON payload
    response = requests.post(url, json=payload)
    logger.info('Webhook response: ', response,'Quiz ID: ', quiz_id, ' || Quiz Result: ', quiz_result, ' || Tokens Embbed: ', tokens_embbed)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # getChunksData(['65bb42cc990d20d41186a373'])
    quiz_result, tokens_in, tokens_out, tokens_embbed = quiz_builder('quiz_id',
        5, '65bb261763a966b42e410886', 'essay_short', 4, ['65bb5818990d20d41186ba97'], 'fr')
    #     quiz_result, tokens_refresh_in, tokens_refresh_out, tokens_refresh_embbed =quiz_editor('65bb261763a966b42e410886', 'essay_short', 3, [{'question': 'What is the purpose of evaluating activities in the context of deep work rituals?', 'answer': 'The purpose of evaluating activities in the context of deep work rituals is to determine their effectiveness in helping individuals achieve deep work and maximize productivity.'}], ['65bb42cc990d20d41186a373'], "en")
    #     print(quiz_result)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/