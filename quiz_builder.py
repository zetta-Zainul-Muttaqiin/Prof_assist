from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.chat_models.openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.prompts.chat import (
    ChatPromptTemplate
)
from langchain.callbacks.manager import get_openai_callback
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_transformers import (
    LongContextReorder,
)
import os
import json
import openai
import requests
import time
import tiktoken
import logging
import re
from setup import vstore, topics_collection

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
print(f'api: {openai_api_key}')

astradb_token_key = os.getenv("ASTRADB_TOKEN_KEY")
astradb_api_endpoint = os.getenv("ASTRADB_API_ENDPOINT")
astradb_collection_name = os.getenv("ASTRADB_COLLECTION_NAME")
url_webhook = os.getenv("URL_WEBHOOK_QUIZ_BUILDER")

# *************** class for Multiple output field in JSON Parser
class ResponseMultiple(BaseModel):
    question: str = Field(description="The question of the quiz")
    option: list = Field(description=f"""list of strings containing options for the question. Start each option with alphabet like {["A)", "B)", "C)", "D)", "E)"]}. Ensure that there is no comma after the final option.""")
    correct_answer: str = Field(description="The correct answer to the question")
    explanation: str = Field(description="Explanation of the correct answer")

# *************** class for Essay output field in JSON Parser
class ResponseEssay(BaseModel):
    question: str = Field(description="The question generated based on the context")
    answer: str = Field(description="The answer to the question")

# *************** retry for refresh_quiz
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

# *************** define prompt based on type of quiz
def get_prompt(quiz_type, option_amount, lang):
    part_prompt = """"""
    final_prompt = """"""
    # *************** quiz generated based on the language requested
    if lang == "fr":
        language = "You must create in French language."
    else:
        language = "You must create in English language."
    if quiz_type == 'multiple':
        option_contents = ['Option 1 content', 'Option 2 content', 'Option 3 content', 'Option 4 content', 'Option 5 content']
    
        options = []
        option_labels = ['A', 'B', 'C', 'D', 'E'] 

        for index in range(min(option_amount, len(option_contents))):
            option = f"{option_labels[index]}) {option_contents[index]}"
            options.append(option)

        part_prompt = '\n'.join(options)
        final_prompt = f"""
    You will read this context, understand it, and create quiz for the students. The context is:
    {{context}}
   
    Avoid 'all of the above' answer.
    When creating multiple choices question, set the {option_amount} hard-to-answer 'options' in bullet points with only ONE right answer and put the right answer below it with explanation. Here is the format you MUST follow:
    {{format_instructions}}
    "question": This is the question. Use question sentence. Create question by using the 'context' from the document.
    "options": an array of {option_amount} strings which consists of {part_prompt}
    "correct_answer": A) ... (make sure the corrent answer option is matched with the value, based on the option)
    "explanation": This is the explanation

    Return in array of object format with each of the above is within an object
    """
    elif (quiz_type == 'essay_long'):
        final_prompt = f"""
    You are an expert of the document to create COMPLEX and IN-DEPTH essay questions that require detailed, long answers for students. Based on this context:
    {{context}}

    Create the question and then the answer below. Here is the format you MUST follow:
    {{format_instructions}}
    "question": Formulate a complex, thought-provoking question that challenges conceptual understanding.
    "answer": Provide a comprehensive, detailed answer to the above question.

    Return in array of object format with each of the above is within an object
    """

    elif (quiz_type == 'essay_short'):
        final_prompt = f"""
    You are an expert of the document to create CONCISE essay questions that require very brief, one or two-sentence answers for students. Based on this context:
    {{context}}

    Create the question and then the answer below. Here is the format you MUST follow:
    {{format_instructions}}
    "question": Draft a straightforward, easily understandable question that can be answered in a few words or a sentence.
    "answer": Provide a brief, to-the-point answer to the above question.

    Return in array of object format with each of the above is within an object
    """

    final_prompt += f'\n{language}' 
    return (final_prompt)
# *************** end of define quiz based on the type

# *************** to remove duplicated questions based on question field
def remove_duplicate_quizzes(quizzes):
    seen_questions = set()
    unique_quizzes = []

    for quiz in quizzes:
        question = quiz.get("question", "").lower()

        # Check if the question is not duplicated
        if question not in seen_questions:
            seen_questions.add(question)
            unique_quizzes.append(quiz)

    return unique_quizzes
# *************** end of to remove duplicated questions based on question field

#*************** function to retrieve header data from chunks in vector database
def getChunksData(document_id, amount_of_quiz):
 print('ENTERING getChunksData')
 from astrapy.db import AstraDB, AstraDBCollection
 import random

 #*************** initializing the DB connection and the generator to retrieve data from vector DB
 array_header = set()
 print('Getting chunks data')
 for index in range(len(document_id)):
   generator = topics_collection.paginated_find(
    filter={"metadata.document_id": document_id[index]},
    options={"limit": 4*(amount_of_quiz*4 + 8)}
)
#*************** end of initializing the DB connection and the generator to retrieve data from vector DB
   
#*************** filtering data to get header with certain characteristics.

   print(generator)
   for doc in generator:
     if 'metadata' in doc:
        for index in range(1, 6):
            header_key = f'header{index}'
            if header_key in doc['metadata']:
                header_value = doc['metadata'][header_key]
                if header_value is not None and 5 <= len(header_value) <= 40:
                    header_value = header_value.replace('\t', ' ').replace('\n', ' ')
                    array_header.add(header_value)
#*************** end of filtering data to get header with certain characteristics.

#*************** shuffling the required header data
 shuffled_array_header = list(array_header)
 random.shuffle(shuffled_array_header)
 print('DATA LENGTH: ', len(shuffled_array_header))
 print('All Topics: ', shuffled_array_header)
 return(shuffled_array_header)
#*************** end of function to retrieve header data from chunks in vector database

#*************** Main function to run the quiz creation
def detect_and_create_quizzes(quiz_id, quiz_builder_description_id, text, source, quiz_type, multiple_option_amount, document_id, lang, quiz_generated=[]):
    tokens_in = 0
    tokens_out = 0
    result_json = []
    #*************** default set to 30 token usage for each request
    tokens_embbed = 30

    #*************** RAG chain initialization
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)
    # *************** define retrieve for get relevants document in multiple document_id
    retriever = vstore.as_retriever(
            search_type ='similarity', 
            search_kwargs = {
                "k": 5, 
                'filter': {
                    "$and": [
                        {
                        'source': source
                        },
                        {
                            "$or": 
                                [{'document_id': doc_id} for doc_id in document_id]
                            
                        }
                    ]
                }
            }
        )
    # ********* set object for parser in chain
    if(quiz_type=='multiple'):
         parser = JsonOutputParser(pydantic_object=ResponseMultiple)
    else:
         parser = JsonOutputParser(pydantic_object=ResponseEssay)
    content_iteration = 0
    quiz_removed_duplicate = quiz_generated
    number_of_quiz_per_iteration = 4
    number_match = re.search(r'\b\d+\b', text)
    # *************** get heade as topic of content
    content = getChunksData(document_id, int(number_match.group()))
    #*************** activate cost track
    with get_openai_callback() as cb_quiz:
            original_number = int(number_match.group())
            fix_original_number = original_number
            print(f"Original Number: {original_number}")
            #*************** Loop to generate quiz per 4 questions
            while (len(result_json) < fix_original_number):
                    #*************** regex substitute the original question. r'\b\d+\b' means targeting digit over words boundary. str(number_of_quiz_per_iteration) is the substituent. {text} is the variable to be substituted by the substituent                     
                    question = re.sub(r'\b\d+\b', str(number_of_quiz_per_iteration), f"""{text}""")
                    #*************** end of regex substitute the original question. r'\b\d+\b' means targeting digit over words boundary. str(number_of_quiz_per_iteration) is the substituent. {text} is the variable to be substituted by the substituent                     
                    question = f"""{question}                        
                    {get_prompt(quiz_type, multiple_option_amount, lang)}"""
                    retrieved_docs = ''
                    # *************** loop topic from header for get context that relevant with that header
                    for index in range(0,3):
                        chunks = retriever.get_relevant_documents(content[(4*content_iteration - index) % len(content)])
                        reordering = LongContextReorder()
                        chunks = reordering.transform_documents(chunks)
                        print('CHUNKS RETRIEVED: ', chunks)
                        # *************** join chunks to one context string
                        for chunk_content in chunks:
                            retrieved_docs += chunk_content.page_content
                    # *************** end of loop topic from header                               
                    
                    CUSTOM_QUIZ_PROMPT = PromptTemplate(
                        template=question,
                        input_variables=["context"],
                        partial_variables={"format_instructions": parser.get_format_instructions()}
                        )     
                               
                    content_iteration += 1
                    print('QUESTION: ', question)
                    # *************** define chain for chain context, prompt, model and parser
                    qa_chain = (
                        {"context": itemgetter("context")}
                        | CUSTOM_QUIZ_PROMPT
                        | llm
                        | parser
                        ) 
                    
                    # *************** try to generate quiz from context given based on header or topic
                    try:
                      result = qa_chain.invoke({"context": retrieved_docs})
                      print('ANSWER: ', result)
                      
                    except Exception as e:
                        print(f"error: {e}")
                        invalid_format = str(e)
                        match = re.search(r'\[', invalid_format)

                        if match:
                            start_index = match.start()
                            array_string = invalid_format[start_index:]
                            json_string = re.sub(r',\s*]', ']', array_string)
                            result = json.loads(json_string)
                        else:
                            print(f"No array found: {e}")

                    result_json.extend(result)
                      # *************** validate option field is same as expected long of option amount
                    if quiz_type == 'multiple':
                        result_json = [obj for obj in result_json if len(obj.get("options", [])) == multiple_option_amount]

                    print("LEN RESULT JSON: ", len(result_json))
                    result_json = remove_duplicate_quizzes(result_json)
                    print('RESULT_JSON: ', result_json)
                    # ****************************** count tokens for cost tracking
                    print(f'generate usage: {cb_quiz.total_tokens}')
                    # ****************************** end of count tokens for cost tracking
                    original_number -= len(result_json)
                #*************** end of Loop to generate quiz per 4 questions

            tokens_in = cb_quiz.prompt_tokens
            tokens_out = cb_quiz.completion_tokens
            #*************** end cost track usage
            callWebhook(quiz_id, quiz_builder_description_id, url_webhook, result_json[:fix_original_number], tokens_in, tokens_out, tokens_embbed, 'success', lang)
            return quiz_removed_duplicate, tokens_in, tokens_out, tokens_embbed


def quiz_builder(quiz_id, quiz_builder_description_id, number_of_quiz, source, type_of_quiz, multiple_option_amount, document_id, lang):
    # rag_chain()
    quiz_word = ''
    if (type_of_quiz == 'multiple'):
        quiz_word = 'questions'
    else:
        quiz_word = 'questions with the answer'

    text = f"Create {number_of_quiz} {quiz_word}"
    #*************** try and catch to handle error rate limit status from OpenAI. If error rate limit, send webhook with status 'Rate Limit'
    try:
        string_quiz, tokens_in, tokens_out, tokens_embbed = detect_and_create_quizzes(quiz_id, quiz_builder_description_id, text, source, type_of_quiz, multiple_option_amount, document_id, lang, [])
        print("Webhook success called")
    except openai.RateLimitError as er:
        print(er)
        callWebhook(quiz_id, quiz_builder_description_id, url_webhook, [], 0, 0, 0, 'rate_limit', lang)
        print("Webhook rate limit called")
    #*************** end of try and catch to handle error rate limit status from OpenAI. If error rate limit, send webhook with status 'Rate Limit'

    print(f"tokens_in: {tokens_in}")
    print(f"tokens_out: {tokens_out}")
    print(f"tokens usage: {tokens_in + tokens_out}")
    return string_quiz, tokens_in, tokens_out, tokens_embbed


#*************** function for quiz refresh
@retry_with_openAI
def quiz_editor(source, type_of_quiz, multiple_option_amount, quiz_generated, document_id, lang):
    tokens_refresh_out = 0
    tokens_refresh_in = 0
    tokens_refresh_embbed = 30
    result_json = []
    quiz_word = ''

    # ********* set prompt type andw object for parser in chain
    if (type_of_quiz == 'multiple'):
        quiz_word = 'quiz'
        parser = JsonOutputParser(pydantic_object=ResponseMultiple)
    else:
        quiz_word = 'question with the answer'
        parser = JsonOutputParser(pydantic_object=ResponseEssay)
    text = f"Create 1 {quiz_word}"
    # *************** get header as content of topic
    content = getChunksData(document_id, 10)
    #*************** RAG chain initialization
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=1)
    # *************** define retrieve for get relevants document in multiple document_id
    retriever = vstore.as_retriever(
                search_type ='similarity', 
                search_kwargs = {
                    "k": 10, 
                    'filter': {
                        "$and": [
                            {
                            'source': source
                            },
                            {
                                "$or": 
                                    [{'document_id': doc_id} for doc_id in document_id]
                                
                            }
                        ]
                    }
                }
            )   
    #*************** end of RAG chain initialization
    content_iteration = 0
    print('QUIZ GENERATED', quiz_generated)
    result_json.extend(quiz_generated)
    print(len(result_json))
    #*************** loop to generate a quiz refresh
    with get_openai_callback() as cb_refresh:
            while (len(result_json) < len(quiz_generated) + 1):
                    question = f"""{text}                        
                    {get_prompt(type_of_quiz, multiple_option_amount, lang)}"""
                    retrieved_docs = ''
                    chunks = retriever.get_relevant_documents(content[(content_iteration) % len(content)])
                    reordering = LongContextReorder()
                    chunks = reordering.transform_documents(chunks)
                    print('CHUNKS RETRIEVED: ', chunks)
                    # *************** join chunks to one context string
                    for chunk_content in chunks:
                        retrieved_docs += chunk_content.page_content                           
                    
                    CUSTOM_QUIZ_PROMPT = PromptTemplate(
                        template=question,
                        input_variables=["context"],
                        partial_variables={"format_instructions": parser.get_format_instructions()}
                        )     
                               
                    content_iteration += 1
                    print('QUESTION: ', question)
                    # *************** define chain for chain context, prompt, model and parser
                    qa_chain = (
                        {"context": itemgetter("context")}
                        | CUSTOM_QUIZ_PROMPT
                        | llm
                        | parser
                        ) 
                    
                    # *************** try to generate quiz from context given based on header or topic
                    try:
                      result = qa_chain.invoke({"context": retrieved_docs})
                      print('ANSWER: ', result)
                      
                    except Exception as e:
                        print(f"error: {e}")
                        invalid_format = str(e)
                        match = re.search(r'\[', invalid_format)

                        if match:
                            start_index = match.start()
                            array_string = invalid_format[start_index:]
                            json_string = re.sub(r',\s*]', ']', array_string)
                            print(f"refresh: {json_string}")
                            result = json.loads(json_string)
                        else:
                            print(f"No array found: {e}")

                    result_json.extend(result)
                      # *************** validate option field is same as expected long of option amount
                    if type_of_quiz == 'multiple':
                        result_json = [obj for obj in result_json if len(obj.get("options", [])) == multiple_option_amount]

                    print("LEN RESULT JSON: ", len(result_json))
                    result_json = remove_duplicate_quizzes(result_json)
                    print('RESULT_JSON: ', result_json)
                    # ****************************** count tokens for cost tracking
                    # ****************************** end of count tokens for cost tracking
    print(f'cb usage: {cb_refresh.total_tokens}')
    tokens_refresh_in = cb_refresh.prompt_tokens
    tokens_refresh_out = cb_refresh.completion_tokens

    #*************** end of loop to generate a quiz refresh
    print(f"in: {tokens_refresh_in}")
    print(f"out: {tokens_refresh_out}")
    print(f"embbed: {tokens_refresh_embbed}")
    return result_json[-1], tokens_refresh_in, tokens_refresh_out, tokens_refresh_embbed


#*************** function to call webhook to send result of quiz creation to BE
def callWebhook(quiz_id, quiz_builder_description_id, url, quiz_result, tokens_in, tokens_out, tokens_embbed, status, lang):
    logging.basicConfig(
        filename='error.log', # Set a file for save logger output 
        level=logging.INFO, # Set the logging level
        format='%(asctime)s [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    payload = {'quiz_builder_id': quiz_id, 'quiz_builder_description_id': quiz_builder_description_id, 'quiz_result': quiz_result, 'tokens_in': tokens_in, 'tokens_out': tokens_out, 'tokens_embbed': tokens_embbed, 'status_quiz': status, 'lang': lang}  # Replace with your JSON payload
    response = requests.post(url, json=payload)
    logger.info('Webhook response: ', response,'Quiz ID: ', quiz_id, ' || Quiz Result: ', quiz_result, ' || Tokens Embbed: ', tokens_embbed)

#*************** end of function to call webhook to send result of quiz creation to BE

if __name__ == '__main__':
    # getChunksData(['65bb42cc990d20d41186a373'])
    # quiz_result, tokens_in, tokens_out, tokens_embbed = quiz_builder('quiz_id', 'quiz_builder_description_id',
    #     2, 'test_vi_15032024_venom', 'multiple', 2, ['test_vi_15032024_venom'], 'en')
        quiz_result, tokens_refresh_in, tokens_refresh_out, tokens_refresh_embbed =quiz_editor('reni_test3', 'essay_short', 3, [{'question': 'What is the purpose of evaluating activities in the context of deep work rituals?', 'answer': 'The purpose of evaluating activities in the context of deep work rituals is to determine their effectiveness in helping individuals achieve deep work and maximize productivity.'}], ['65bb42cc990d20d41186a373'], "en")
        print(quiz_result)