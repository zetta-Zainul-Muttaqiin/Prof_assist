# *************** IMPORTS ***************
import re
import time
from operator                               import itemgetter

# ************ IMPORT FRAMEWORKS ************** 
from langchain.prompts                      import (
                                                PromptTemplate, 
                                                MessagesPlaceholder, 
                                                ChatPromptTemplate
                                            )
from langchain_core.messages                import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents               import Document
from langchain_core.runnables               import Runnable
from langchain_core.output_parsers          import StrOutputParser
from langchain_community.callbacks          import get_openai_callback
from langchain.chains.combine_documents     import create_stuff_documents_chain

from setup import LOGGER, GREETINGS_EN, GREETINGS_FR

# *************** IMPORTS MODELS ***************
from models.llms                            import LLMModels

# *************** IMPORTS HELPERS ***************
from helpers.astradb_connect_helper         import get_vector_collection
from helpers.language_helper                import get_language_used

# *************** IMPORTS VALIDATORS ***************
from validator.data_type_validatation       import (
                                                validate_dict_input,
                                                validate_list_input,
                                                validate_string_input,
                                            )

# *************** Function helper for help engine to convert chat history to chat messages
def convert_chat_history(chat_history: list) -> list[BaseMessage]:
    """
    Convert chat history to the chat messages for inputted to LLM.

    Args:
        chat_history (list): List of chat messages, each containing human and AI content.

    Returns:
        list: Converted chat history with alternating HumanMessage and AIMessage objects.
    """
    
    # *************** Validate inputs chat_history is alist
    if not validate_list_input(chat_history, 'chat_history', False):
        LOGGER.error("'chat_history' must be a list of message.")
    
    # *************** Initialize formatted history
    history_inputted = []

    # *************** Add messages to formatted history
    for chat in chat_history:
        if chat['type'] == 'human':
            history_inputted.append(HumanMessage(content=chat['content']))
        elif chat['type'] == 'ai':
            history_inputted.append(AIMessage(content=chat['content']))
    
    if history_inputted:
        LOGGER.info(f"Chat History is Converted to BaseMessages: {len(history_inputted)} messages")  
    else:
        LOGGER.warning("No Chat History Inputted")

    # *************** Return formatted history
    return history_inputted

def detect_greetings(text):
    # ********* detect question input
    # ********* word_tokenize into english if question is french
    words = text.lower().split(' ')
    greetings = GREETINGS_EN + GREETINGS_FR
    # ********* check each word
    for word in words:
        # ********* True if find greetings word from question input
        if word in greetings:
            return True
    
    return False

def topic_creation(chat_history):
    # ****** Define a template for generating the topic title
    topic_template = """
    Input:
        'chat_history':{chat}
    
    Instructions:
        1. Create a topic title about what the conversation is about based on the 'chat_history'.
        2. Concern to the title language and tone response should primarily follow the 'chat_history'.
    
    Example Output (in HTML):
    ```html<b>Filtering Student Data for Efap Paris: Scholar Season 24-25 with Payment Confirmation</b>```
    """
    
    # ****** Initialize the prompt template
    topicPrompt = PromptTemplate(
        template=topic_template,
        input_variables=[
            "chat"
        ],
    )

    # ****** Define the processing chain for topic generation
    topic_chain = (
        {
            "chat": itemgetter("chat")  # Map the chat input for processing
        }
        | topicPrompt  # Apply the prompt template
        | LLMModels(temperature=1).llm_cv  # Use the LLM to generate the topic
    )

    # ****** Invoke the chain to generate a topic
    result = topic_chain.invoke(
        {
            "chat": chat_history[0]  # Extract the human message from the chat history
        }
    )

    # ****** Clean the result to remove extra characters
    clear_result = result.content.strip('"').replace('```html', '').replace('```', '').replace('\n', '')

    # ****** Return the cleaned topic title
    return clear_result

def get_context_based_question(query: str, course_id: str) -> list[tuple[Document, float]]:

    vector_coll = get_vector_collection()
    
    relevant_docs_score = vector_coll.similarity_search_with_relevance_scores(
        query=query, k=4, filter={'course_id': course_id}, score_threshold=0.4
    )
    return relevant_docs_score


def generate_akadbot_chain() -> Runnable:
    """
    
    """

    # ********* set QA chain prompt for bot can understand context
    qa_system_prompt = """
    You are an expert of the document. Generate answer only based on the given context. Do not make up the answer.
    If you don't know the answer basend on the given context, telling you can't answer question outside of this document.
    The context is '''{context}'''
    Please generate answer in {language} language 
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # ********* build chain with retriever context, the context it self, prompt and llm
    
    ragChain = create_stuff_documents_chain(
        llm=LLMModels(temperature=0.2).llm_cv, prompt=qa_prompt, output_parser=StrOutputParser()
    )

    LOGGER.info("Akadbot Chain Generated")

    return ragChain

def get_question_history(conversation: list[BaseMessage], latest_chat: int = 2) -> list[HumanMessage]:

        length = -(latest_chat*2)
        question = []
        for chat in conversation[length:]:
            if chat.type == 'human':
                question.append(chat.content)
        
        return question


def get_context_based_history(conversation: list[BaseMessage], course_id) -> list[tuple[Document, float]]:

    conversation_question = get_question_history(conversation)

    context_history = []
    for question in conversation_question:
        question_context = get_context_based_question(question, course_id)
        for context in question_context:
            context_history.extend([context])

    return context_history


def ask_with_memory(question, course_id, chat_history=[], topic=''):

    lang_used = get_language_used(question)
    message = ''
    header_ref = ''

    with get_openai_callback() as cb:
        start_time = time.time()
        if detect_greetings(question.lower()):
            # ********* invoke answer for greetings
            llm = LLMModels(temperature=1.0, max_tokens=100).llm_cv
            message = llm.invoke(f"{question}. Response with {lang_used} language")
            # ********* save as message response
            if not isinstance(message, str):
                message = message.content

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"TIME TO INVOKE: {elapsed_time} seconds")
            # ********* Add current question and answer into chat history, also header reference but null
            chat_history.extend(
                [
                    {"type": "human", "content": question}, 
                    {"type": "ai", "content": message, 'header_ref': ''}
                ]
            )
        else:
            
            ragChain = generate_akadbot_chain()

            history_input = convert_chat_history(chat_history)
            
            history_input.extend([HumanMessage(content=question)])
            print("MESSAGES :", history_input)

            if chat_history:
                context = get_context_based_history(history_input, course_id)
            
            else:
                context = get_context_based_question(question, course_id)

            LOGGER.info(f"CONTEXT: {len(context)}\n{context}")

            docs = []
            for doc in context:
                docs.append(doc[0])

            # ********* Answering question with rag_chain
            message = ragChain.invoke(
                {
                    "context": docs,
                    "messages": history_input, 
                    "language": lang_used,
                }
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"TIME TO INVOKE: {elapsed_time} seconds")

            # Get context reference
            for doc in context:

                if doc[1] > 0.6:
                
                    # ********* set first reference is document name
                    header_ref = f"{header_ref} - {doc[0].metadata['document_name']}"
                    # ********* if there is header1 on metadata and set as next reference
                    if ('header1' in doc[0].metadata) & (doc[0].metadata['header1'] != None):
                        header_ref = f"{header_ref} > {doc[0].metadata['header1']}"
                    # ********* if there is header2 on metadata and set as next reference
                    if ('header2' in doc[0].metadata) & (doc[0].metadata['header2'] != None):
                        header_ref = f"{header_ref} > {doc[0].metadata['header2']}"
                    # ********* if there is header3 on metadata and set as next reference
                    if ('header3' in doc[0].metadata) & (doc[0].metadata['header3'] != None):
                        header_ref = f"{header_ref} > {doc[0].metadata['header3']}"
                    # ********* if there is header4 on metadata and set as next reference
                    if ('header4' in doc[0].metadata) & (doc[0].metadata['header4'] != None):
                        header_ref = f"{header_ref} > {doc[0].metadata['header4']}"
                    header_ref += "\n"
                    print(f"header chuhks :{header_ref}")

              
            # ********* add all header_reference and removing the duplicated reference
            unique_lines = set(header_ref.split('\n'))
            header_ref = '\n'.join(unique_lines)
            # ********* end of add all header_reference and removing the duplicated reference
            print('HEADER REF ARRAY: ', header_ref)
            print('')
            print('-' * 250)
            
            # ********* Add current question and answer into chat history, also header reference but null
            chat_history.extend(
                [
                    {"type": "human", "content": question}, 
                    {"type": "ai", "content": message, 'header_ref': header_ref}
                ]
            )

                # Chek if topic already exist and create summary as Topic
        print(f'history: {chat_history}')
        if (topic == ''):
            print(f"no topics")
            topic = topic_creation(chat_history)
        print(cb)

    tokens_out = cb.completion_tokens
    tokens_in = cb.prompt_tokens

    return message, chat_history, topic, tokens_out, tokens_in

def main():
    chat_history = []
    topic = ""
    ask_with_memory("hi", 'ai_doc_001', chat_history, topic)

if __name__ == "__main__":
    main()