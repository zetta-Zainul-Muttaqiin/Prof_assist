# *************** IMPORTS ***************
import re
import time
from operator                               import itemgetter
import json 
from functools import partial

# ************ IMPORT FRAMEWORKS ************** 
from langchain.prompts                      import (
                                                PromptTemplate, 
                                                MessagesPlaceholder, 
                                                ChatPromptTemplate
                                            )
from langchain_core.messages                import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents               import Document
from langchain_core.runnables               import Runnable, RunnablePassthrough
from langchain_core.output_parsers          import StrOutputParser, JsonOutputParser
from langchain_community.callbacks          import get_openai_callback
from langchain.chains.combine_documents     import create_stuff_documents_chain
from pydantic                               import Field, BaseModel
from setup import LOGGER, GREETINGS_EN, GREETINGS_FR

# *************** IMPORTS MODELS ***************
from models.llms                            import LLMModels

# *************** IMPORTS HELPERS ***************
from helpers.language_helper                import get_language_used
from helpers.chat_akadbot_helpers           import (
                                                update_chat_history,
                                                get_context_based_history,
                                                get_context_based_question,
                                            )
from helpers.json_formatting_helper         import format_json_format
from validator.data_type_validatation       import validate_message_response

# *************** IMPORTS VALIDATORS ***************
from validator.chunks_validation            import (
                                                validate_context_input,
                                                validate_document_input
                                            )
from validator.data_type_validatation       import (
                                                validate_list_input,
                                                validate_string_input,
                                                validate_dict_input
                                            )

class ExpectedAnswer(BaseModel):
    """ 
    Expected answer defines the expected structure for response need to be in JSON with key message and is_answered
    """
    
    response:str = Field(description=("Your detailed answer here explaining the response to the user's question.")),
    is_answered:str = Field(description=("'True' if you provided an answer in the 'response' field, 'False' if you cannot provide an answer"))
    
    
# *************** Function helper for help engine to convert chat history to chat messages
def convert_chat_history(chat_history: list) -> list[BaseMessage]:
    """
    Convert chat history to the chat messages for inputted to LLM.

    Args:
        chat_history (list): List of chat messages, each containing human and AI content.

    Returns:
        list: Converted chat history with alternating HumanMessage and AIMessage objects. 
                Default to empty list if no chat_hsitory 
    """
    # *************** Return empty list because no chat_history inputted and skip the process
    if not chat_history:
        LOGGER.warning("No Chat History Inputted")
        return []
    
    # *************** Validate inputs chat_history is a list
    if not validate_list_input(chat_history, 'chat_history'):
        LOGGER.error("'chat_history' must be a list of message.")
    
    # *************** Initialize formatted history
    history_inputted = []

    # *************** Add messages to formatted history
    for chat in chat_history:
        if chat['type'] == 'human':
            history_inputted.append(HumanMessage(content=chat['content']))
        elif chat['type'] == 'ai':
            history_inputted.append(AIMessage(content=chat['content']))
    
    # *************** Log history process   
    if history_inputted:
        LOGGER.info(f"Chat History is Converted to BaseMessages: {len(history_inputted)} messages")

    # *************** Return formatted history
    return history_inputted

# *************** Function to detect greeting from the question
def detect_greetings(text: str) -> bool:
    """
    Detects whether the given text contains a greeting word in English or French.

    This function checks if any word in the input text matches predefined greeting 
    words from English (`GREETINGS_EN`) or French (`GREETINGS_FR`). The input text 
    is tokenized by splitting it into words and converting them to lowercase.

    Args:
        text (str): The input text to analyze.

    Returns:
        bool: True if a greeting word is detected, otherwise False.
    """
    # *************** Validate inputs text_greetings is not an empty-string
    if not validate_string_input(text, 'text_greetings'):
        LOGGER.error("'text_greetings' must be a string.")
    
    # *************** Detect question input
    # *************** Convert text to lowercase and split into words
    words = text.lower().split(' ')
    # *************** Combine English and French greetings
    greetings = GREETINGS_EN + GREETINGS_FR

    # *************** Check each word
    for word in words:
        # *************** Return True if a greeting word is found
        if word in greetings:
            return True
    
    return False

# *************** Function to generate a topic based on chat history
def topic_creation(chat_history: list) -> str:
    """
    Generates a topic title summarizing the conversation in the chat history.

    This function uses a predefined prompt template to create a relevant topic title 
    based on the chat history. The generated topic follows the language and tone of 
    the conversation and is formatted in HTML.

    Args:
        chat_history (list[dict]): The chat history containing messages exchanged in the conversation.

    Returns:
        str: The generated topic title in HTML format.
    """
    # *************** Validate inputs chat_history is a list
    if not validate_list_input(chat_history, 'chat_history_topic'):
        LOGGER.error("'chat_history_topic' must be a list of message.")

    # *************** Define a template for generating the topic title
    topic_template = """
    Input:
        'chat_history':{chat}
    
    Instructions:
        1. Create a topic title about what the conversation is about based on the 'chat_history'.
        2. Ensure the title language and tone follow the 'chat_history'.
    
    Example Output (in HTML):
    ```html<b>Filtering Student Data for Efap Paris: Scholar Season 24-25 with Payment Confirmation</b>```
    """
    
    # *************** Initialize the prompt template
    topicPrompt = PromptTemplate(
        template=topic_template,
        input_variables=["chat"],
    )

    # *************** Define the processing chain for topic generation
    topic_chain = (
        {
            "chat": itemgetter("chat") 
        }
        | topicPrompt
        | LLMModels(temperature=1).llm_cv
    )

    # *************** Invoke the chain to generate a topic from first user question
    result = topic_chain.invoke(
        {
            "chat": chat_history[0] 
        }
    )

    # *************** Clean the result to remove extra characters
    clear_result = result.content.strip('"').replace('```html', '').replace('```', '').replace('\n', '')

    # *************** Return the cleaned topic title
    return clear_result

# *************** Function to get current question
def get_current_question(input_dict: dict) -> list:
    """
    Get current question to pass it to the retriever 
    
    Args:
        input_dict (dict): Dictionary containing messages and other inputs
        course_id (str): Course identifier for filtering documents
        
    Returns:
        list: List of Document objects
    """
    
    if not validate_dict_input(input_dict, "input_dict"): 
        LOGGER.error("input_dict must be a string")
    
    # *************** get current question from user
    history_input = input_dict["messages"]
    current_question = history_input[-1].content
    return current_question

# *************** Function for defining RAG-Chaining with LLM as a chatbot for documents
def generate_akadbot_chain(query: str ,course_id: str) -> Runnable:
    """
    Creates a RAG (Retrieval-Augmented Generation) chain for answering questions using an LLM.

    This function sets up a chatbot-style question-answering system where responses are 
    generated strictly based on retrieved document context. It ensures that the bot does 
    not generate answers beyond the given context.

    Args:
        query (str): The user's input question or request
        course_id (str): Identifier for the course to search within
        
    Returns:
        Runnable: A configured RAG chain ready for answering document-based questions.
    """

    # *************** Set QA chain prompt for bot to understand context
    qa_system_prompt = """
    You are an expert on the document. Generate answers only based on the given context. 
    Do not make up answers. Always return a response in JSON format.
    
    Instruction: 
    - If the context is provided, analyze the "messages" and 
      generate the answer based on the given context. Do not use external knowledge or assumptions.
    - If "messages" want to know explanation more, make sure to see the "context" clearly.
    
    The context is: '''{context}'''
    
    Please generate the answer in {language} language.
    
    """

    # *************** Parser json output
    parser = JsonOutputParser(pydantic_object=ExpectedAnswer)
    format_instructions = parser.get_format_instructions()
    
    # *************** Define the system prompt format for the LLM must follow
    format_system_prompt = """
    Your response **MUST** follow this exact JSON FORMAT OUTPUT **in all cases**:
    {{
    "response": "Your detailed answer here explaining the response to the user's question.",
    "is_answered": "If the "response" HAVE AN EXPLANATION ANSWER return to 'True' and return 'False' if the "response" CANNOT ANSWER."
    }}
    """

    # *************** Combine system prompts and message placeholder into a chat template
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        ("system", format_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]).partial(format_instructions=format_instructions)

    # *************** Create the document processing chain with configured LLM
    ragChain = create_stuff_documents_chain(
        llm=LLMModels(temperature=0.2).llm_cv,
        prompt=qa_prompt,
    )
    
    # *************** Build the final chain that handles context retrieval and response generation
    conv_retrieval_chain = RunnablePassthrough.assign(
        context= get_current_question | get_context_based_question(query, course_id),
    ).assign(
        answer=ragChain,
    )
    
    LOGGER.info("Akadbot Chain Generated")
    return conv_retrieval_chain

# *************** Functions helper for get reference from checking metadata header level
def build_reference(document: Document) -> str:
    """
    Constructs a hierarchical reference string from document metadata.

    Args:
        document (Document): A document object containing metadata.

    Returns:
        str: A formatted reference string.
    """
     # *************** Validate document input
    if not validate_document_input(document, 'document'):
        LOGGER.error("'document' must be a valid Document instance.")
    
    # *************** Get metadata from Document with document_name, default to "Unnamed Document"
    metadata = document.metadata
    reference_parts = [metadata.get("document_name", "Unnamed Document")]

    # *************** Extract hierarchical headers (header1 to header4) if available
    for level in range(1, 5):
        header_key = f"header{level}"
        if metadata.get(header_key):
            reference_parts.append(metadata[header_key])

    return " > ".join(reference_parts)

# *************** Function get reference after joining correct metadata
def join_reference(context: list[Document]) -> str:
    """
    Joins context references where the similarity score exceeds the threshold.

    This function extracts document names and hierarchical headers from a list of documents with their
    similarity scores, ensuring unique references in a structured format.

    Args:
        context (List[Tuple[Document, float]]): A list of tuples where each contains a Document 
            object and its similarity score.
        similarity_threshold (float, optional): Minimum similarity score to include a document 
            in the reference. Defaults to 0.6.

    Returns:
        str: A formatted string containing unique references extracted from the provided documents.
    """
    # *************** Input Validation
    if not validate_context_input(context, "context"):
        LOGGER.error("'context' Not in required format data")

    # *************** Initalize reference for get unique header
    references = set()

    # *************** Iterate through documents and extract references
    for document in context:
        
        # *************** Only get reference with simialrity aboev the threshold
        reference = build_reference(document)
        references.add(reference)
    
    sorted_reference = sorted(references)[:4]
    # *************** Formatting reference into string
    return "\n".join(sorted_reference)

# *************** Function to handling responsive chat with greetings
def greetings_chat_handler(question: str, lang_used: str) -> str:
    """
    Handles greeting messages by generating a response in the specified language.

    This function utilizes a language model to process a greeting question and return 
    a response in the requested language.

    Args:
        question (str): The greeting or message input from the user.
        lang_used (str): The language in which the response should be generated.

    Returns:
        str: A generated greeting response in the specified language.
    """

    # *************** Input Validation
    if not validate_string_input(question, "question_greetings"):
        LOGGER.error("'question_greetings' must be a string and not empty")
    if not validate_string_input(lang_used, "lang_used_greetings"):
        LOGGER.error("'lang_used_greetings' must be a string and not empty")

    # *************** Initialize Language Model
    llm = LLMModels(temperature=1.0, max_tokens=100).llm_cv
    
    # *************** Generate Response from LLM
    message = llm.invoke(f"{question}. Respond in {lang_used} language.")
    
    # *************** Ensure Response is a String
    if not isinstance(message, str):
        message = message.content

    return message

# *************** MAIN FUNCTION
def ask_with_memory(question: str, course_id: str, chat_history: list = [], topic: str = '') -> dict:
    """
    Main function to process student queries using Akadbot with memory support.

    This function handles user queries by determining the language, detecting greetings, 
    retrieving relevant document contexts, and generating a response using a RAG chain model.
    It also updates the conversation history and assigns a topic if none is provided.

    Args:
        question (str): The question asked by the student.
        course_id (str): The identifier of the course for context retrieval.
        chat_history (optional, list[dict]): The previous chat conversation to maintain context.
        topic (optional, str): The current topic of discussion. If empty, a topic is generated.

    Returns:
        dict: A dictionary containing:
            - message (str): The AI-generated response.
            - topic (str): The assigned topic for the conversation.
            - chat_history (list[dict]): Updated conversation history.
            - tokens_in (int): Number of input tokens used.
            - tokens_out (int): Number of output tokens generated.
    """
    # *************** Input Validation
    if not validate_string_input(question, "question"):
        LOGGER.error("'question' must be a string and not empty")
    if not validate_string_input(course_id, "course_id"):
        LOGGER.error("'course_id' must be a string and not empty")
    if chat_history and not validate_list_input(chat_history, "chat_history", False):
        LOGGER.error("'chat_history' must be a list and not empty")
    if topic and not validate_string_input(topic, "topic", False):
        LOGGER.error("'topic' must be a string and not empty")

    # *************** Define output target parameter for message and header refrenece in empty
    message = ''
    
    # *************** Detect language used from current question
    lang_used = get_language_used(question)

    with get_openai_callback() as cb:
        start_time = time.time()

        if detect_greetings(question.lower()):
            # *************** Invoke answer for greetings
            message = greetings_chat_handler(question, lang_used)

            end_time = time.time()
            elapsed_time = end_time - start_time
            LOGGER.info(f"TIME TO INVOKE: {elapsed_time} seconds")

            # *************** Add current question and answer into chat history (header reference is null)
            update_chat_history(chat_history, question, message)

        else:
            # *************** Generate Akadbot RAG Chain
            ragChain = generate_akadbot_chain(question,course_id)

            # *************** Convert chat history into structured format
            history_input = convert_chat_history(chat_history)
            history_input.extend([HumanMessage(content=question)])
           
            # *************** Answering question using the RAG chain
            output = ragChain.invoke({
                "messages": history_input[-5:],
                "language": lang_used,
            })
            
            print(f"\n\n history_input: {history_input}")
            print(f"\n\n output: {output}")
            
            # **************** Format json when output is not in json format  
            output_format_json = format_json_format(output)
            
            # *************** Get values message and is_answered from output json
            message_response = output_format_json.get('response')
            is_answered = output_format_json.get('is_answered')
            
            # *************** validate message if not a string
            message = validate_message_response(message_response)
            print(f"\n\n mesasages: {message}")
            print(f"\n\n is_answered: {is_answered}")
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            LOGGER.info(f"TIME TO INVOKE: {elapsed_time} seconds")

            # *************** Compile reference headers from retrieved context
            header_ref = ""
            if is_answered == 'True':
                get_contexts = output.get('context')
                print(f"\n\n get_contexts: {get_contexts}")
                header_ref = join_reference(get_contexts)

            # *************** Add current question and answer into chat history with reference
            update_chat_history(chat_history, question, message, header_ref)

        # *************** Check if topic exists; if not, generate a topic summary
        if topic == '':
            LOGGER.info("No topics inputted")
            topic = topic_creation(chat_history)

    # *************** Get token cost track from openAI
    tokens_out = cb.completion_tokens
    tokens_in = cb.prompt_tokens

    response = {
        "message": message,
        "topic": topic,
        "chat_history": chat_history,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }
    return response


def main():
    chat_history = []
    topic = ""
    ask_with_memory("what is admtc?", 'doc_1_charte', chat_history, topic)

if __name__ == "__main__":
    main()