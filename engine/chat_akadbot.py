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
from helpers.language_helpers                import get_language_used
from helpers.chat_akadbot_helpers           import (
                                                update_chat_history,
                                                get_context_based_history,
                                                get_context_based_question,
                                            )

# *************** IMPORTS VALIDATORS ***************
from validator.chunks_validation            import (
                                                validate_context_input,
                                                validate_document_input
                                            )
from validator.data_type_validatation       import (
                                                validate_list_input,
                                                validate_string_input
                                            )

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

# *************** Function for defining RAG-Chaining with LLM as a chatbot for documents
def generate_akadbot_chain() -> Runnable:
    """
    Creates a RAG (Retrieval-Augmented Generation) chain for answering questions using an LLM.

    This function sets up a chatbot-style question-answering system where responses are 
    generated strictly based on retrieved document context. It ensures that the bot does 
    not generate answers beyond the given context.

    Returns:
        Runnable: A configured RAG chain ready for answering document-based questions.
    """

    # *************** Set QA chain prompt for bot to understand context
    qa_system_prompt = """
    You are an expert on the document. Generate answers only based on the given context. 
    Do not make up answers. If you don't know the answer based on the given context, 
    state that you can't answer questions outside of this document.
    
    The context is: '''{context}'''
    
    Please generate the answer in {language} language.
    """

    # *************** Define the chat prompt template
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # *************** Build RAG chain with document retriever, prompt, and LLM
    ragChain = create_stuff_documents_chain(
        llm=LLMModels(temperature=0.2).llm_cv,
        prompt=qa_prompt,
        output_parser=StrOutputParser(),
    )

    LOGGER.info("Akadbot Chain Generated")

    return ragChain

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
def join_reference(context: list[tuple[Document, float]], similarity_threshold: float = 0.6) -> str:
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
    for document, similarity_value in context:
        
        # *************** Only get reference with simialrity aboev the threshold
        if similarity_value > similarity_threshold:
            reference = build_reference(document)
            references.add(reference)
        
    # *************** Formatting reference into string
    return "\n".join(sorted(references))

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
            ragChain = generate_akadbot_chain()

            # *************** Convert chat history into structured format
            history_input = convert_chat_history(chat_history)
            history_input.extend([HumanMessage(content=question)])

            # *************** Retrieve relevant context from past conversations or documents
            if chat_history:
                context = get_context_based_history(history_input, course_id)
            else:
                context = get_context_based_question(question, course_id)

            LOGGER.info(f"CONTEXT: {len(context)}\n{context}")

            # *************** Get document chunks from context
            docs = [doc[0] for doc in context]

            # *************** Answering question using the RAG chain
            message = ragChain.invoke(
                {
                    "context": docs,
                    "messages": history_input,
                    "language": lang_used,
                }
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            LOGGER.info(f"TIME TO INVOKE: {elapsed_time} seconds")

            # *************** Compile reference headers from retrieved context
            header_ref = join_reference(context)

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
    ask_with_memory("hi", 'ai_doc_001', chat_history, topic)

if __name__ == "__main__":
    main()