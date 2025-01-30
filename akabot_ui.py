# ************ IMPORTS FRAMEWORK ************
import streamlit                                as st
from streamlit_option_menu                      import option_menu

# ************ IMPORTS ************
import os
import pytz
import json
from datetime import datetime

from typing                                     import List 
from datetime                                   import datetime

# ************ IMPORTS ENGINE ************
from Bilip_Ref_007                              import ask_with_memory

# ************ IMPORTS SETUP ************
from setup                                      import (
                                                        LOGGER,
                                                        DB_FILE, 
                                                        LIST_DOC_FILE,
                                                    
                                                )
# ************ IMPORTS HELPER ************
from helpers.streamlit_styling_helper           import (
                                                        styling, 
                                                        plot_title, 
                                                        padding_height, 
                                                        plot_title_chat_topic,
                                                        plot_title_select_document,
                                                )
from helpers.streamlit_format_history_helper    import (
                                                        format_chat_history, 
                                                        format_and_extract_header_returned
                                                )
from helpers.sending_payload                    import  send_log_data
# ************ IMPORTS VALIDATOR ************
from validator.data_type_validatation           import (
                                                        validate_dict_input,  
                                                        validate_list_input, 
                                                        validate_string_input
                                                )



st.set_page_config(layout="wide")

# ************ padding height
padding_height()

# ************ list docs json
with open(LIST_DOC_FILE, "r") as file:
    LIST_DOCS = json.load(file)

def save_list_to_json(list_docs: list) -> None:
    """
    Saves a list of dictionaries to a JSON file.

    Args:
        list_docs (list): The list of dictionaries to save.
        file_name (str): The name of the JSON file to save to.

    Returns:
        None
    """
    if not validate_list_input(list_docs, "list_docs"):
        raise ValueError("list_docs must be a list of dictionaries.")
    
    with open('LIST_DOC.json', "w") as json_file:
        json.dump(list_docs, json_file, indent=4)

# ************ initiate json list dict
def initialize_db():
    """
    Initiate json to save topic chat
    """
    
    # ************ initiate default format json
    default_db = [
        {
            "topic": "New Chat",
            "message": [],
            "created_at": "",
            "document": ""
        }
    ]
    
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, 'w') as file:
            json.dump(default_db, file, indent=2)
    else:
        # ************ Check if file is empty or invalid
        try:
            with open(DB_FILE, 'r') as file:
                json.load(file)
        except json.JSONDecodeError:
            # ************ If file is empty or invalid, write default structure
            with open(DB_FILE, 'w') as file:
                json.dump(default_db, file, indent=2)

# ************ load chat history
def load_chat_history():
    """
    function to load chat history
    """
    
    try:
        with open(DB_FILE, 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        # ************ If error loading the file, initialize with empty list
        default_db = []
        with open(DB_FILE, 'w') as file:
            json.dump(default_db, file, indent=2)
        return default_db

# ************ save chat history to json
def save_chat_history(topics_list):
    """
    function to save chat history
    """
    
    # ************ validate topic_list 
    if not validate_list_input(topics_list, "topic_list", False):
        LOGGER.error("topic_list must be a list")
    
    try:
        with open(DB_FILE, 'w') as file:
            json.dump(topics_list, file, indent=2)
    except Exception as e:
        st.error(f"Error saving chat history: {str(e)}")

# ************ get source doc by course name
def get_source_by_name(doc_name: str) -> str:
    """
    Function to get course_id by its name
    
    Args:
        doc_name: document name  

    Returns:
        course_id:  course id 
    """
    
    # ************ validate doc name data type 
    if not validate_string_input(doc_name, "doc_name"):
        LOGGER.error("doc_name must be a string")
            
    for doc in LIST_DOCS:
        if doc["course_name"] == doc_name:
            return doc["course_id"]
    return None

# ************ function to format topic creation
def format_topic_name(topic: str) -> str:
    """
    Function to format topic name to replace <br> with empty string
    Args:
        topic (str): topic created

    Returns:
        topic (str): topic created
    """
    
    # ************ validate topic 
    if not validate_dict_input(topic, "topic"):
        LOGGER.error(f"{topic} must be a dict")
    
    # ************ format to replace <br>
    topic_name = topic.get('topic', 'Unnamed Topic').replace("<b>", "").replace("</b>", "")
    created_at = topic.get('created_at', '')
    
    if created_at:
        return f"{topic_name}"
    return topic_name

# *********** Function to create data request and response
def log_request_response(question: str, course_id: str, chat_history: List[dict], topic:str, response_data: dict) -> dict:
    """
    Logs the request and response data to a JSON file.

    Args:
        question (str): The user's question.
        course_id (str): The document/course ID.
        chat_history (list): The chat history before the request.
        topic (str): The chat topic.
        response_data (dict): The response returned from the assistant.
    """
    
    # *********** validate input data type
    input_strings = [question, course_id, topic]
    for input_string in input_strings:
        if not validate_string_input(input_string, "input_string"):
            LOGGER.error("input must be string to create log response data")
    
    if not validate_list_input(chat_history, "chat_history"):
        LOGGER.error("chat history must be list to procces log response data")
        
    if not validate_dict_input(response_data, "response_data"):
        LOGGER.error("response data must be a dict")
    
    # *********** create data log_entry local time french
    france_tz = pytz.timezone("Europe/Paris")
    local_time_french = datetime.now(pytz.utc).astimezone(france_tz)
    
    log_entry = {
        "date_request": local_time_french.strftime("%d-%m-%Y %H:%M"),
        "endpoint": "streamlit/student_question",
        "request_data": {
            "question": question,
            "course_id": course_id,
            "chat_history": chat_history,
            "topic": topic
        },
        "response_data": {
            "message": response_data.get("message", ""),
            "chat_history": response_data.get("chat_history", []),
            "topic": response_data.get("topic", ""),
            "token_in": response_data.get("tokens_in", 0),
            "token_out": response_data.get("tokens_out", 0),
        }
    }
    
    return log_entry

# ************ function to handle buble chat to clear and create new default topic
def clear_current_chat(db_data: List[dict]):
    """
    function to handle buble chat when click start new chat 
    
    Args:
        db_data (_type_): _description_
    """
    
    # ************ initiate default topic
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_topic = f"New Chat_{timestamp}"

    # ************ Save current messages into history if not already saved
    if st.session_state.get("messages", []):
        current_topic = st.session_state.get("current_topic", "")
        if current_topic.startswith("New Chat_") or len(st.session_state["messages"]) == 0:
            LOGGER.info("New Chat topic is empty and will not be saved.")
        elif any(entry["topic"] == current_topic for entry in db_data):
            LOGGER.info("Current topic already saved in DB, skipping save.")
        else:
            db_data.append({
                "topic": current_topic,
                "messages": st.session_state["messages"],
                "created_at": datetime.now().isoformat(),
                "document": st.session_state.get("selected_doc", ""),
            })
            LOGGER.info("Saved current topic to DB.")

    # ************ Reset session state for new chat
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.current_topic = new_topic

    # ************ Avoid duplicate new topics in db_data
    existing_topics = [entry["topic"] for entry in db_data]
    if new_topic not in existing_topics:
        db_data.append({
            "topic": new_topic,
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "document": st.session_state.get("selected_doc", ""),
        })
        LOGGER.info(f"Added new topic: {new_topic}")
    else:
        LOGGER.info(f"Topic {new_topic} already exists in DB.")

    # ************ Save changes to database
    save_chat_history(db_data)

    st.sidebar.success("Started a new chat while preserving history!")
    st.rerun()

# ************ main function to run akabot streamlit ui
def akabot_ui():
    """
    Main function to render the Akabot Streamlit UI.
    
    Handles interactions for the Akabot application.
    """
    # ********** initiate db
    initialize_db()
    # ********** styling
    styling()
    # ********** Load existing chat history
    db_data = load_chat_history()
    # ********** initiate timestamp for default current topic
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ********** Initialize session states with proper message structure
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = f"New Chat_{timestamp}"
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    # ********** ui for title in main display 
    text = plot_title()
    st.markdown(text, unsafe_allow_html=True)
    st.caption("Talk with your document")
    
    # ********** Sidebar for upload and buble chat
    with st.sidebar:
        
        # ********** plot title select document in sidebar 
        plot_title_select_document()

        # ********** select box to choose course_name based on list_docs
        selected_doc = st.selectbox(
            "select document",
            options=[doc["course_name"] for doc in LIST_DOCS],
            index=0,
            label_visibility="hidden"
        )
        
        # ********** plot title chat topic in sidebar
        plot_title_chat_topic()

        # ********** sorted topic based on created_at date
        topics = sorted(db_data, key=lambda x: x.get('created_at', ''), reverse=True)

        # ********** handle if topics to be buble chat using option_menu
        if topics:
            selected_topic = option_menu(
                menu_title="",
                options=[format_topic_name(topic) for topic in topics],
                icons=["chat-dots"] * len(topics),
                menu_icon=None,
                default_index=0 if topics else None,
                styles={
                    "nav-link-selected": {"background-color": "#FF6D00"},
                    "nav-link": {"white-space": "normal", "height": "auto", "min-height": "44px"}
                }
            )
            
            selected_base_topic = selected_topic #.split(" (")[0] if selected_topic else None
            
            # ********** if selected topic normalized json to replace <br>
            if selected_base_topic:
                normalized_db_data = [
                    {
                        **item,
                        "normalized_topic": item["topic"].replace("<b>", "").replace("</b>", "")
                    }
                    for item in db_data
                ]
                
                topic_data = next(
                    (item for item in normalized_db_data if item['normalized_topic'] == selected_base_topic), 
                    None
                )
                            
                if topic_data:
                    # ********** Check if the key is 'message' or 'messages' 
                    if 'message' in topic_data:
                        st.session_state.messages = topic_data['message']  
                    elif 'messages' in topic_data:
                        st.session_state.messages = topic_data['messages']  
                    else:
                        st.session_state.messages = []  
                    
                    # ********** Update current topic
                    st.session_state.current_topic = topic_data['topic']

            # ********** handle button start new chat
            clear_button = st.button("Start New Chat", key="clear_button") 
            if clear_button:
                clear_current_chat(db_data)
            
            # ********** tooltip for clarity  
            st.markdown("""
                <div style="margin: 10px 0; border-bottom: 1px solid #ccc;"></div>
                <div style="font-size: 0.8em; color: #666;">
                    Click 'Start New Chat' to begin a new topic doc.
                </div>
            """, unsafe_allow_html=True)
            
    # ********** Main chat interface
    source = get_source_by_name(selected_doc)

    # ********** Chat container
    with st.container(border=True, height=600):
        for message in st.session_state.messages:
            if isinstance(message, dict) and "type" in message and "content" in message:
                with st.chat_message('assitant' if message["type"]=='assistant' else message["type"]):
                    st.markdown(message["content"])
                    if message["type"] == "ai" and "header_ref" in message and message["header_ref"]:
                        backslach_enter = "\n"
                        st.markdown(
                            f"""
                            <p style="color: gray; font-size: 12px;">
                                References: <br>- {message['header_ref'].replace(backslach_enter, "- ").replace("- ", "<br>- ")}
                            </p>
                            """,
                            unsafe_allow_html=True
                        )

    # ********** Chat input
    with st.container():
        if prompt := st.chat_input("What would you like to know?"):
            # ********** Add user message to display
            st.session_state.messages.append({"type": "human", "content": prompt})
                
            # ********** Process RAG
            with st.spinner("Thinking..."):
                formatted_chat_history = format_chat_history(st.session_state.chat_history)
                response = ask_with_memory(
                    prompt,
                    source,
                    formatted_chat_history,
                    "" if st.session_state.current_topic.startswith("New Chat") else st.session_state.current_topic
                )
                message = response["message"]
                chat_history = response["chat_history"]
                topics = response["topic"]
                tokens_in = response["tokens_in"]
                tokens_out = response["tokens_out"]

                # ********** Prepare response data
                response_data = {
                    "message": message,
                    "chat_history": chat_history,
                    "topic": topics,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out
                }
                
                # ********** log request-response data
                log_data = log_request_response(prompt, source, formatted_chat_history, topics, response_data)
                
                # ********** Validate and send log data to database
                if validate_dict_input(log_data, "data"):
                    send_log_data(log_data)
                else:
                    LOGGER.error("Invalid log data format, skipping log storage.")
                    
                # ********** Convert returned chat history to dictionary format
                formatted_returned_history, header_ref_extracted = format_and_extract_header_returned(chat_history)
                
                # ********** add to session 
                st.session_state.chat_history = formatted_returned_history
                st.session_state.current_topic = topics
                
                # ********** Add AI response to display
                ai_message = {
                    "type": "ai",
                    "content": message,
                    "header_ref": st.session_state.chat_history[-1].get("header_ref", header_ref_extracted)
                }
                st.session_state.messages.append(ai_message)
                
                # ********** Update database
                if topics:  
                    db_data = [entry for entry in db_data if not entry["topic"].startswith("New Chat")]
                    topic_found = False 
                    
                    for entry in db_data:
                        if entry["topic"] == topics:
                            # ********** Update the existing topic
                            entry["message"] = st.session_state.messages
                            entry["created_at"] = datetime.now().isoformat()
                            entry["document"] = selected_doc
                            topic_found = True
                            break
                    
                    # ********** If the topic was not found, create a new entry
                    if not topic_found:
                        db_data.append({
                            "topic": topics,
                            "message": st.session_state.messages,
                            "created_at": datetime.now().isoformat(),
                            "document": selected_doc
                        })
                    # ********** save chat topic to json
                    save_chat_history(db_data)
                # ********** rerun streamlit widget
                st.rerun()
                        
if __name__ == '__main__':
    akabot_ui()