import streamlit as st
from streamlit_option_menu import option_menu
from Bilip_Ref_007 import ask_with_memory
import json
import os
from datetime import datetime
from typing import Optional

DB_FILE = 'DB_FILE.json'
LIST_DOCS = [
    {"source":"ai_doc_001","document_name":"AI_book"},
    {"source": "002", "document_name": "Doc_B"},
    {"source": "003", "document_name": "Doc_C"}
]

# ********** json list 

# def initialize_db():
#     """Initialize the database with default structure if it doesn't exist"""
#     default_db = []  
    
#     if not os.path.exists(DB_FILE):
#         with open(DB_FILE, 'w') as file:
#             json.dump(default_db, file, indent=2)
#     else:
#         # Check if file is empty or invalid
#         try:
#             with open(DB_FILE, 'r') as file:
#                 json.load(file)
#         except json.JSONDecodeError:
#             # If file is empty or invalid, write default structure
#             with open(DB_FILE, 'w') as file:
#                 json.dump(default_db, file, indent=2)

# def load_chat_history():
#     
#     try:
#         with open(DB_FILE, 'r') as file:
#             return json.load(file)
#     except (json.JSONDecodeError, FileNotFoundError):
#         # If there's an error loading the file, initialize with empty list
#         default_db = []
#         with open(DB_FILE, 'w') as file:
#             json.dump(default_db, file, indent=2)
#         return default_db

# def save_chat_history(topics_list):
#   
#     try:
#         with open(DB_FILE, 'w') as file:
#             json.dump(topics_list, file, indent=2)
#     except Exception as e:
#         st.error(f"Error saving chat history: {str(e)}")

# def get_topic_by_name(topics_list, topic_name):
# 
#     for topic in topics_list:
#         if topic["topic"] == topic_name:
#             return topic
#     return None

# ********** dict 

def initialize_db():
    """Initialize the database with default structure if it doesn't exist"""
    default_db = {
        'topics': {},
        'current_topic': ''
    }
    
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, 'w') as file:
            json.dump(default_db, file, indent=2)
    else:
        # Check if file is empty or invalid
        try:
            with open(DB_FILE, 'r') as file:
                json.load(file)
        except json.JSONDecodeError:
            # If file is empty or invalid, write default structure
            with open(DB_FILE, 'w') as file:
                json.dump(default_db, file, indent=2)

def load_chat_history():
    """Load chat history from JSON file with error handling"""
    try:
        with open(DB_FILE, 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        # If there's an error loading the file, initialize with default structure
        default_db = {
            'topics': {},
            'current_topic': ''
        }
        with open(DB_FILE, 'w') as file:
            json.dump(default_db, file, indent=2)
        return default_db

def save_chat_history(db_data):
    """Save chat history to JSON file with error handling"""
    try:
        with open(DB_FILE, 'w') as file:
            json.dump(db_data, file, indent=2)
    except Exception as e:
        st.error(f"Error saving chat history: {str(e)}")

def get_source_by_name(doc_name):
    for doc in LIST_DOCS:
        if doc["document_name"] == doc_name:
            return doc["source"]
    return None

def styling():
    st.markdown("""
    <style>
    /* styling button */
    .stButton > button {
                        background-color: #FF6D00;
                        border: none;
                        color: white;
                        padding: 8px 16px;
                        text-align: center;
                        text-decoration: none;
                        display: inline-block;
                        font-size: 16px;
                        margin: 4px 2px;
                        cursor: pointer;
                        border-radius: 12px;
                    }
    
    /* Style for the selectbox component */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #ffffff;       /* White background */
        color: #000000;                  /* Black text color */
        padding: 10px 15px;              /* Padding for a spacious look */
        font-size: 16px;                 /* Font size */
        border: 1px solid #000000;       /* Thin black border */
        border-radius: 7px;              /* Rounded corners for a smooth look */
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
        transition: box-shadow 0.3s ease; /* Smooth shadow transition */
    }

    /* Hover effect for the selectbox */
    .stSelectbox div[data-baseweb="select"]:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Enhanced shadow on hover */
    }
    </style>
    """, unsafe_allow_html=True)

def plot_title():
    gradient_text_html = """
    <style>
    .gradient-text {
        font-weight: bold;
        background: -webkit-linear-gradient(left, red, orange);
        background: linear-gradient(to right, red, orange);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline;
        font-size: 3em;
    }
    </style>
    <div class="gradient-text">AKABOT</div>
    """
    return gradient_text_html

def process_query_with_context(prompt, source, chat_history, current_topic):
    return ask_with_memory(
            prompt,
            source,
            chat_history,
            current_topic
        )
    
def format_topic_name(topic, db_data):
    """Format topic name with creation date if available"""
    if topic in db_data['topics']:
        created_at = datetime.fromisoformat(db_data['topics'][topic]['created_at'])
        return f"{topic}"
    return topic

def clear_current_chat(db_data):
    # try:
        # Generate a new topic name for the fresh chat
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # static topic name
    new_topic = f"New Chat"
        
        # Store the current messages in history if there are any
    if st.session_state.current_topic and st.session_state.messages:
            old_topic = st.session_state.current_topic
            if old_topic in db_data['topics']:
                db_data['topics'][old_topic]['messages'] = st.session_state.messages
        
        # Reset current chat
    st.session_state.messages = []
    st.session_state.chat_history = []
        
        # Update current topic to the new one
    st.session_state.current_topic = new_topic
    db_data['current_topic'] = new_topic
        
        # Initialize new topic in database
    if new_topic not in db_data['topics']:
            db_data['topics'][new_topic] = {
                'messages': [],
                'created_at': datetime.now().isoformat(),
                'document': st.session_state.get('selected_doc', '')  # Store current document
            }

        # Save changes to database
    save_chat_history(db_data)
        
    st.sidebar.success("Started a new chat while preserving history!")
    
    st.rerun()
        
    # except Exception as e:
    #     st.sidebar.error(f"Error starting new chat: {str(e)}")
    #     print(f"Error details: {str(e)}")


def akabot_ui2():
    # Initialize database
    initialize_db()
    # styling
    styling()
    # Load existing chat history
    db_data = load_chat_history()
    
    # Initialize session states with proper message structure
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = db_data.get('current_topic', '')
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Rest of  UI code...
    text = plot_title()
    st.markdown(text, unsafe_allow_html=True)
    st.caption("Talk with your document")
    
    # Sidebar with topic navigation
    with st.sidebar:
        for _ in range(4):
            st.write("")
            
        st.markdown("""
            <div style="
                background-color: #ffffff;
                border-radius: 5px;
                padding: 5px;
                margin-bottom: 10px;
                text-align: center;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15);
            ">
                <h3 style="
                    color: #333333;
                    font-size: 20px;
                    font-weight: bold;
                    margin: 0;
                    text-transform: uppercase;
                    letter-spacing: 1.2px;
                ">Choose Doc</h3>
            </div>
        """, unsafe_allow_html=True)
         
        selected_doc = st.selectbox(
            "",
            options=[doc["document_name"] for doc in LIST_DOCS],
            index=0
        )
        
        for _ in range(4):
            st.write("")
        
        st.markdown("""
            <div style="
                background-color: #ffffff;
                border-radius: 5px;
                padding: 5px;
                margin-bottom: 10px;
                text-align: center;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15);
            ">
                <h3 style="
                    color: #333333;
                    font-size: 20px;
                    font-weight: bold;
                    margin: 0;
                    text-transform: uppercase;
                    letter-spacing: 1.2px;
                ">Chat Topic</h3>
            </div>
        """, unsafe_allow_html=True)

        topics = list(db_data['topics'].keys())
        topics.sort(key=lambda x: db_data['topics'][x]['created_at'], reverse=True)
        
        if topics:
            selected_topic = option_menu(
                menu_title="",
                options=[format_topic_name(topic, db_data) for topic in topics],
                icons=["chat-dots"] * len(topics),
                menu_icon=None,
                default_index=0 if topics else None,
                styles={
                    "nav-link-selected": {"background-color": "#FF6D00"},
                    "nav-link": {"white-space": "normal", "height": "auto", "min-height": "44px"}
                }
            )
            
            # Extract the base topic name (without timestamp)
            selected_base_topic = selected_topic.split(" (")[0] if selected_topic else None
            
            if selected_base_topic:
                st.session_state.messages = db_data['topics'][selected_base_topic]['messages']
                st.session_state.current_topic = selected_base_topic
                
            clear_button = st.button("Start New Chat", key="clear_button") 
            if clear_button:
                clear_current_chat(db_data)
            
            # tooltip for clarity  
            st.markdown("""
                <div style="margin: 10px 0; border-bottom: 1px solid #ccc;"></div>
                <div style="font-size: 0.8em; color: #666;">
                    Click 'Start New Chat' to begin a fresh conversation.
                </div>
            """, unsafe_allow_html=True)
            

    # Main chat interface
    source = get_source_by_name(selected_doc)
    
    # Chat container
    with st.container(border=True, height=550):
        
        # Display chat messages
        with st.container(border=False, height=500):
            for message in st.session_state.messages:
                with st.chat_message(message["type"]):
                    st.markdown(message["content"])
                    if message["type"] == "ai" and "header_ref" in message and message["header_ref"]:
                        st.markdown(f"*References:* {message['header_ref']}", unsafe_allow_html=True)
        
        # Chat input
    with st.container():
            if prompt := st.chat_input("What would you like to know?"):
                # Add user message to display
                st.session_state.messages.append({"type": "human", "content": prompt})
                
                # Process RAG
                with st.spinner("Thinking..."):
                    # try:
                        # Convert session chat history to the format expected by process_query_with_context
                        formatted_chat_history = []
                        for msg in st.session_state.chat_history:
                            # Check if msg is a Message object or dictionary
                            # Message object
                            if hasattr(msg, 'type'): 
                                msg_type = msg.type
                                msg_content = msg.content
                                msg_header_ref = getattr(msg, 'header_ref', '') if msg_type == 'ai' else ''
                            # dict
                            else:  
                                msg_type = msg["type"]
                                msg_content = msg["content"]
                                msg_header_ref = msg.get("header_ref", "") if msg_type == "ai" else ""
                            
                            formatted_chat_history.append({
                                "type": msg_type,
                                "content": msg_content,
                                "header_ref": msg_header_ref
                            })
                        
                        message, chat_history, topics, _, _, header_ref_array = process_query_with_context(
                            prompt,
                            source,
                            formatted_chat_history,
                            st.session_state.current_topic
                        )
                        print(f"\n\n Topic: {topics}")
                        # Update session states
                        # Convert returned chat history to dictionary format
                        formatted_returned_history = []
                        for msg in chat_history:
                            # message object
                            if hasattr(msg, 'type'):  
                                msg_dict = {
                                    "type": msg.type,
                                    "content": msg.content
                                }
                                if msg.type == "ai":
                                    msg_dict["header_ref"] = getattr(msg, 'header_ref', '')
                                formatted_returned_history.append(msg_dict)
                            # already dict
                            else:  
                                formatted_returned_history.append(msg)
                        
                        st.session_state.chat_history = formatted_returned_history
                        st.session_state.current_topic = topics
                        
                        # Add AI response to display
                        ai_message = {
                            "type": "ai",
                            "content": message,
                            "header_ref": header_ref_array[-1] if header_ref_array else ""
                        }
                        st.session_state.messages.append(ai_message)
                        
                        # Update database
                        if topics:  # Only save if we have a topic
                            if topics not in db_data['topics']:
                                db_data['topics'][topics] = {
                                    'messages': [],
                                    'created_at': datetime.now().isoformat(),
                                    'document': selected_doc
                                }
                            db_data['topics'][topics]['messages'] = st.session_state.messages
                            db_data['current_topic'] = topics
                            save_chat_history(db_data)
                        
                        # Force a rerun to update the UI
                        st.rerun()
                        
                    # except Exception as e:
                    #     st.error(f"An error occurred: {str(e)}")
                    #     print(f"Error details: {str(e)}")  # For debugging purposes
                
if __name__ == '__main__':
    akabot_ui2()