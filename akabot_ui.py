# *********** import libraries
import streamlit            as st
from streamlit_option_menu  import option_menu
from Bilip_Ref_007          import ask_with_memory
import json
import os
from datetime               import datetime

st.set_page_config(layout="wide")

# padding height
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

DB_FILE = 'DB_FILE.json'
LIST_DOCS = [
 {'course_id': 'doc_1_charte', 'course_name': 'Charte du Centre Partenaire'},
 {'course_id': 'doc_2_regle', 'course_name': 'Reglement de la certification'},
]

# ********** initiate json list dict
def initialize_db():
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
        # ********** Check if file is empty or invalid
        try:
            with open(DB_FILE, 'r') as file:
                json.load(file)
        except json.JSONDecodeError:
            # ********** If file is empty or invalid, write default structure
            with open(DB_FILE, 'w') as file:
                json.dump(default_db, file, indent=2)

def load_chat_history():
    try:
        with open(DB_FILE, 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        # If there's an error loading the file, initialize with empty list
        default_db = []
        with open(DB_FILE, 'w') as file:
            json.dump(default_db, file, indent=2)
        return default_db

def save_chat_history(topics_list):
    try:
        with open(DB_FILE, 'w') as file:
            json.dump(topics_list, file, indent=2)
    except Exception as e:
        st.error(f"Error saving chat history: {str(e)}")

def get_source_by_name(doc_name):
    for doc in LIST_DOCS:
        if doc["course_name"] == doc_name:
            return doc["course_id"]
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
        padding: 5px 6px;              /* Padding for a spacious look */
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
    
def format_topic_name(topic):
    topic_name = topic.get('topic', 'Unnamed Topic').replace("<b>", "").replace("</b>", "")
    created_at = topic.get('created_at', '')
    
    if created_at:
        return f"{topic_name}"
    return topic_name


def clear_current_chat(db_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_topic = f"New Chat_{timestamp}"

    # DEBUG
    # print("\n=== Debug Start ===")
    # print("Initial Messages:", st.session_state.get("messages", []))
    # print("Initial DB Data:", db_data)
    # print("New Topic:", new_topic)

    # Save current messages into history if not already saved
    if st.session_state.get("messages", []):
        current_topic = st.session_state.get("current_topic", "")
        if current_topic.startswith("New Chat_") or len(st.session_state["messages"]) == 0:
            print("New Chat topic is empty and will not be saved.")
        elif any(entry["topic"] == current_topic for entry in db_data):
            print("Current topic already saved in DB, skipping save.")
        else:
            db_data.append({
                "topic": current_topic,
                "messages": st.session_state["messages"],
                "created_at": datetime.now().isoformat(),
                "document": st.session_state.get("selected_doc", ""),
            })
            print("Saved current topic to DB.")

    # Reset session state for new chat
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.current_topic = new_topic

    # Avoid duplicate new topics in db_data
    existing_topics = [entry["topic"] for entry in db_data]
    if new_topic not in existing_topics:
        db_data.append({
            "topic": new_topic,
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "document": st.session_state.get("selected_doc", ""),
        })
        print(f"Added new topic: {new_topic}")
    else:
        print(f"Topic {new_topic} already exists in DB.")

    # DEBUG: Log updated states
    # print("Updated Messages:", st.session_state.get("messages", []))
    # print("Updated DB Data:", db_data)
    # print("=== Debug End ===\n")

    # Save changes to database
    save_chat_history(db_data)

    st.sidebar.success("Started a new chat while preserving history!")
    st.rerun()

def akabot_ui2():
    # ********** initiate db
    initialize_db()
    # ********** styling
    styling()
    # Load existing chat history
    db_data = load_chat_history()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Initialize session states with proper message structure
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = f"New Chat_{timestamp}"
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Rest of  UI code...
    text = plot_title()
    st.markdown(text, unsafe_allow_html=True)
    st.caption("Talk with your document")
    
    # Sidebar with topic navigation
    with st.sidebar:
        for _ in range(1):
            st.write("")
            
        st.markdown("""
            <div style="
                background-color: #ffffff;
                border-radius: 5px;
                padding: 3px;
                margin-bottom: 3px;
                text-align: center;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15);
            ">
                <h3 style="
                    color: #333333;
                    font-size: 12px;
                    font-weight: bold;
                    margin: 0;
                    text-transform: uppercase;
                    letter-spacing: 1.2px;
                ">Select Document</h3>
            </div>
        """, unsafe_allow_html=True)
         
        selected_doc = st.selectbox(
            "",
            options=[doc["course_name"] for doc in LIST_DOCS],
            index=0
        )
        
        for _ in range(1):
            st.write("")
        
        st.markdown("""
            <div style="
                background-color: #ffffff;
                border-radius: 5px;
                padding: 3px;
                margin-bottom: 3px;
                text-align: center;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15);
            ">
                <h3 style="
                    color: #333333;
                    font-size: 12px;
                    font-weight: bold;
                    margin: 0;
                    text-transform: uppercase;
                    letter-spacing: 1.2px;
                ">Chat Topic</h3>
            </div>
        """, unsafe_allow_html=True)

        topics = sorted(db_data, key=lambda x: x.get('created_at', ''), reverse=True)

        #print(f"\n\n Topics: {topics}")
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
            print(f"\n\n selected: {selected_base_topic}")
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
                print(f"\n\n topic data: {topic_data}")
                            
                if topic_data:
                    # Check if the key is 'message' or 'messages' 
                    if 'message' in topic_data:
                        st.session_state.messages = topic_data['message']  
                    elif 'messages' in topic_data:
                        st.session_state.messages = topic_data['messages']  
                    else:
                        st.session_state.messages = []  
                    
                    # Update current topic
                    st.session_state.current_topic = topic_data['topic']

            clear_button = st.button("Start New Chat", key="clear_button") 
            if clear_button:
                clear_current_chat(db_data)
            
            # tooltip for clarity  
            st.markdown("""
                <div style="margin: 10px 0; border-bottom: 1px solid #ccc;"></div>
                <div style="font-size: 0.8em; color: #666;">
                    Click 'Start New Chat' to begin a new topic doc.
                </div>
            """, unsafe_allow_html=True)
            
    # Main chat interface
    source = get_source_by_name(selected_doc)
    # st.write(st.session_state.messages)
    
    # Chat container
    with st.container(border=True, height=600):
        for message in st.session_state.messages:
            if isinstance(message, dict) and "type" in message and "content" in message:
                with st.chat_message(message["type"]):
                    st.markdown(message["content"])
                    if message["type"] == "ai" and "header_ref" in message and message["header_ref"]:
                        st.markdown(
                            f"""
                            <p style="color: gray; font-size: 12px;">
                                References: {message['header_ref'].replace("- ", "<br>-")}
                            </p>
                            """,
                            unsafe_allow_html=True
                        )
            elif isinstance(message, list): 
                for sub_message in message:  
                    if isinstance(sub_message, dict) and "type" in sub_message and "content" in sub_message:
                        with st.chat_message(sub_message["type"]):
                            st.markdown(sub_message["content"])
                            if sub_message["type"] == "ai" and "header_ref" in sub_message and sub_message["header_ref"]:
                                st.markdown(
                                    f"""
                                    <p style="color: gray; font-size: 12px;">
                                        References: {sub_message['header_ref'].replace("- ", "<br>-")}
                                    </p>
                                    """,
                                    unsafe_allow_html=True
                                )

        # Chat input
    with st.container():
            if prompt := st.chat_input("What would you like to know?"):
                # Add user message to display
                st.session_state.messages.append({"type": "human", "content": prompt})
                
                # Process RAG
                with st.spinner("Thinking..."):
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
                        
                        message, chat_history, topics, _, _ = process_query_with_context(
                            prompt,
                            source,
                            formatted_chat_history,
                            "" if st.session_state.current_topic.startswith("New Chat") else st.session_state.current_topic
                        )
                        # print(f"\n\n Topic: {topics}")
                        # Update session states
                        # Convert returned chat history to dictionary format
                        formatted_returned_history = []
                        header_ref_extracted = ""
                        for msg in chat_history:
                            # message object
                            if hasattr(msg, 'type'):  
                                msg_dict = {
                                    "type": msg.type,
                                    "content": msg.content
                                }
                                if msg.type == "ai":
                                    msg_dict["header_ref"] = getattr(msg, 'header_ref', '')
                                    header_ref_extracted = getattr(msg, 'header_ref', '')
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
                            "header_ref": st.session_state.chat_history[-1].get("header_ref", header_ref_extracted)
                        }
                        st.session_state.messages.append(ai_message)
                        
                        # Update database
                        if topics:  
                            db_data = [entry for entry in db_data if not entry["topic"].startswith("New Chat")]
                            topic_found = False 
                            
                            for entry in db_data:
                                if entry["topic"] == topics:
                                    # Update the existing topic
                                    entry["message"] = st.session_state.messages,
                                    entry["created_at"] = datetime.now().isoformat()
                                    entry["document"] = selected_doc
                                    topic_found = True
                                    break
                            
                            # If the topic was not found, create a new entry
                            if not topic_found:
                                db_data.append({
                                    "topic": topics,
                                    "message": st.session_state.messages,
                                    "created_at": datetime.now().isoformat(),
                                    "document": selected_doc
                                })
                            save_chat_history(db_data)
                            
                        st.rerun()
                        
                
if __name__ == '__main__':
    akabot_ui2()
    # st.write(st.session_state)