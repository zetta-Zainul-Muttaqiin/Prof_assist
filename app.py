# ************ import
from flask import Flask, jsonify, request, g
import os
import time
from dotenv import load_dotenv
import threading
import time
from upload_doc import callRequest, delete
from Bilip_Ref_007 import ask_with_memory
from quiz_builder import quiz_builder, quiz_editor

load_dotenv()
app = Flask(__name__)
chat_history = []
user_chat_histories = {}

@app.route("/ping", methods=["GET"])
def ping():
    return "ok"


@app.route("/version", methods=["GET"])
def version():
    return os.getenv('VERSION')


@app.route("/", methods=["GET"])
def index():
    # Accessing request-specific data
    get_history = request.headers.get("get_history")

    return f"Hello! Your history is: {get_history}"


@app.route("/process_pdf", methods=["POST"])
def process_pdf():
    try:

        astradb_token_key = os.getenv("ASTRADB_TOKEN_KEY")
        astradb_api_endpoint = os.getenv("ASTRADB_API_ENDPOINT")
        astradb_collection_name = os.getenv("ASTRADB_COLLECTION_NAME")
        url_webhook = os.getenv("URL_WEBHOOK")

        print(f"astradb_token_key === {astradb_token_key}")
        print(f"astradb_api_endpoint === {astradb_api_endpoint}")
        print(f"astradb_collection_name === {astradb_collection_name}")
        print(f"url_webhook === {url_webhook}")

        # ************ req.body
        data = request.get_json()
        pdf_url = data.get("pdf_url")
        document_name = data.get("document_name")
        document_id = data.get("document_id")
        book_name = data.get("book_name")

        # ************ Call the function to process the PDF
        processPDFThread = threading.Thread(target=callRequest, args=(
            pdf_url, document_name, book_name, document_id))
        processPDFThread.start()

        return jsonify({"message": "PDF processing initiated successfully"})
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/delete_pdf", methods=["POST"])
def delete_pdf():
    try:
        data = request.get_json()
        course_id = data.get("document_id")

        # deletePDFThread = threading.Thread(target=delete, args=(course_id))
        # deletePDFThread.start()
        delete(course_id)

        return jsonify({"message": "Document deleted"})
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/student_question", methods=["POST"])
def ask_question():
    try:
        start_time = time.time()
        # ********* get request data from BE
        data = request.get_json()
        question = data.get("question")
        source = data.get("source")
        chat_history = data.get("chat_history")
        topic = data.get("topic")

        print(f"data_get: {data}")
        # ******** send request data to engine
        message, chat_history, topic, tokens_out, tokens_in, tokens_embbed, header_ref_array = ask_with_memory(
                question, source, chat_history, topic)
        print(f"done ask_with_memory")
        print(f"Header Refrence: {header_ref_array}")

        # ******** Serialize the chat history to make it JSON-serializable
        serialized_chat_history = []
        index_header = 0
        print(len(chat_history))
        for msg in chat_history:
            # ******** get human input question to hisotry
            if msg.type == "human":
                serialized_msg = {"type": "human", "content": msg.content}
            else:
                # ******** else as ai output to hisotry with the reference
                serialized_msg = {"type": "ai", "content": msg.content, 'header_ref': header_ref_array[index_header]}
                index_header=index_header+1
            # ******** add history to array
            serialized_chat_history.append(serialized_msg)

        # ******** set response to BE
        response = {"message": message,
                    "topic": topic,
                    "chat_history": serialized_chat_history,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "tokens_embbed": tokens_embbed,
                    }
        end_time = time.time()
        print(f'All time consumed: {end_time-start_time} seconds')
        return jsonify(response)
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/generate_quiz", methods=["POST"])
def generate_quiz():
    try:
        data = request.get_json()
        quiz_id = data.get("quiz_builder_id")
        quiz_builder_description_id = data.get("quiz_builder_description_id")
        source = data.get("source")
        type_of_quiz = data.get("type_of_quiz")
        number_of_quiz = data.get("number_of_quiz")
        document_id = data.get("document_id")
        lang = data.get("lang")
        if type_of_quiz == "multiple":
            multiple_option_amount = data.get("multiple_option_amount")
        else:
            multiple_option_amount = 0

        # *************** Multi-threading quiz creation. Will send response from webhook
        processQuiz = threading.Thread(target=quiz_builder, args=(quiz_id, quiz_builder_description_id, number_of_quiz, source, type_of_quiz,
                                                                  multiple_option_amount, document_id, lang))
        processQuiz.start()
        # *************** End of Multi-threading quiz creation. Will send response from webhook

        return jsonify({"message": "Quiz creation initiated successfully"})
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/refresh_quiz", methods=["POST"])
def refresh_quiz():
    try:
        data = request.get_json()
        # nambah berapa number ke refresh
        quiz_generated = data.get("quiz_generated")
        source = data.get("source")
        type_of_quiz = data.get("type_of_quiz")
        document_id = data.get("document_id")
        lang = data.get("lang")
        if type_of_quiz == "multiple":
            multiple_option_amount = data.get("multiple_option_amount")
        else:
            multiple_option_amount = 0

        quiz, tokens_refresh_in, tokens_refresh_out, tokens_refresh_embbed = quiz_editor(source, type_of_quiz,
                                                                                         multiple_option_amount,
                                                                                         quiz_generated, document_id, lang)
        response = {
            "quiz_result": quiz,
            "tokens_in": tokens_refresh_in,
            "tokens_out": tokens_refresh_out,
            "tokens_embbed": tokens_refresh_embbed
        }
        return jsonify(response)
    except Exception as error:
        return jsonify({"error": str(error)}), 500
    
if __name__ == "__main__":
    app.run(host='192.168.1.14', port=9874)