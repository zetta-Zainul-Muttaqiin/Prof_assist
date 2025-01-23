# ************ import
from flask import Flask, jsonify, request
import os
import time
from dotenv import load_dotenv
import threading
import time
from upload_doc import callRequest #delete
from Bilip_Ref_007 import ask_with_memory
from setup import LOGGER
from helpers.error_handle_helper import handle_error
# from quiz_builder import quiz_builder, quiz_editor

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

        # ************ req.body
        data = request.get_json()
        pdf_url = data.get("pdf_url")
        document_name = data.get("document_name")
        document_id = data.get("document_id")
        course_name = data.get("course_name")
        course_id = data.get("course_id")

        check_payload = (pdf_url and course_name and course_id and document_name and document_id)
        if check_payload:
            # ************ Call the function to process the PDF
            processPDFThread = threading.Thread(target=callRequest, args=(
                pdf_url, course_id, document_name, course_name, document_id))
            processPDFThread.start()
        
            return jsonify({"message": "PDF processing initiated successfully"}), 200

        return jsonify({"message": "One of request payload is empty."}), 400
    except (KeyError, IndexError, TypeError, ValueError) as error:
        error_message = str(error)
        LOGGER.error(f"error occured (Upload Doc): {error_message}")
        handle_error(error, 'finance_assist_chat')
        return jsonify({'error': True, 'message': error_message}), 500
    
    except Exception as un_error:
        error_message = str(un_error)
        LOGGER.error(f"An unexpected error occurred (Upload Doc): {error_message}")
        handle_error(un_error, 'finance_assist_chat')
        return jsonify({'error': True, 'message': error_message}), 404



# @app.route("/delete_pdf", methods=["POST"])
# def delete_pdf():
#     try:
#         data = request.get_json()
#         course_id = data.get("document_id")

#         # deletePDFThread = threading.Thread(target=delete, args=(course_id))
#         # deletePDFThread.start()
#         delete(course_id)

#         return jsonify({"message": "Document deleted"})
#     except Exception as error:
#         return jsonify({"error": str(error)}), 500


@app.route("/student_question", methods=["POST"])
def ask_question():
    try:
        start_time = time.time()
        # ********* get request data from BE
        data = request.get_json()
        question = data.get("question")
        course_id = data.get("course_id")
        chat_history = data.get("chat_history")
        topic = data.get("topic")

        print(f"data_get: {data}")
        # ******** send request data to engine
        message, chat_history, topic, tokens_out, tokens_in = ask_with_memory(
                question, course_id, chat_history, topic)
        print(f"done ask_with_memory")

        # ******** set response to BE
        response = {"message": message,
                    "topic": topic,
                    "chat_history": chat_history,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    }
        end_time = time.time()
        print(f'All time consumed: {end_time-start_time} seconds')
        return jsonify(response)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        error_message = str(error)
        LOGGER.error(f"error occured (Akadbot): {error_message}")
        handle_error(error, 'finance_assist_chat')
        return jsonify({'error': True, 'message': error_message}), 500
    
    except Exception as un_error:
        error_message = str(un_error)
        LOGGER.error(f"An unexpected error occurred (Akadbot): {error_message}")
        handle_error(un_error, 'finance_assist_chat')
        return jsonify({'error': True, 'message': error_message}), 404



# @app.route("/generate_quiz", methods=["POST"])
# def generate_quiz():
#     try:
#         data = request.get_json()
#         quiz_id = data.get("quiz_builder_id")
#         quiz_builder_description_id = data.get("quiz_builder_description_id")
#         source = data.get("source")
#         type_of_quiz = data.get("type_of_quiz")
#         number_of_quiz = data.get("number_of_quiz")
#         document_id = data.get("document_id")
#         lang = data.get("lang")
#         if type_of_quiz == "multiple":
#             multiple_option_amount = data.get("multiple_option_amount")
#         else:
#             multiple_option_amount = 0

#         # *************** Multi-threading quiz creation. Will send response from webhook
#         processQuiz = threading.Thread(target=quiz_builder, args=(quiz_id, quiz_builder_description_id, number_of_quiz, source, type_of_quiz,
#                                                                   multiple_option_amount, document_id, lang))
#         processQuiz.start()
#         # *************** End of Multi-threading quiz creation. Will send response from webhook

#         return jsonify({"message": "Quiz creation initiated successfully"})
#     except Exception as error:
#         return jsonify({"error": str(error)}), 500


# @app.route("/refresh_quiz", methods=["POST"])
# def refresh_quiz():
#     try:
#         data = request.get_json()
#         # nambah berapa number ke refresh
#         quiz_generated = data.get("quiz_generated")
#         source = data.get("source")
#         type_of_quiz = data.get("type_of_quiz")
#         document_id = data.get("document_id")
#         lang = data.get("lang")
#         if type_of_quiz == "multiple":
#             multiple_option_amount = data.get("multiple_option_amount")
#         else:
#             multiple_option_amount = 0

#         quiz, tokens_refresh_in, tokens_refresh_out, tokens_refresh_embbed = quiz_editor(source, type_of_quiz,
#                                                                                          multiple_option_amount,
#                                                                                          quiz_generated, document_id, lang)
#         response = {
#             "quiz_result": quiz,
#             "tokens_in": tokens_refresh_in,
#             "tokens_out": tokens_refresh_out,
#             "tokens_embbed": tokens_refresh_embbed
#         }
#         return jsonify(response)
#     except Exception as error:
#         return jsonify({"error": str(error)}), 500
    
if __name__ == "__main__":
    app.run(host='192.168.1.12', port=9874)