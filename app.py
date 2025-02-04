# *************** IMPORT LIBRARY ***************
import os
import threading
from flask                          import Flask, jsonify, request

from setup                          import LOGGER


from engine.chat_akadbot            import ask_with_memory
from engine.process_doc_akadbot     import (
                                        upload_akadbot_document, 
                                        delete_documents_id,
                                        delete_list_document,
                                    )
# from quiz_builder import quiz_builder, quiz_editor

from helpers.error_handle_helpers    import handle_error

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


# *************** Endpoint route for upload a document to akadbot
@app.route("/process_pdf", methods=["POST"])
def process_pdf():
    """
    Handles the processing of a PDF document by extracting necessary metadata
    from the request body and initiating a separate thread to process the document.

    Request (JSON):
        {
            "pdf_url": URL of the PDF document
            "document_name": Name of the document
            "document_id": Unique ID for the document
            "course_name": Name of the course that document uploaded
            "course_id": Unique ID for the course
        }

    Returns:
        Response (JSON):
            - Success:
                {"message": "PDF processing initiated successfully"}, HTTP 200
            - Bad Request (400):{
                "error": True, 
                "message": "Bad Request: missing required input data: {'pdf_url', 'course_id', 'document_name', 'document_id'}"
                }, HTTP 400
            - Specific Errors: {"error": True, "message": error_message}, HTTP 501
            - Unexpected Errors: {"error": True, "message": error_message}, HTTP 500
    """
    try:
        # *************** Parse request body
        data = request.get_json()
        pdf_url = data.get("pdf_url")
        document_name = data.get("document_name")
        document_id = data.get("document_id")
        course_name = data.get("course_name")
        course_id = data.get("course_id")

        # *************** Validate required input fields
        check_payload = all([pdf_url, course_name, course_id, document_name, document_id])
        if check_payload:
            # *************** Start a new thread for processing the PDF
            processPDFThread = threading.Thread(target=upload_akadbot_document, args=(
                pdf_url, course_id, document_name, course_name, document_id))
            processPDFThread.start()
            
            return jsonify({"message": "PDF processing initiated successfully"}), 200
        
        # *************** Handle missing required fields
        return jsonify({"error": True, "message": "Bad Request: missing required input data: {pdf_url, course_id, course_name, document_name, document_id}"}), 400
    
    # *************** Handle specific known errors
    except (KeyError, IndexError, TypeError, ValueError) as error:
        error_message = str(error)
        LOGGER.error(f"Error occurred (Akadbot): {error_message}")
        handle_error(error, 'chat_akadbot')
        return jsonify({"error": True, "message": error_message}), 501
    
    # *************** Handle unexpected errors
    except Exception as un_error:
        error_message = str(un_error)
        LOGGER.error(f"An unexpected error occurred (Akadbot): {error_message}")
        handle_error(un_error, 'chat_akadbot')
        return jsonify({"error": True, "message": error_message}), 500

# *************** Endpoint route for delete only one pdf expected
@app.route("/delete_one_pdf", methods=["POST"])
def delete_pdf():
    """
    Handles the deletion of a single PDF document by receiving the document ID
    from the request body and removing it from the system.

    Request (JSON):
        {
            "document_id": Unique ID for the document want to delete
        }

    Returns:
        Response (JSON):
            - Success: {"message": "Document ID '{document_id}' deleted successfully!"}, HTTP 200
            - Warning: {"message": "Document with ID '{document_id}' not found."}, HTTP 200
            - Bad Request: {"error": True, "message": "Bad Request: missing required input data: {'document_id'}"}, HTTP 400
            - Specific Errors: {"error": True, "message": error_message}, HTTP 501
            - Unexpected Errors: {"error": True, "message": error_message}, HTTP 500
    """
    try:
        # ************ Parse request body
        data = request.get_json()
        document_id = data.get("document_id")
        
        # ************ Validate required input field
        if document_id:
            status = delete_documents_id(document_id)

            # ************ Return delete status with code response
            return jsonify({"message": status}), 200
        
        return jsonify({"error": True, "message": "Bad Request: missing required input data: {'document_id'}"}), 400
    
    # *************** Handle specific known errors
    except (KeyError, IndexError, TypeError, ValueError) as error:
        error_message = str(error)
        LOGGER.error(f"Error occurred (Delete Document): {error_message}")
        handle_error(error, 'delete_one_document')
        return jsonify({"error": True, "message": error_message}), 501
    
    # *************** Handle unexpected errors
    except Exception as un_error:
        error_message = str(un_error)
        LOGGER.error(f"An unexpected error occurred (Delete Document): {error_message}")
        handle_error(un_error, 'delete_one_document')
        return jsonify({"error": True, "message": error_message}), 500

# *************** Endpoint route for delete many pdf in list of document_id 
@app.route("/delete_many_pdf", methods=["POST"])
def delete_many_pdf():
    """
    Handles the deletion of multiple PDF documents by receiving a list of document IDs
    from the request body and removing them from the system.

    Request (JSON):
        {
            "list_document_id": list of Unique ID for the documents want to deleted
        }

    Returns:
        Response (JSON):
            - Success: {"message": "Documents with IDs '{list_document}' deleted successfully!"}, HTTP 200
            - Warning: {"message": "Documents with IDs '{list_document}' not found."}, HTTP 200
            - Bad Request: {"error": True, "message": "Bad Request: missing required input data: {'list_document_id'}"}, HTTP 400
            - Specific Errors: {"error": True, "message": error_message}, HTTP 501
            - Unexpected Errors: {"error": True, "message": error_message}, HTTP 500
    """
    try:
        # ************ Parse request body
        data = request.get_json()
        list_documents = data.get("list_document_id")
        
        # ************ Validate required input field
        if list_documents:
            status = delete_list_document(list_documents)

            # ************ Return delete status with code response
            return jsonify({"message": status}), 200
        
        return jsonify({"error": True, "message": "Bad Request: missing required input data: {'list_document_id'}"}), 400
    
    # *************** Handle specific known errors
    except (KeyError, IndexError, TypeError, ValueError) as error:
        error_message = str(error)
        LOGGER.error(f"Error occurred (Delete Document): {error_message}")
        handle_error(error, 'delete_many_document')
        return jsonify({"error": True, "message": error_message}), 501
    
    # *************** Handle unexpected errors
    except Exception as un_error:
        error_message = str(un_error)
        LOGGER.error(f"An unexpected error occurred (Delete Document): {error_message}")
        handle_error(un_error, 'delete_many_document')
        return jsonify({"error": True, "message": error_message}), 500

# *************** Endpoint route for question answering to akadbot document
@app.route("/student_question", methods=["POST"])
def ask_question():
    """
    Handles student queries related to a course using the Akadbot engine.

    This endpoint receives a student's question along with the course ID, chat history, 
    and topic, then processes the question using the Akadbot engine with memory 
    support and returns a response.

    Request (JSON):
        {
            "question": Student's question
            "course_id": Identifier for the course
            "chat_history": (Optional) Chat history for context
            "topic": (Optional) Topic to refine the query
        }

    Returns:
        Response (JSON):
        - Success:
            {
                "message": AI-generated response of question,
                "topic": Generated topic response of first question,
                "chat_history": Updated conversation history,
                "tokens_in": Number of input tokens used,
                "tokens_out": Number of output tokens generated
            }, HTTP 200
        - Bad Request: {"error": True, "message": "Bad Request: missing required input data: {question, course_id}"}, HTTP 400
        - Specific Errors: {"error": True, "message": error_message}, HTTP 501
        - Unexpected Errors: {"error": True, "message": error_message}, HTTP 500
    """
    try:
        # *************** Get request data from BE in JSON format
        data = request.get_json()
        question = data.get("question")
        course_id = data.get("course_id")
        chat_history = data.get("chat_history")
        topic = data.get("topic")

        if data and question and course_id:

            # *************** Send request data to engine as input
            response = ask_with_memory(
                    question=question, 
                    course_id=course_id, 
                    chat_history=chat_history, 
                    topic=topic
            )

            # *************** Send response to backend
            return jsonify(response), 200
        
        # *************** Handle missing required input of student_question
        return jsonify({"error": True, "message": "Bad Request: missing required input data: {question, course_id}"}), 400
    
    # *************** Handling specific error
    except (KeyError, IndexError, TypeError, ValueError) as error:
        error_message = str(error)
        LOGGER.error(f"error occured (Akadbot): {error_message}")
        handle_error(error, 'chat_akadbot')
        return jsonify({"error": True, "message": error_message}), 501
    
    # *************** Handle unexpected error
    except Exception as un_error:
        error_message = str(un_error)
        LOGGER.error(f"An unexpected error occurred (Akadbot): {error_message}")
        handle_error(un_error, 'chat_akadbot')
        return jsonify({"error": True, "message": error_message}), 500



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
    app.run(host='192.168.1.6', port=9874)