from flask import Flask, render_template, request, jsonify, session
from main import output_report
import os
import google.generativeai as genai

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'd254d2b275ae0877f0ae384167251e64'

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure Google Gemini API
genai.configure(api_key="AIzaSyBQRbnr3lkX2Iocnd-Jx2ljvNt_RcZm5Sc")

# Instantiate Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")

# Base prompt template for chatbot
base_prompt = """
As a medical chatbot, you assist users by providing accurate, empathetic, and concise responses based on their medical queries.
Consider the following:
- Respond in a professional, friendly, and supportive tone.
- Address each query with a focus on medical accuracy and user reassurance.
- Use plain language to explain complex medical terms and processes as needed.

Below is the conversation history between you and the user. Use this context to understand their concerns and respond accurately.
If additional medical terms or details are requested, provide brief definitions and suggestions based on general medical guidelines.

User Queries and Conversation History:
"""

# Function to generate chatbot response
def generate_message(query, conversation_history):
    conversation_context = "\n".join(f"User: {item['user']}\nBot: {item['bot']}" for item in conversation_history)
    full_prompt = f"{base_prompt}\n{conversation_context}\nUser: {query}\nBot:"

    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=1024,
            top_p=1
        )
    )
    return response.text

# Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main')
def main_page():  # <-- Renamed to avoid conflict
    return render_template('main.html')

@app.route('/report', methods=['POST'])
def report():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    custom_text = request.form.get("custom_text")
    if not custom_text:
        return jsonify({"error": "Please provide prescription text."}), 400

    report_data = output_report(img_path, custom_text)
    if report_data is None:
        return jsonify({"error": "Report generation failed. Please check your inputs."}), 400

    return render_template("result.html", report_data=report_data)

@app.route('/chatbot')
def chatbot():
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    return render_template('chatbot.html')

@app.route('/ask_chatbot', methods=['POST'])
def ask_chatbot():
    user_question = request.form.get("question")
    if not user_question:
        return jsonify({"error": "No question provided."}), 400

    conversation_history = session.get('conversation_history', [])
    response = generate_message(user_question, conversation_history)

    conversation_history.append({'user': user_question, 'bot': response})
    if len(conversation_history) > 10:
        conversation_history.pop(0)

    session['conversation_history'] = conversation_history
    session.modified = True

    return jsonify({"response": response})

import easyocr
@app.route('/analyze_medicine', methods=['GET', 'POST'])
def analyze_medicine():
    """Handle transcription from uploaded image using EasyOCR."""
    if request.method == 'GET':
        return render_template('analyze_medicine_upload.html')

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    # Initialize EasyOCR
    reader = easyocr.Reader(['en'])  # You can add more languages if needed
    results = reader.readtext(img_path)

    # Check if results exist and are valid
    if not results:
        return jsonify({"error": "No text could be extracted from the image."}), 400

    try:
        # Join all detected text from the OCR results
        extracted_text = " ".join([text[1] for text in results])
    except Exception as e:
        return jsonify({"error": f"OCR parsing error: {str(e)}"}), 500

    if not extracted_text.strip():
        return jsonify({"error": "Extracted text is empty."}), 400

    return render_template("analyze_medicine_result.html", transcription=extracted_text, img_path=file.filename)

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=5001)
