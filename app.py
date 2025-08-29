import os
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

import google.generativeai as genai
from datetime import datetime

# --- Initialize Flask app ---
app = Flask(__name__)
CORS(app)

# Secret key for session handling (required for session management)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey123")

# Load environment variable for Gemini API
GENAI_API_KEY = os.getenv("API_KEY")
if not GENAI_API_KEY:
    raise ValueError("Gemini API key not found. Set it in your Render environment variables.")

genai.configure(api_key=GENAI_API_KEY)

# --- Use embeddings folder inside repo ---
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "embeddings")

chroma_settings = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
    persist_directory=PERSIST_DIRECTORY
)

chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY, settings=chroma_settings)

huggingface_model_path = os.path.join(os.path.dirname(__file__), "hugging_face")

vector_db = Chroma(
    client=chroma_client,
    collection_name="hr_faqs",
    embedding_function=HuggingFaceEmbeddings(model_name=huggingface_model_path)
)

# Feedback & analytics storage
feedback_data = []
analytics_data = {
    'total_queries': 0,
    'positive_feedback': 0,
    'negative_feedback': 0,
    'common_questions': {},
    'unanswered_questions': []
}

@app.before_request
def check_session_age():
    """Reset session if inactive for over 1 hour"""
    last_active = session.get('last_active')
    if last_active:
        time_elapsed = datetime.now() - datetime.fromisoformat(last_active)
        if time_elapsed.total_seconds() > 3600:
            session.pop('conversation_history', None)
    session['last_active'] = datetime.now().isoformat()

@app.route('/')
def index():
    return "HR Assistant Chatbot is running!"

@app.route('/query', methods=['POST'])
def chat():
    session.pop('conversation_history', None)
    session['conversation_history'] = []

    data = request.get_json()
    user_query = data.get('query')
    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    try:
        session['conversation_history'].append(f"User: {user_query}")

        results = vector_db.similarity_search(user_query, k=3)
        kb_context = "\n".join([r.page_content for r in results]) if results else "No relevant knowledge found."

        prompt = f"""
        You are a highly intelligent and helpful HR assistant.
        Provide answers based on the provided knowledge base context.

        Knowledge Base Context:
        {kb_context}

        Current Question:
        User: {user_query}

        Your answer should be direct, accurate, and conversational.
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        bot_response = response.text

        session['conversation_history'].append(f"Bot: {bot_response}")

        return jsonify({"response": bot_response, "knowledge_matched": bool(results)})

    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({"error": "Something went wrong!"}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    feedback_type = data.get('feedback_type')
    feedback_comment = data.get('comment', '')

    if not feedback_type:
        return jsonify({"error": "Feedback type is required"}), 400

    try:
        feedback_data.append({
            'feedback_type': feedback_type,
            'comment': feedback_comment,
            'timestamp': datetime.now().isoformat()
        })

        if feedback_type == 'positive':
            analytics_data['positive_feedback'] += 1
        elif feedback_type == 'negative':
            analytics_data['negative_feedback'] += 1

        return jsonify({"message": "Feedback submitted successfully"})
    except Exception as e:
        app.logger.error(f"Error submitting feedback: {e}")
        return jsonify({"error": "Failed to submit feedback"}), 500

@app.route('/analytics', methods=['GET'])
def get_analytics():
    try:
        total_feedback = analytics_data['positive_feedback'] + analytics_data['negative_feedback']
        feedback_ratio = analytics_data['positive_feedback'] / total_feedback if total_feedback > 0 else 0
        top_questions = sorted(analytics_data['common_questions'].items(), key=lambda x: x[1], reverse=True)[:10]

        return jsonify({
            "total_queries": analytics_data['total_queries'],
            "feedback_ratio": feedback_ratio,
            "top_questions": top_questions,
        })
    except Exception as e:
        app.logger.error(f"Error getting analytics: {e}")
        return jsonify({"error": "Failed to get analytics"}), 500

@app.route('/clear', methods=['POST'])
def clear_session():
    session.pop('conversation_history', None)
    return jsonify({"message": "Conversation history cleared."})

if __name__ == "__main__":
    from gunicorn.app.base import BaseApplication
    from gunicorn.six import iteritems

    class GunicornApp(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load(self):
            return self.application

        def load_config(self):
            for key, value in iteritems(self.options):
                if key in self.cfg.settings and value is not None:
                    self.cfg.set(key.lower(), value)

    options = {
        'bind': f"0.0.0.0:{os.getenv('PORT', 5000)}",
        'workers': 2,
    }
    GunicornApp(app, options).run()
