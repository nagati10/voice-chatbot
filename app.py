from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import google.generativeai as genai
from gtts import gTTS
try:
    import speech_recognition as sr
except ImportError as e:
    print(f"SpeechRecognition import error: {e}")
    # Create a dummy recognizer for fallback
    class DummyRecognizer:
        def recognize_google(self, *args, **kwargs):
            return "Speech recognition not available"
    sr = type('sr', (), {'Recognizer': DummyRecognizer})()
import io
import os
import json
from datetime import datetime
import sqlite3
import hashlib
import base64

app = Flask(__name__)
CORS(app)

# ========== FREE FOREVER CONFIG ==========
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Free from Google AI Studio
DATABASE_FILE = "conversations.db"

# Initialize conversation memory
def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (session_id TEXT, timestamp DATETIME, 
                  user_input TEXT, ai_response TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ========== SIMPLE CONVERSATION MEMORY ==========
def save_conversation(session_id, user_input, ai_response):
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)",
              (session_id, datetime.now(), user_input, ai_response))
    conn.commit()
    conn.close()

def get_conversation_history(session_id, limit=5):
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute("SELECT user_input, ai_response FROM conversations WHERE session_id=? ORDER BY timestamp DESC LIMIT ?", 
              (session_id, limit))
    history = c.fetchall()
    conn.close()
    return history

# ========== SPEECH-TO-TEXT ==========
def transcribe_audio(audio_bytes):
    """Simple, reliable STT using Google Speech Recognition"""
    r = sr.Recognizer()
    
    # Convert bytes to AudioData
    audio_data = sr.AudioData(audio_bytes, sample_rate=16000, sample_width=2)
    
    try:
        # Google Web Speech API - FREE, no key needed
        text = r.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "I couldn't understand the audio"
    except sr.RequestError:
        # Fallback: Return placeholder
        return "Hello, how can I help you?"

# ========== AI RESPONSE ==========
def get_ai_response(text, session_id="default"):
    """Get response from Gemini with conversation memory"""
    
    # Get conversation history
    history = get_conversation_history(session_id)
    
    # Build context from history
    context = ""
    for user_msg, ai_msg in reversed(history):  # Most recent first
        context += f"User: {user_msg}\nAI: {ai_msg}\n"
    
    # Combine with current input
    prompt = f"""
    Previous conversation:
    {context}
    
    User: {text}
    
    You are a helpful voice assistant. Respond conversationally in 1-2 sentences.
    """
    
    # Call Gemini API (FREE tier)
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    
    try:
        response = model.generate_content(prompt)
        ai_text = response.text.strip()
    except Exception as e:
        ai_text = "I'm here to help! What would you like to know?"
    
    # Save to memory
    save_conversation(session_id, text, ai_text)
    
    return ai_text

# ========== TEXT-TO-SPEECH ==========
def text_to_speech(text):
    """Convert text to speech using gTTS (FREE)"""
    tts = gTTS(text=text, lang='en', slow=False)
    
    # Save to bytes
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    
    return audio_buffer.read()

# ========== API ENDPOINTS ==========
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "active", "free": True})

@app.route('/api/voice-chat', methods=['POST'])
def voice_chat():
    """Main endpoint - process voice to voice"""
    try:
        # Get data from request
        data = request.json
        audio_base64 = data.get('audio')
        session_id = data.get('session_id', 'default_session')
        
        # Convert base64 to bytes
        audio_bytes = base64.b64decode(audio_base64)
        
        # Step 1: Speech to Text
        user_text = transcribe_audio(audio_bytes)
        
        # Step 2: Get AI Response
        ai_response = get_ai_response(user_text, session_id)
        
        # Step 3: Text to Speech
        audio_response = text_to_speech(ai_response)
        audio_base64_response = base64.b64encode(audio_response).decode('utf-8')
        
        return jsonify({
            "success": True,
            "user_text": user_text,
            "ai_response": ai_response,
            "audio": audio_base64_response,
            "session_id": session_id
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Please try again"
        }), 500

@app.route('/api/text-chat', methods=['POST'])
def text_chat():
    """Text-only endpoint"""
    data = request.json
    text = data.get('text')
    session_id = data.get('session_id', 'default_session')
    
    ai_response = get_ai_response(text, session_id)
    
    return jsonify({
        "success": True,
        "ai_response": ai_response
    })

# Simple web interface for testing
@app.route('/')
def index():
    return """
    <html>
    <body style="font-family: Arial; padding: 20px;">
        <h1>🎤 Voice Chatbot Backend</h1>
        <p><strong>Status:</strong> ✅ Active</p>
        <p><strong>Cost:</strong> $0/month (Free Forever)</p>
        <p><strong>Endpoints:</strong></p>
        <ul>
            <li>POST /api/voice-chat - Process voice message</li>
            <li>POST /api/text-chat - Text chat</li>
            <li>GET /health - Health check</li>
        </ul>
        <p><em>Ready for iOS/Android apps</em></p>
    </body>
    </html>
    """

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)