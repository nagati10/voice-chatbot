from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai  # NEW: Updated import
from google.genai import types  # NEW: For config types
from gtts import gTTS
import io
import os
import json
from datetime import datetime
import sqlite3
import base64

# Handle SpeechRecognition import with fallback
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
    print("✅ SpeechRecognition is available")
except Exception as e:
    print(f"⚠️ SpeechRecognition not available: {e}")
    SR_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# ========== FREE FOREVER CONFIG ==========
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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

# ========== CONVERSATION MEMORY ==========
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
    """Speech-to-text using Google Speech Recognition"""
    if not SR_AVAILABLE:
        return "Speech recognition is temporarily unavailable"
    
    try:
        r = sr.Recognizer()
        # Convert bytes to AudioData (16kHz, 16-bit)
        audio_data = sr.AudioData(audio_bytes, sample_rate=16000, sample_width=2)
        
        # Google Web Speech API - FREE
        text = r.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "I couldn't understand the audio"
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        return "Speech service temporarily unavailable"
    except Exception as e:
        print(f"Transcription error: {e}")
        return "Error processing audio"

# ========== AI RESPONSE (UPDATED) ==========
def get_ai_response(text, session_id="default"):
    """Get response from Gemini with conversation memory"""
    
    # Get conversation history
    history = get_conversation_history(session_id)
    
    # Build context from history
    context = ""
    for user_msg, ai_msg in reversed(history):
        context += f"User: {user_msg}\nAssistant: {ai_msg}\n"
    
    # Combine with current input
    prompt = f"""Previous conversation:
{context}

User: {text}

You are a helpful voice assistant. Respond conversationally in 1-3 sentences."""
    
    # Call Gemini API using NEW SDK
    try:
        # Initialize client with API key
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Create chat session with memory
        if not hasattr(get_ai_response, "chats"):
            get_ai_response.chats = {}
        
        if session_id not in get_ai_response.chats:
            get_ai_response.chats[session_id] = client.chats.create(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction="You are a helpful, friendly voice assistant. Keep responses conversational and brief (1-3 sentences).",
                    temperature=0.7
                )
            )
        
        # Send message and get response
        chat = get_ai_response.chats[session_id]
        response = chat.send_message(text)
        ai_text = response.text.strip()
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        print(f"Error type: {type(e).__name__}")
        # Fallback response
        ai_text = "I'm here to help! What would you like to know?"
    
    # Save to database memory
    save_conversation(session_id, text, ai_text)
    
    return ai_text

# ========== TEXT-TO-SPEECH ==========
def text_to_speech(text):
    """Convert text to speech using gTTS (FREE)"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        print(f"TTS error: {e}")
        # Return empty audio on error
        return b''

# ========== API ENDPOINTS ==========
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "active",
        "free": True,
        "speech_recognition": SR_AVAILABLE,
        "gemini_api": "2.5-flash",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/voice-chat', methods=['POST'])
def voice_chat():
    """Main endpoint - process voice to voice"""
    try:
        data = request.json
        audio_base64 = data.get('audio')
        session_id = data.get('session_id', 'default_session')
        
        if not audio_base64:
            return jsonify({
                "success": False,
                "error": "No audio data provided"
            }), 400
        
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
        print(f"Voice chat error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Please try again"
        }), 500

@app.route('/api/text-chat', methods=['POST'])
def text_chat():
    """Text-only endpoint"""
    try:
        data = request.json
        text = data.get('text')
        session_id = data.get('session_id', 'default_session')
        
        if not text:
            return jsonify({
                "success": False,
                "error": "No text provided"
            }), 400
        
        ai_response = get_ai_response(text, session_id)
        
        return jsonify({
            "success": True,
            "ai_response": ai_response,
            "session_id": session_id
        })
    except Exception as e:
        print(f"Text chat error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear conversation history for a session"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default_session')
        
        # Clear database history
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM conversations WHERE session_id=?", (session_id,))
        conn.commit()
        conn.close()
        
        # Clear chat session if exists
        if hasattr(get_ai_response, "chats") and session_id in get_ai_response.chats:
            del get_ai_response.chats[session_id]
        
        return jsonify({
            "success": True,
            "message": "History cleared"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice AI Chatbot</title>
        <meta http-equiv="refresh" content="0; url=/test">
    </head>
    <body>
        <p>Redirecting to test page...</p>
    </body>
    </html>
    """

# Simple test page
@app.route('/test')
def test_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice AI Test</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            .container { max-width: 600px; margin: 0 auto; }
            .btn { padding: 10px 20px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
            .btn:hover { background: #0056b3; }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .user { background: #e3f2fd; }
            .ai { background: #f5f5f5; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Voice AI Test Page</h1>
            <button class="btn" onclick="testText()">Test Text Chat</button>
            <div id="result"></div>
        </div>
        <script>
            async function testText() {
                const response = await fetch('/api/text-chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: 'Hello!', session_id: 'test' })
                });
                const data = await response.json();
                document.getElementById('result').innerHTML = 
                    `<div class="message ai"><strong>AI:</strong> ${data.ai_response}</div>`;
            }
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)