from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from gtts import gTTS
import io
import os
from datetime import datetime
import sqlite3
import base64

# Handle SpeechRecognition import with fallback
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
    print("✅ SpeechRecognition 3.14.4 is available")
except Exception as e:
    print(f"⚠️ SpeechRecognition not available: {e}")
    SR_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# ========== CONFIG ==========
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_FILE = "conversations.db"

if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not set!")
else:
    print("✅ GEMINI_API_KEY is configured")

# Initialize database
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
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)",
                  (session_id, datetime.now(), user_input, ai_response))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")

def get_conversation_history(session_id, limit=5):
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("SELECT user_input, ai_response FROM conversations WHERE session_id=? ORDER BY timestamp DESC LIMIT ?", 
                  (session_id, limit))
        history = c.fetchall()
        conn.close()
        return history
    except Exception as e:
        print(f"Database error: {e}")
        return []

# ========== SPEECH-TO-TEXT ==========
def transcribe_audio(audio_bytes):
    """Speech-to-text using Google Speech Recognition (free)"""
    if not SR_AVAILABLE:
        return "Speech recognition is temporarily unavailable. Please use text mode."
    
    try:
        r = sr.Recognizer()
        
        # Convert bytes to AudioData
        audio_data = sr.AudioData(audio_bytes, sample_rate=16000, sample_width=2)
        
        # Use Google Speech Recognition (FREE)
        text = r.recognize_google(audio_data, language='en-US')
        print(f"✅ Transcribed: {text}")
        return text
        
    except sr.UnknownValueError:
        print("⚠️ Could not understand audio")
        return "I couldn't understand the audio. Please try speaking more clearly."
    except sr.RequestError as e:
        print(f"❌ Speech recognition service error: {e}")
        return "Speech service is temporarily unavailable. Please try again."
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return "Error processing audio. Please try again."

# ========== AI RESPONSE ==========
def get_ai_response(text, session_id="default"):
    """Get response from Gemini 2.5 Flash with conversation memory"""
    
    if not GEMINI_API_KEY:
        return "AI service is not configured. Please set GEMINI_API_KEY environment variable."
    
    # Get conversation history
    history = get_conversation_history(session_id)
    
    # Build context from history
    context = ""
    if history:
        for user_msg, ai_msg in reversed(history):
            context += f"User: {user_msg}\nAssistant: {ai_msg}\n"
        context += "\n"
    
    # Combine with current input
    full_prompt = f"""{context}User: {text}

Respond conversationally in 1-3 sentences as a helpful voice assistant."""
    
    # Call Gemini API using new SDK
    try:
        # Initialize client
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Generate content with Gemini 2.5 Flash
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt
        )
        
        # Extract text from response
        if response and response.text:
            ai_text = response.text.strip()
            print(f"✅ Gemini 2.5 response: {ai_text[:50]}...")
        else:
            print("⚠️ Empty response from Gemini")
            ai_text = "I heard you, but I'm having trouble responding right now. Could you rephrase that?"
        
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        
        # Fallback response
        ai_text = "I'm having trouble connecting to my AI service right now. Please try again in a moment."
    
    # Save to memory
    save_conversation(session_id, text, ai_text)
    
    return ai_text

# ========== TEXT-TO-SPEECH ==========
def text_to_speech(text):
    """Convert text to speech using gTTS (free)"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        audio_bytes = audio_buffer.read()
        print(f"✅ Generated TTS audio: {len(audio_bytes)} bytes")
        return audio_bytes
        
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return b''

# ========== API ENDPOINTS ==========
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "active",
        "free": True,
        "speech_recognition": SR_AVAILABLE,
        "speech_recognition_version": "3.14.4",
        "gemini_api": "gemini-2.5-flash",
        "gemini_configured": bool(GEMINI_API_KEY),
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/voice-chat', methods=['POST'])
def voice_chat():
    """Main endpoint - process voice to voice"""
    try:
        data = request.json
        
        if not data:
            return jsonify({"success": False, "error": "No JSON data received"}), 400
        
        audio_base64 = data.get('audio')
        session_id = data.get('session_id', 'default_session')
        
        if not audio_base64:
            return jsonify({"success": False, "error": "No audio data provided"}), 400
        
        print(f"📥 Received audio: {len(audio_base64)} chars base64")
        
        # Convert base64 to bytes
        try:
            audio_bytes = base64.b64decode(audio_base64)
            print(f"✅ Decoded audio: {len(audio_bytes)} bytes")
        except Exception as e:
            return jsonify({"success": False, "error": f"Invalid base64 audio: {str(e)}"}), 400
        
        # Step 1: Speech to Text
        print("🎤 Step 1: Transcribing audio...")
        user_text = transcribe_audio(audio_bytes)
        
        if not user_text or user_text.startswith("Speech") or user_text.startswith("Error") or user_text.startswith("I couldn't"):
            return jsonify({"success": False, "error": user_text, "user_text": user_text}), 400
        
        # Step 2: Get AI Response
        print("🤖 Step 2: Getting AI response...")
        ai_response = get_ai_response(user_text, session_id)
        
        # Step 3: Text to Speech
        print("🔊 Step 3: Generating speech...")
        audio_response = text_to_speech(ai_response)
        
        if not audio_response:
            return jsonify({
                "success": False,
                "error": "Failed to generate audio response",
                "user_text": user_text,
                "ai_response": ai_response
            }), 500
        
        audio_base64_response = base64.b64encode(audio_response).decode('utf-8')
        print(f"✅ Response audio: {len(audio_base64_response)} chars base64")
        
        return jsonify({
            "success": True,
            "user_text": user_text,
            "ai_response": ai_response,
            "audio": audio_base64_response,
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"❌ Voice chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e), "message": "Internal server error"}), 500

@app.route('/api/text-chat', methods=['POST'])
def text_chat():
    """Text-only endpoint"""
    try:
        data = request.json
        
        if not data:
            return jsonify({"success": False, "error": "No JSON data received"}), 400
        
        text = data.get('text')
        session_id = data.get('session_id', 'default_session')
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        print(f"💬 Text chat: {text}")
        
        # Get AI response
        ai_response = get_ai_response(text, session_id)
        print(f"✅ AI response: {ai_response}")
        
        return jsonify({
            "success": True,
            "ai_response": ai_response,
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"❌ Text chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear conversation history for a session"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default_session')
        
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM conversations WHERE session_id=?", (session_id,))
        deleted = c.rowcount
        conn.commit()
        conn.close()
        
        print(f"🗑️ Cleared {deleted} messages for session {session_id}")
        
        return jsonify({
            "success": True,
            "message": f"Cleared {deleted} messages",
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"❌ Clear history error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/')
def index():
    """Homepage with status"""
    sr_status = "✅ Available (v3.14.4)" if SR_AVAILABLE else "⚠️ Unavailable"
    gemini_status = "✅ Configured" if GEMINI_API_KEY else "❌ Not Configured"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice AI Chatbot Backend</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                padding: 40px;
                max-width: 900px;
                margin: 0 auto;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }}
            h1 {{ color: #333; margin-bottom: 10px; }}
            .status {{
                background: #e8f5e9;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                border-left: 5px solid #4caf50;
            }}
            .status.warning {{ background: #fff3cd; border-left-color: #ffc107; }}
            .endpoint {{
                background: #f5f5f5;
                padding: 15px;
                margin: 15px 0;
                border-left: 4px solid #2196F3;
                font-family: 'Courier New', monospace;
                border-radius: 5px;
            }}
            button {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 50px;
                font-size: 16px;
                cursor: pointer;
                margin: 10px 5px;
                transition: transform 0.2s;
            }}
            button:hover {{ transform: scale(1.05); }}
            #testResult {{
                margin-top: 20px;
                padding: 15px;
                border-radius: 10px;
                background: #f5f5f5;
                min-height: 60px;
            }}
            .success {{ color: #4caf50; }}
            .error {{ color: #f44336; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Voice AI Chatbot Backend</h1>
            <p style="color: #666;">Real-time Speech-to-Speech AI Assistant</p>
            
            <div class="status">
                <p><strong>Status:</strong> ✅ Active</p>
                <p><strong>Cost:</strong> $0/month</p>
            </div>
            
            <div class="status {'warning' if not GEMINI_API_KEY else ''}">
                <p><strong>Gemini 2.5 Flash:</strong> {gemini_status}</p>
            </div>
            
            <div class="status {'' if SR_AVAILABLE else 'warning'}">
                <p><strong>Speech Recognition:</strong> {sr_status}</p>
            </div>
            
            <h2>API Endpoints</h2>
            <div class="endpoint"><strong>POST</strong> /api/voice-chat</div>
            <div class="endpoint"><strong>POST</strong> /api/text-chat</div>
            <div class="endpoint"><strong>POST</strong> /api/clear-history</div>
            <div class="endpoint"><strong>GET</strong> /health</div>
            
            <h3>Test API</h3>
            <button onclick="testTextChat()">Test Text Chat</button>
            <button onclick="testHealth()">Test Health</button>
            
            <div id="testResult"></div>
            
            <script>
                async function testTextChat() {{
                    const result = document.getElementById('testResult');
                    result.innerHTML = '<div style="color: orange;">Testing...</div>';
                    
                    try {{
                        const response = await fetch('/api/text-chat', {{
                            method: 'POST',
                            headers: {{'Content-Type': 'application/json'}},
                            body: JSON.stringify({{text: 'Hello!', session_id: 'test_' + Date.now()}})
                        }});
                        
                        const data = await response.json();
                        
                        if (data.success) {{
                            result.innerHTML = `<div class="success"><strong>✅ SUCCESS!</strong><br>Response: "${{data.ai_response}}"</div>`;
                        }} else {{
                            result.innerHTML = `<div class="error"><strong>❌ Error:</strong> ${{data.error}}</div>`;
                        }}
                    }} catch (error) {{
                        result.innerHTML = `<div class="error"><strong>❌ Network Error:</strong> ${{error.message}}</div>`;
                    }}
                }}
                
                async function testHealth() {{
                    const result = document.getElementById('testResult');
                    result.innerHTML = '<div style="color: orange;">Checking health...</div>';
                    
                    try {{
                        const response = await fetch('/health');
                        const data = await response.json();
                        
                        result.innerHTML = `
                            <div class="success">
                                <strong>✅ Healthy</strong><br>
                                Speech Recognition: ${{data.speech_recognition ? '✅ v' + data.speech_recognition_version : '❌'}}<br>
                                Gemini: ${{data.gemini_configured ? '✅' : '❌'}}<br>
                                Model: ${{data.gemini_api}}
                            </div>
                        `;
                    }} catch (error) {{
                        result.innerHTML = `<div class="error"><strong>❌ Failed:</strong> ${{error.message}}</div>`;
                    }}
                }}
                
                window.addEventListener('load', testHealth);
            </script>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Voice Chatbot Backend on port {port}")
    print(f"🐍 Python: {os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}")
    print(f"📊 Speech Recognition: {SR_AVAILABLE}")
    print(f"🤖 Gemini: {'Configured (2.5 Flash)' if GEMINI_API_KEY else 'NOT CONFIGURED'}")
    app.run(host='0.0.0.0', port=port, debug=False)