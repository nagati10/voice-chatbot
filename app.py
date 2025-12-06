from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from gtts import gTTS
import io
import os
from datetime import datetime
import sqlite3
import base64
from pydub import AudioSegment
import tempfile
from langdetect import detect, detect_langs, DetectorFactory

# Set seed for consistent language detection
DetectorFactory.seed = 0

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

# Language code mapping for gTTS
LANGUAGE_MAPPING = {
    'en': 'en',      # English
    'es': 'es',      # Spanish
    'fr': 'fr',      # French
    'de': 'de',      # German
    'it': 'it',      # Italian
    'pt': 'pt',      # Portuguese
    'ru': 'ru',      # Russian
    'ja': 'ja',      # Japanese
    'ko': 'ko',      # Korean
    'zh-cn': 'zh',   # Chinese Simplified
    'zh-tw': 'zh-tw', # Chinese Traditional
    'ar': 'ar',      # Arabic
    'hi': 'hi',      # Hindi
}

# Speech recognition languages
SPEECH_LANGUAGES = {
    'en': 'en-US',
    'es': 'es-ES',
    'fr': 'fr-FR',
    'de': 'de-DE',
    'it': 'it-IT',
    'pt': 'pt-PT',
    'ru': 'ru-RU',
    'ja': 'ja-JP',
    'ko': 'ko-KR',
    'zh': 'zh-CN',
    'ar': 'ar-SA',
    'hi': 'hi-IN',
}

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
                  user_input TEXT, ai_response TEXT, language TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ========== LANGUAGE DETECTION ==========
def detect_language_text(text):
    """Detect language from text"""
    try:
        if not text or len(text.strip()) < 3:
            return 'en'  # Default to English for short text
        
        detected = detect(text)
        
        # Map to our supported language codes
        if detected in LANGUAGE_MAPPING:
            return LANGUAGE_MAPPING[detected]
        elif detected.split('-')[0] in LANGUAGE_MAPPING:
            return LANGUAGE_MAPPING[detected.split('-')[0]]
        
        return 'en'  # Default fallback
    except Exception as e:
        print(f"⚠️ Language detection error: {e}")
        return 'en'

# ========== AUDIO PREPROCESSING ==========
def preprocess_audio(audio_bytes, target_format='wav'):
    """Preprocess audio data to improve recognition rate"""
    try:
        print(f"🎵 Preprocessing audio: {len(audio_bytes)} bytes")
        
        # Method 1: Try using pydub conversion
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            # Load and convert with pydub
            audio = AudioSegment.from_file(tmp_path, format="webm")
            
            # Convert to mono
            audio = audio.set_channels(1)
            
            # Normalize volume
            audio = audio.normalize()
            
            # Convert to 16kHz sample rate (Speech Recognition recommendation)
            audio = audio.set_frame_rate(16000)
            
            # Export as wav
            output = io.BytesIO()
            audio.export(output, format="wav")
            processed_bytes = output.getvalue()
            
            print(f"✅ Audio preprocessed: {len(processed_bytes)} bytes")
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return processed_bytes
            
        except Exception as e:
            print(f"⚠️ Pydub preprocessing failed: {e}")
            # If pydub fails, return original audio
            return audio_bytes
            
    except Exception as e:
        print(f"❌ Audio preprocessing error: {e}")
        return audio_bytes

# ========== CONVERSATION MEMORY ==========
def save_conversation(session_id, user_input, ai_response, language='en'):
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO conversations VALUES (?, ?, ?, ?, ?)",
                  (session_id, datetime.now(), user_input, ai_response, language))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")

def get_conversation_history(session_id, limit=5):
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("SELECT user_input, ai_response, language FROM conversations WHERE session_id=? ORDER BY timestamp DESC LIMIT ?", 
                  (session_id, limit))
        history = c.fetchall()
        conn.close()
        return history
    except Exception as e:
        print(f"Database error: {e}")
        return []

# ========== SPEECH-TO-TEXT ==========
def transcribe_audio(audio_bytes, language_hint='en-US'):
    """Speech-to-text using Google Speech Recognition"""
    if not SR_AVAILABLE:
        return "Speech recognition is temporarily unavailable. Please use text mode."
    
    try:
        r = sr.Recognizer()
        
        print(f"🔊 Audio bytes length: {len(audio_bytes)}")
        
        # Try different sample rates
        sample_rates = [16000, 44100, 48000]
        
        for sample_rate in sample_rates:
            try:
                print(f"🔄 Trying sample rate: {sample_rate} Hz")
                
                # Try different byte widths
                for sample_width in [2, 4]:  # 16-bit or 32-bit
                    try:
                        audio_data = sr.AudioData(audio_bytes, 
                                                 sample_rate=sample_rate, 
                                                 sample_width=sample_width)
                        
                        # Adjust recognizer settings
                        r.energy_threshold = 300  # Lower energy threshold
                        r.dynamic_energy_threshold = True
                        r.pause_threshold = 0.8   # Reduce pause threshold
                        
                        # Try recognition
                        text = r.recognize_google(audio_data, 
                                                 language=language_hint,
                                                 show_all=False)
                        
                        if text and len(text.strip()) > 0:
                            print(f"✅ Transcribed ({sample_rate}Hz, {sample_width} bytes): {text}")
                            return text
                            
                    except Exception as e:
                        print(f"⚠️ Try {sample_rate}Hz/{sample_width} failed: {e}")
                        continue
                        
            except Exception as e:
                print(f"⚠️ Sample rate {sample_rate} failed: {e}")
                continue
        
        # If all attempts fail with specific language, try auto-detect
        if language_hint != 'en-US':
            print("🔄 Trying auto-detect as fallback...")
            try:
                audio_data = sr.AudioData(audio_bytes, sample_rate=16000, sample_width=2)
                text = r.recognize_google(audio_data, language=None)
                if text and len(text.strip()) > 0:
                    print(f"✅ Transcribed (auto-detect): {text}")
                    return text
            except:
                pass
        
        print("❌ All transcription attempts failed")
        return "I couldn't understand the audio. Please try speaking more clearly."
        
    except sr.UnknownValueError:
        print("⚠️ Speech recognition could not understand audio")
        return "I couldn't understand the audio. Please try speaking more clearly."
    except sr.RequestError as e:
        print(f"❌ Speech recognition service error: {e}")
        return "Speech service is temporarily unavailable. Please try again."
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        import traceback
        traceback.print_exc()
        return "Error processing audio. Please try again."

# ========== AI RESPONSE ==========
def get_ai_response(text, session_id="default", language='en'):
    """Get response from Gemini 2.5 Flash with conversation memory"""
    
    if not GEMINI_API_KEY:
        return "AI service is not configured. Please set GEMINI_API_KEY environment variable.", language
    
    # Get conversation history
    history = get_conversation_history(session_id)
    
    # Build context from history
    context = ""
    if history:
        for user_msg, ai_msg, hist_lang in reversed(history):
            context += f"User: {user_msg}\nAssistant: {ai_msg}\n"
        context += "\n"
    
    # Add language instruction if not English
    language_instruction = ""
    if language != 'en':
        language_instruction = f"\nPlease respond in {language} language."
    
    # Combine with current input
    full_prompt = f"""{context}User: {text}

Respond conversationally in 1-3 sentences as a helpful voice assistant.{language_instruction}"""
    
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
            print(f"✅ Gemini 2.5 response ({language}): {ai_text[:50]}...")
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
    save_conversation(session_id, text, ai_text, language)
    
    return ai_text, language

# ========== TEXT-TO-SPEECH ==========
def text_to_speech(text, language='en'):
    """Convert text to speech using gTTS"""
    try:
        # Map language code
        tts_lang = LANGUAGE_MAPPING.get(language, 'en')
        if language.startswith('zh'):
            tts_lang = language  # Keep zh-cn or zh-tw
        
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        audio_bytes = audio_buffer.read()
        print(f"✅ Generated TTS audio ({tts_lang}): {len(audio_bytes)} bytes")
        return audio_bytes
        
    except Exception as e:
        print(f"❌ TTS error: {e}")
        # Fallback to English
        if language != 'en':
            try:
                tts = gTTS(text=text, lang='en', slow=False)
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                return audio_buffer.read()
            except:
                return b''
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
        "multilingual": True,
        "supported_languages": list(LANGUAGE_MAPPING.keys()),
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
        language_hint = data.get('language_hint', 'en')  # Default to English
        
        if not audio_base64:
            return jsonify({"success": False, "error": "No audio data provided"}), 400
        
        print(f"📥 Received audio: {len(audio_base64)} chars base64")
        
        # Convert base64 to bytes
        try:
            audio_bytes = base64.b64decode(audio_base64)
            print(f"✅ Decoded audio: {len(audio_bytes)} bytes")
        except Exception as e:
            return jsonify({"success": False, "error": f"Invalid base64 audio: {str(e)}"}), 400
        
        # Preprocess audio to improve recognition
        print("🎵 Preprocessing audio...")
        audio_bytes = preprocess_audio(audio_bytes)
        
        # Get speech recognition language code
        speech_lang = SPEECH_LANGUAGES.get(language_hint, 'en-US')
        
        # Step 1: Speech to Text
        print(f"🎤 Step 1: Transcribing audio (language hint: {language_hint})...")
        user_text = transcribe_audio(audio_bytes, speech_lang)
        
        # Check transcription result
        error_phrases = ["Speech recognition", "Error processing", "I couldn't understand", "service is temporarily"]
        if any(phrase in user_text for phrase in error_phrases):
            return jsonify({
                "success": False, 
                "error": user_text, 
                "user_text": user_text,
                "debug": "Transcription failed"
            }), 400
        
        # Step 2: Detect language from text
        print("🌐 Step 2: Detecting language...")
        detected_language = detect_language_text(user_text)
        
        # Use hint if provided, otherwise use detected
        response_language = language_hint if language_hint != 'auto' else detected_language
        print(f"🌐 Language: detected={detected_language}, using={response_language}")
        
        # Step 3: Get AI Response
        print(f"🤖 Step 3: Getting AI response in {response_language}...")
        ai_response, final_language = get_ai_response(user_text, session_id, response_language)
        
        # Step 4: Text to Speech
        print(f"🔊 Step 4: Generating speech in {final_language}...")
        audio_response = text_to_speech(ai_response, final_language)
        
        if not audio_response:
            return jsonify({
                "success": False,
                "error": "Failed to generate audio response",
                "user_text": user_text,
                "ai_response": ai_response,
                "detected_language": detected_language
            }), 500
        
        audio_base64_response = base64.b64encode(audio_response).decode('utf-8')
        print(f"✅ Response audio: {len(audio_base64_response)} chars base64")
        
        return jsonify({
            "success": True,
            "user_text": user_text,
            "ai_response": ai_response,
            "audio": audio_base64_response,
            "detected_language": detected_language,
            "response_language": final_language,
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"❌ Voice chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e), "message": "Internal server error"}), 500

@app.route('/api/text-chat', methods=['POST'])
def text_chat():
    """Text-only endpoint with language support"""
    try:
        data = request.json
        
        if not data:
            return jsonify({"success": False, "error": "No JSON data received"}), 400
        
        text = data.get('text')
        session_id = data.get('session_id', 'default_session')
        language_hint = data.get('language_hint', 'auto')
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        print(f"💬 Text chat: {text}")
        
        # Detect language if auto
        if language_hint == 'auto':
            detected_language = detect_language_text(text)
            print(f"🌐 Auto-detected language: {detected_language}")
        else:
            detected_language = language_hint
        
        # Get AI response
        ai_response, response_language = get_ai_response(text, session_id, detected_language)
        print(f"✅ AI response ({response_language}): {ai_response}")
        
        return jsonify({
            "success": True,
            "ai_response": ai_response,
            "detected_language": detected_language,
            "response_language": response_language,
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"❌ Text chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/detect-language', methods=['POST'])
def detect_language_endpoint():
    """Detect language from text"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        language = detect_language_text(text)
        probabilities = []
        
        try:
            lang_probs = detect_langs(text)
            probabilities = [{"lang": str(lp.lang), "prob": lp.prob} for lp in lang_probs]
        except:
            pass
        
        return jsonify({
            "success": True,
            "detected_language": language,
            "probabilities": probabilities,
            "text_sample": text[:100] + ("..." if len(text) > 100 else "")
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/supported-languages', methods=['GET'])
def supported_languages():
    """Get list of supported languages"""
    return jsonify({
        "success": True,
        "languages": list(LANGUAGE_MAPPING.keys()),
        "speech_languages": list(SPEECH_LANGUAGES.keys()),
        "total": len(LANGUAGE_MAPPING)
    })

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
        <title>🌍 Multilingual Voice AI Chatbot</title>
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
            .languages-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
                gap: 10px;
                margin: 20px 0;
            }}
            .language-tag {{
                background: #e3f2fd;
                padding: 8px 12px;
                border-radius: 20px;
                text-align: center;
                font-size: 14px;
                border: 1px solid #bbdefb;
            }}
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
            <h1>🌍 Multilingual Voice AI Chatbot</h1>
            <p style="color: #666;">Real-time Speech-to-Speech in Multiple Languages</p>
            
            <div class="status">
                <p><strong>Status:</strong> ✅ Active</p>
                <p><strong>Cost:</strong> $0/month</p>
                <p><strong>Multilingual:</strong> ✅ {len(LANGUAGE_MAPPING)} Languages Supported</p>
            </div>
            
            <div class="status {'warning' if not GEMINI_API_KEY else ''}">
                <p><strong>Gemini 2.5 Flash:</strong> {gemini_status}</p>
            </div>
            
            <div class="status {'' if SR_AVAILABLE else 'warning'}">
                <p><strong>Speech Recognition:</strong> {sr_status}</p>
            </div>
            
            <h3>🌐 Supported Languages</h3>
            <div class="languages-grid">
                <div class="language-tag">English (en)</div>
                <div class="language-tag">Spanish (es)</div>
                <div class="language-tag">French (fr)</div>
                <div class="language-tag">German (de)</div>
                <div class="language-tag">Italian (it)</div>
                <div class="language-tag">Portuguese (pt)</div>
                <div class="language-tag">Japanese (ja)</div>
                <div class="language-tag">Korean (ko)</div>
                <div class="language-tag">Chinese (zh)</div>
                <div class="language-tag">Arabic (ar)</div>
                <div class="language-tag">Hindi (hi)</div>
                <div class="language-tag">Russian (ru)</div>
            </div>
            
            <h2>API Endpoints</h2>
            <div class="endpoint"><strong>POST</strong> /api/voice-chat - Voice chat (add "language_hint": "es" for Spanish, etc.)</div>
            <div class="endpoint"><strong>POST</strong> /api/text-chat - Text chat with language detection</div>
            <div class="endpoint"><strong>POST</strong> /api/detect-language - Detect language from text</div>
            <div class="endpoint"><strong>GET</strong> /api/supported-languages - List supported languages</div>
            <div class="endpoint"><strong>POST</strong> /api/clear-history - Clear conversation history</div>
            <div class="endpoint"><strong>GET</strong> /health - Health check</div>
            
            <h3>Test API</h3>
            <button onclick="testTextChat('en')">Test English</button>
            <button onclick="testTextChat('es')">Test Spanish</button>
            <button onclick="testTextChat('fr')">Test French</button>
            <button onclick="testHealth()">Test Health</button>
            
            <div id="testResult"></div>
            
            <script>
                async function testTextChat(lang) {{
                    const testTexts = {{
                        'en': 'Hello! How are you today?',
                        'es': '¡Hola! ¿Cómo estás hoy?',
                        'fr': 'Bonjour! Comment allez-vous aujourd\\\\'hui?'
                    }};
                    
                    const text = testTexts[lang] || testTexts['en'];
                    const result = document.getElementById('testResult');
                    result.innerHTML = '<div style="color: orange;">Testing ' + lang + '...</div>';
                    
                    try {{
                        const response = await fetch('/api/text-chat', {{
                            method: 'POST',
                            headers: {{'Content-Type': 'application/json'}},
                            body: JSON.stringify({{
                                text: text, 
                                session_id: 'test_' + Date.now(),
                                language_hint: lang
                            }})
                        }});
                        
                        const data = await response.json();
                        
                        if (data.success) {{
                            result.innerHTML = '<div class="success">' +
                                '<strong>✅ ' + lang.toUpperCase() + ' SUCCESS!</strong><br>' +
                                'Language: ' + data.response_language + '<br>' +
                                'Response: "' + data.ai_response + '"' +
                            '</div>';
                        }} else {{
                            result.innerHTML = '<div class="error"><strong>❌ Error:</strong> ' + data.error + '</div>';
                        }}
                    }} catch (error) {{
                        result.innerHTML = '<div class="error"><strong>❌ Network Error:</strong> ' + error.message + '</div>';
                    }}
                }}
                
                async function testHealth() {{
                    const result = document.getElementById('testResult');
                    result.innerHTML = '<div style="color: orange;">Checking health...</div>';
                    
                    try {{
                        const response = await fetch('/health');
                        const data = await response.json();
                        
                        result.innerHTML = 
                            '<div class="success">' +
                                '<strong>✅ Healthy</strong><br>' +
                                'Speech Recognition: ' + (data.speech_recognition ? '✅ v' + data.speech_recognition_version : '❌') + '<br>' +
                                'Gemini: ' + (data.gemini_configured ? '✅' : '❌') + '<br>' +
                                'Multilingual: ' + (data.multilingual ? '✅ ' + data.supported_languages.length + ' languages' : '❌') +
                            '</div>';
                    }} catch (error) {{
                        result.innerHTML = '<div class="error"><strong>❌ Failed:</strong> ' + error.message + '</div>';
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
    print(f"🚀 Starting Multilingual Voice Chatbot Backend on port {port}")
    print(f"🌐 Supports {len(LANGUAGE_MAPPING)} languages")
    print(f"🐍 Python: {os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}")
    print(f"📊 Speech Recognition: {SR_AVAILABLE}")
    print(f"🤖 Gemini: {'Configured (2.5 Flash)' if GEMINI_API_KEY else 'NOT CONFIGURED'}")
    app.run(host='0.0.0.0', port=port, debug=False)