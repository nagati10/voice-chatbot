from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from gtts import gTTS
import io
import os
from datetime import datetime
import sqlite3
import base64
from pydub import AudioSegment
import tempfile
import json

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
    'zh': 'zh',      # Chinese
    'zh-cn': 'zh',   # Chinese Simplified
    'zh-tw': 'zh-tw', # Chinese Traditional
    'ar': 'ar',      # Arabic
    'hi': 'hi',      # Hindi
}

if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not set!")
else:
    print("✅ GEMINI_API_KEY is configured")
    # Initialize Gemini client
    client = genai.Client(api_key=GEMINI_API_KEY)

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

# ========== AUDIO PREPROCESSING ==========
def preprocess_audio(audio_bytes):
    """Convert WebM to MP3 for Gemini"""
    try:
        print(f"🎵 Preprocessing audio: {len(audio_bytes)} bytes")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            # Load WebM with pydub
            audio = AudioSegment.from_file(tmp_path, format="webm")
            
            # Convert to mono
            audio = audio.set_channels(1)
            
            # Normalize volume
            audio = audio.normalize()
            
            # Convert to MP3 (Gemini supports MP3)
            output = io.BytesIO()
            audio.export(output, format="mp3", parameters=["-ar", "16000"])
            processed_bytes = output.getvalue()
            
            print(f"✅ Converted to MP3: {len(processed_bytes)} bytes")
            
            # Clean up
            os.unlink(tmp_path)
            
            return processed_bytes, "audio/mp3"
            
        except Exception as e:
            print(f"⚠️ Pydub conversion failed: {e}")
            # Try to use as-is
            os.unlink(tmp_path)
            return audio_bytes, "audio/webm"
            
    except Exception as e:
        print(f"❌ Audio preprocessing error: {e}")
        return audio_bytes, "audio/webm"


# In your Flask backend, update the transcribe_with_gemini function:

def transcribe_with_gemini(audio_bytes, mime_type="audio/mp3"):
    """Transcribe speech to text using Gemini 2.5 Flash"""
    if not GEMINI_API_KEY:
        return "AI service is not configured.", "en"
    
    try:
        print(f"🎤 Transcribing with Gemini ({len(audio_bytes)} bytes, {mime_type})...")
        
        # Create prompt for transcription with language detection
        # UPDATED: More specific prompt for better Arabic support
        prompt = """Please transcribe this speech to text. 
        
        IMPORTANT: 
        1. Detect the language of the speech (especially if it's Arabic, Chinese, etc.)
        2. If you hear Arabic speech, transcribe it carefully with Arabic script
        3. Provide ONLY a JSON response with this exact format:
        {
            "text": "the transcribed text here",
            "language": "language code (e.g., en, es, fr, ar)",
            "confidence": 0.95
        }
        
        If you cannot understand the audio, return:
        {
            "text": "",
            "language": "unknown",
            "confidence": 0.0
        }"""
        
        # Send to Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mime_type,
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,  # Low temperature for accurate transcription
                max_output_tokens=100
            )
        )
        
        # Parse the response
        result_text = response.text.strip()
        print(f"📝 Gemini raw response: {result_text}")
        
        # Try to parse as JSON
        try:
            # Extract JSON from response (Gemini might add markdown)
            if result_text.startswith("```json"):
                result_text = result_text[7:-3]  # Remove ```json and ```
            elif result_text.startswith("```"):
                result_text = result_text[3:-3]  # Remove ``` and ```
            
            result = json.loads(result_text)
            
            text = result.get("text", "")
            language = result.get("language", "en")
            confidence = result.get("confidence", 0)
            
            # LOWER CONFIDENCE THRESHOLD for Arabic/other languages
            if confidence > 0.1 or language == 'ar':  # Lowered from 0.3 to 0.1
                print(f"✅ Transcribed ({language}, confidence: {confidence}): {text}")
                return text, language
            else:
                print(f"⚠️ Low confidence transcription: {text}")
                # Try to manually detect language from text
                if text:
                    detected_lang = detect_language_from_text(text)
                    return text, detected_lang
                return "I couldn't understand clearly. Please try again.", "en"
                
        except json.JSONDecodeError:
            # If not JSON, use raw text
            print(f"⚠️ Could not parse JSON, using raw text: {result_text}")
            # Try to detect language from text
            language = detect_language_from_text(result_text)
            return result_text, language
            
    except Exception as e:
        print(f"❌ Gemini transcription error: {e}")
        import traceback
        traceback.print_exc()
        return "Error processing audio with AI. Please try again.", "en"


# ========== SIMPLE LANGUAGE DETECTION ==========
def detect_language_from_text(text):
    """Simple language detection from text patterns"""
    if not text:
        return 'en'
    
    text_lower = text.lower()
    
    # Check for Arabic characters (more comprehensive range)
    arabic_range = '\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF'
    if any(c for c in text if c in arabic_range):
        return 'ar'
    
    # Check for Chinese characters
    if any('\u4e00' <= c <= '\u9fff' for c in text):
        return 'zh'
    
    # Check for Japanese characters
    japanese_ranges = [
        '\u3040-\u309F',  # Hiragana
        '\u30A0-\u30FF',  # Katakana
        '\u4E00-\u9FAF',  # Kanji
    ]
    for jp_range in japanese_ranges:
        if any(c for c in text if c in jp_range):
            return 'ja'
    
    # Check for Korean characters
    if any('\uAC00' <= c <= '\uD7A3' for c in text):
        return 'ko'
    
    # Check for common phrases in various languages
    language_patterns = {
        'es': ['hola', 'gracias', 'sí', 'no', 'por favor', 'cómo', 'qué'],
        'fr': ['bonjour', 'merci', 'oui', 'non', 's\'il vous plaît', 'comment'],
        'de': ['hallo', 'danke', 'ja', 'nein', 'bitte', 'wie'],
        'it': ['ciao', 'grazie', 'sì', 'no', 'per favore', 'come'],
        'ru': ['привет', 'спасибо', 'да', 'нет', 'пожалуйста', 'как'],
        'ar': ['مرحبا', 'شكرا', 'نعم', 'لا', 'من فضلك', 'كيف'],
        'hi': ['नमस्ते', 'धन्यवाद', 'हाँ', 'नहीं', 'कृपया', 'कैसे'],
    }
    
    for lang_code, patterns in language_patterns.items():
        for pattern in patterns:
            if pattern in text_lower:
                return lang_code
    
    # Default to English
    return 'en'

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

# ========== AI RESPONSE ==========
def get_ai_response(text, session_id="default", language='en'):
    """Get response from Gemini 2.5 Flash"""
    
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
    
    # Language-specific instructions
    language_prompts = {
        'en': "Respond conversationally in 1-3 sentences as a helpful voice assistant.",
        'es': "Responde de manera conversacional en 1-3 frases como un asistente de voz útil.",
        'fr': "Répondez de manière conversationnelle en 1-3 phrases comme un assistant vocal utile.",
        'ar': "رد بطريقة محادثة في 1-3 جملة كمساعد صوتي مفيد.",
        'de': "Antworten Sie konversationell in 1-3 Sätzen als hilfreicher Sprachassistent.",
        'it': "Rispondi conversazionalmente in 1-3 frasi come un assistente vocale utile.",
        'ja': "役立つ音声アシスタントとして、1〜3文で会話形式で答えてください。",
        'ko': "유용한 음성 비서로서 1~3문장으로 대화식으로 응답하세요.",
        'zh': "以有用的语音助手身份，用1-3句话进行对话式回答。",
        'hi': "एक मददगार आवाज सहायक के रूप में 1-3 वाक्यों में बातचीत के तरीके से जवाब दें।",
        'ru': "Отвечайте разговорным тоном в 1-3 предложениях в качестве полезного голосового помощника.",
        'pt': "Responda conversacionalmente em 1-3 frases como um assistente de voz útil.",
    }
    
    instruction = language_prompts.get(language, language_prompts['en'])
    
    # Combine with current input
    full_prompt = f"""{context}User: {text}

{instruction}

IMPORTANT: Respond ONLY in {language} language."""
    
    try:
        # Generate content with Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=150
            )
        )
        
        # Extract text from response
        if response and response.text:
            ai_text = response.text.strip()
            print(f"✅ Gemini response ({language}): {ai_text[:100]}...")
        else:
            print("⚠️ Empty response from Gemini")
            ai_text = "I heard you, but I'm having trouble responding right now. Could you rephrase that?"
        
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        
        # Fallback responses
        fallback_responses = {
            'en': "I'm having trouble connecting to my AI service right now. Please try again in a moment.",
            'es': "Estoy teniendo problemas para conectarme a mi servicio de IA en este momento. Por favor, inténtalo de nuevo en un momento.",
            'fr': "J'ai du mal à me connecter à mon service d'IA pour le moment. Veuillez réessayer dans un instant.",
            'ar': "أواجه صعوبة في الاتصال بخدمة الذكاء الاصطناعي الخاصة بي في الوقت الحالي. يرجى المحاولة مرة أخرى في لحظة.",
            'de': "Ich habe gerade Probleme, mich mit meinem KI-Dienst zu verbinden. Bitte versuchen Sie es in einem Moment erneut.",
            'ja': "現在、AIサービスに接続するのに問題が発生しています。しばらくしてからもう一度お試しください。",
            'ko': "현재 AI 서비스에 연결하는 데 문제가 있습니다. 잠시 후 다시 시도해 주세요.",
            'zh': "我目前连接AI服务时遇到问题。请稍后再试。",
            'hi': "मुझे अभी अपनी AI सेवा से कनेक्ट होने में परेशानी हो रही है। कृपया एक पल में फिर से प्रयास करें।",
            'ru': "У меня возникли проблемы с подключением к моему сервису ИИ. Пожалуйста, попробуйте еще раз через мгновение.",
            'pt': "Estou tendo problemas para me conectar ao meu serviço de IA no momento. Por favor, tente novamente em um momento.",
            'it': "Sto avendo problemi a connettermi al mio servizio di IA in questo momento. Per favore, riprova tra un momento.",
        }
        
        ai_text = fallback_responses.get(language, fallback_responses['en'])
    
    # Save to memory
    save_conversation(session_id, text, ai_text, language)
    
    return ai_text, language

# ========== TEXT-TO-SPEECH ==========
def text_to_speech(text, language='en'):
    """Convert text to speech using gTTS"""
    try:
        # Map language code
        tts_lang = LANGUAGE_MAPPING.get(language, 'en')
        
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
        "gemini_api": "gemini-2.5-flash",
        "gemini_configured": bool(GEMINI_API_KEY),
        "multilingual": True,
        "supported_languages": list(LANGUAGE_MAPPING.keys()),
        "audio_support": True,
        "speech_recognition": "Gemini-native",
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/voice-chat', methods=['POST'])
def voice_chat():
    """Main endpoint - process voice to voice using Gemini"""
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
        
        # Preprocess audio (convert WebM to MP3)
        print("🎵 Preprocessing audio...")
        processed_audio, mime_type = preprocess_audio(audio_bytes)
        
        # Step 1: Transcribe with Gemini (includes language detection)
        print("🎤 Step 1: Transcribing with Gemini...")
        user_text, detected_language = transcribe_with_gemini(processed_audio, mime_type)
        
        # Check transcription result
        error_phrases = ["I couldn't understand", "Error processing", "Please try speaking"]
        if any(phrase in user_text for phrase in error_phrases):
            return jsonify({
                "success": False, 
                "error": user_text, 
                "user_text": user_text,
                "detected_language": detected_language
            }), 400
        
        # Step 2: Get AI Response
        print(f"🤖 Step 2: Getting AI response in {detected_language}...")
        ai_response, response_language = get_ai_response(user_text, session_id, detected_language)
        
        # Step 3: Text to Speech
        print(f"🔊 Step 3: Generating speech in {response_language}...")
        audio_response = text_to_speech(ai_response, response_language)
        
        if not audio_response or len(audio_response) < 100:
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
            "response_language": response_language,
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
        language_hint = data.get('language_hint', 'auto')
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        print(f"💬 Text chat: {text}")
        
        # Detect language
        detected_language = detect_language_from_text(text)
        
        # Use hint if provided
        if language_hint != 'auto' and language_hint in LANGUAGE_MAPPING:
            detected_language = language_hint
        
        print(f"🌐 Language: {detected_language}")
        
        # Get AI response
        ai_response, response_language = get_ai_response(text, session_id, detected_language)
        print(f"✅ AI response ({response_language}): {ai_response[:100]}...")
        
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

@app.route('/api/test-audio', methods=['POST'])
def test_audio():
    """Test audio processing with Gemini"""
    try:
        data = request.json
        audio_base64 = data.get('audio')
        
        if not audio_base64:
            return jsonify({"success": False, "error": "No audio provided"}), 400
        
        audio_bytes = base64.b64decode(audio_base64)
        
        # Preprocess
        processed_audio, mime_type = preprocess_audio(audio_bytes)
        
        # Test transcription
        text, language = transcribe_with_gemini(processed_audio, mime_type)
        
        return jsonify({
            "success": True,
            "transcribed_text": text,
            "detected_language": language,
            "audio_size": len(audio_bytes),
            "processed_size": len(processed_audio),
            "mime_type": mime_type
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default_session')
        
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM conversations WHERE session_id=?", (session_id,))
        deleted = c.rowcount
        conn.commit()
        conn.close()
        
        print(f"🗑️ Cleared {deleted} messages")
        
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
    """Homepage"""
    gemini_status = "✅ Configured" if GEMINI_API_KEY else "❌ Not Configured"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🎤 Gemini Voice AI Chatbot</title>
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
            <h1>🎤 Gemini Voice AI Chatbot</h1>
            <p style="color: #666;">Real-time Speech-to-Speech using Gemini 2.5 Flash</p>
            
            <div class="status">
                <p><strong>Status:</strong> ✅ Active</p>
                <p><strong>Speech Recognition:</strong> ✅ Gemini-native</p>
                <p><strong>Multilingual:</strong> ✅ Auto-detection</p>
            </div>
            
            <div class="status {'warning' if not GEMINI_API_KEY else ''}">
                <p><strong>Gemini 2.5 Flash:</strong> {gemini_status}</p>
            </div>
            
            <h3>🌐 Supported Languages (Auto-detected)</h3>
            <div class="languages-grid">
                <div class="language-tag">English</div>
                <div class="language-tag">Spanish</div>
                <div class="language-tag">French</div>
                <div class="language-tag">Arabic</div>
                <div class="language-tag">German</div>
                <div class="language-tag">Italian</div>
                <div class="language-tag">Japanese</div>
                <div class="language-tag">Korean</div>
                <div class="language-tag">Chinese</div>
                <div class="language-tag">Hindi</div>
                <div class="language-tag">Russian</div>
                <div class="language-tag">Portuguese</div>
            </div>
            
            <h2>API Endpoints</h2>
            <div class="endpoint"><strong>POST</strong> /api/voice-chat - Voice chat with auto language detection</div>
            <div class="endpoint"><strong>POST</strong> /api/text-chat - Text chat</div>
            <div class="endpoint"><strong>POST</strong> /api/test-audio - Test audio processing</div>
            <div class="endpoint"><strong>POST</strong> /api/clear-history - Clear history</div>
            <div class="endpoint"><strong>GET</strong> /health - Health check</div>
            
            <h3>Features</h3>
            <ul>
                <li>✅ Speech-to-text using Gemini 2.5 Flash</li>
                <li>✅ Automatic language detection from audio</li>
                <li>✅ Multilingual responses in same language</li>
                <li>✅ Text-to-speech with gTTS</li>
                <li>✅ Conversation memory</li>
                <li>✅ Free to use (Render + Gemini free tier)</li>
            </ul>
            
            <button onclick="testHealth()">Test Health</button>
            
            <div id="testResult"></div>
            
            <script>
                async function testHealth() {{
                    const result = document.getElementById('testResult');
                    result.innerHTML = '<div style="color: orange;">Checking...</div>';
                    
                    try {{
                        const response = await fetch('/health');
                        const data = await response.json();
                        
                        result.innerHTML = 
                            '<div class="success">' +
                                '<strong>✅ Healthy</strong><br>' +
                                'Gemini: ' + (data.gemini_configured ? '✅' : '❌') + '<br>' +
                                'Audio Support: ' + (data.audio_support ? '✅' : '❌') + '<br>' +
                                'Multilingual: ' + (data.multilingual ? '✅' : '❌') + '<br>' +
                                'Languages: ' + data.supported_languages.length +
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
    print(f"🚀 Starting Gemini Voice Chatbot on port {port}")
    print(f"🤖 Using Gemini 2.5 Flash for speech recognition")
    print(f"🌐 Supports {len(LANGUAGE_MAPPING)} languages with auto-detection")
    print(f"🔊 Speech-to-text: Gemini-native")
    app.run(host='0.0.0.0', port=port, debug=False)