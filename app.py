# Remove pydub import and replace with alternative approach
import io
import os
import wave
import base64
from datetime import datetime
import sqlite3
import tempfile
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from gtts import gTTS

app = Flask(__name__)
CORS(app)

# ========== CONFIG - DUAL API KEYS ==========
GEMINI_STT_KEY = os.getenv("GEMINI_STT")  # For Speech-to-Text (transcription)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # For conversation/AI responses
DATABASE_FILE = "conversations.db"

# Language code mapping for gTTS
LANGUAGE_MAPPING = {
    'en': 'en',    # English
    'es': 'es',    # Spanish
    'fr': 'fr',    # French
    'de': 'de',    # German
    'it': 'it',    # Italian
    'pt': 'pt',    # Portuguese
    'ru': 'ru',    # Russian
    'ja': 'ja',    # Japanese
    'ko': 'ko',    # Korean
    'zh': 'zh',    # Chinese
    'zh-cn': 'zh', # Chinese Simplified
    'zh-tw': 'zh-tw', # Chinese Traditional
    'ar': 'ar',    # Arabic
    'hi': 'hi',    # Hindi
}

if not GEMINI_STT_KEY:
    print("⚠️ WARNING: GEMINI_STT not set!")
else:
    print("✅ GEMINI_STT is configured")

if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not set!")
else:
    print("✅ GEMINI_API_KEY is configured")

# Initialize Gemini clients (one for each API key)
client_stt = genai.Client(api_key=GEMINI_STT_KEY) if GEMINI_STT_KEY else None
client_conversation = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Initialize database
def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (session_id TEXT, timestamp DATETIME, user_input TEXT, ai_response TEXT, language TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ========== SIMPLE AUDIO PREPROCESSING ==========
def preprocess_audio(audio_bytes):
    """Simple audio preprocessing without pydub"""
    try:
        print(f"🎵 Processing audio: {len(audio_bytes)} bytes")
        return audio_bytes, "audio/webm"
    except Exception as e:
        print(f"❌ Audio processing error: {e}")
        return audio_bytes, "audio/webm"

# ========== GEMINI SPEECH-TO-TEXT (Uses GEMINI_STT) ==========
def transcribe_with_gemini(audio_bytes, mime_type="audio/webm"):
    """Transcribe speech to text using Gemini 2.5 Flash (STT key)"""
    if not GEMINI_STT_KEY or not client_stt:
        return "Speech-to-text service is not configured.", "en"
    
    try:
        print(f"🎤 Transcribing with Gemini STT ({len(audio_bytes)} bytes, {mime_type})...")
        
        prompt = """Listen to this audio and transcribe the speech to text.
Detect the language and return ONLY a valid JSON response:
{"text": "transcribed text", "language": "en", "confidence": 0.95}

Language codes: en, es, fr, ar, zh, ja, ko, ru, de, it, pt, hi
If you cannot understand, return: {"text": "", "language": "unknown", "confidence": 0.0}"""
        
        response = client_stt.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mime_type,
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE"
                    ),
                ]
            )
        )
        
        # Check if response is empty
        if not response or not hasattr(response, 'candidates') or not response.candidates:
            print(f"⚠️ Empty response from Gemini STT")
            return "Could not transcribe audio. Please try again.", "en"
        
        # Try to get text from response
        result_text = None
        if hasattr(response, 'text') and response.text:
            result_text = response.text.strip()
        elif response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    result_text = candidate.content.parts[0].text.strip()
        
        if not result_text:
            print(f"⚠️ No text in transcription response")
            return "Could not transcribe audio. Please try again.", "en"
        
        print(f"📝 Gemini STT response: {result_text}")
        
        # Clean markdown code blocks
        if "json" in result_text.lower() and '`' in result_text:
            result_text = result_text.replace('```json', '').replace('```', '').strip()
        
        # Parse JSON
        try:
            result = json.loads(result_text)
            text = result.get("text", "").strip()
            language = result.get("language", "unknown")
            confidence = result.get("confidence", 0)
            
            if confidence > 0.1 and text:
                print(f"✅ Transcribed ({language}, confidence: {confidence}): {text}")
                if language == "unknown" and text:
                    language = detect_language_from_text(text)
                return text, language
            else:
                return "Could you please repeat that more clearly?", "en"
                
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed: {e}")
            print(f"⚠️ Result text: {result_text}")
            
            # Try regex extraction
            import re
            text_match = re.search(r'"text"\s*:\s*"([^"]+)"', result_text)
            lang_match = re.search(r'"language"\s*:\s*"([^"]+)"', result_text)
            
            if text_match:
                text = text_match.group(1)
                language = lang_match.group(1) if lang_match else detect_language_from_text(text)
                print(f"✅ Extracted via regex: {text} ({language})")
                return text, language
            
            # Last resort
            return "Could not understand the audio. Please try again.", "en"
            
    except Exception as e:
        print(f"❌ Gemini STT error: {e}")
        import traceback
        traceback.print_exc()
        return "Error processing audio. Please try again.", "en"

# ========== LANGUAGE DETECTION ==========
def detect_language_from_text(text):
    """Improved language detection from text"""
    if not text or not text.strip():
        return 'en'
    
    text_lower = text.lower().strip()
    
    # Check for Arabic characters
    arabic_chars = set('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
    if any(char in arabic_chars for char in text):
        return 'ar'
    
    # Check for Chinese characters
    if any('\u4e00' <= c <= '\u9fff' for c in text):
        return 'zh'
    
    # Check for Japanese
    if any('\u3040' <= c <= '\u309f' for c in text):
        return 'ja'
    if any('\u30a0' <= c <= '\u30ff' for c in text):
        return 'ja'
    
    # Check for Korean
    if any('\uac00' <= c <= '\ud7a3' for c in text):
        return 'ko'
    
    language_keywords = {
        'en': ['hello', 'hi', 'thank you', 'please', 'how are', 'what is'],
        'es': ['hola', 'gracias', 'por favor', 'cómo', 'qué'],
        'fr': ['bonjour', 'merci', 's\'il vous plaît', 'comment', 'quoi'],
        'de': ['hallo', 'danke', 'bitte', 'wie', 'was'],
        'it': ['ciao', 'grazie', 'per favore', 'come', 'che'],
        'pt': ['olá', 'obrigado', 'por favor', 'como', 'o que'],
        'ru': ['привет', 'спасибо', 'пожалуйста', 'как', 'что'],
        'ar': ['مرحبا', 'شكرا', 'من فضلك', 'كيف', 'ما'],
        'hi': ['नमस्ते', 'धन्यवाद', 'कृपया', 'कैसे', 'क्या'],
        'ja': ['こんにちは', 'ありがとう', 'お願いします', 'どう', '何'],
        'ko': ['안녕하세요', '감사합니다', '부탁합니다', '어떻게', '무엇'],
        'zh': ['你好', '谢谢', '请', '怎么', '什么'],
    }
    
    for lang, keywords in language_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                return lang
    
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

# ========== AI RESPONSE (Uses GEMINI_API_KEY) ==========
def get_ai_response(text, session_id="default", language='en'):
    """Get response from Gemini 2.5 Flash (Conversation key)"""
    if not GEMINI_API_KEY or not client_conversation:
        return "AI service is not configured.", language
    
    history = get_conversation_history(session_id)
    
    context = ""
    if history:
        for user_msg, ai_msg, hist_lang in reversed(history):
            context += f"User: {user_msg}\nAssistant: {ai_msg}\n"
        context += "\n"
    
    language_prompts = {
        'en': "Respond in English in 1-3 sentences.",
        'es': "Responde en español en 1-3 frases.",
        'fr': "Répondez en français en 1-3 phrases.",
        'ar': "رد بالعربية في 1-3 جمل.",
        'de': "Antworten Sie auf Deutsch in 1-3 Sätzen.",
        'it': "Rispondi in italiano in 1-3 frasi.",
        'ja': "日本語で1〜3文で答えてください。",
        'ko': "한국어로 1~3문장으로 답하세요.",
        'zh': "用中文回答1-3句话。",
        'hi': "हिंदी में 1-3 वाक्यों में उत्तर दें।",
        'ru': "Ответьте на русском в 1-3 предложениях.",
        'pt': "Responda em português em 1-3 frases.",
    }
    
    instruction = language_prompts.get(language, language_prompts['en'])
    full_prompt = f"{context}User: {text}\n{instruction}"
    
    try:
        print(f"📤 Sending to Gemini API: {full_prompt[:150]}...")
        
        response = client_conversation.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=2048,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                ]
            )
        )
        
        # Extract text with multiple fallbacks
        ai_text = None
        
        if response and hasattr(response, 'text') and response.text:
            ai_text = response.text.strip()
        elif response and hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    if hasattr(candidate.content.parts[0], 'text'):
                        ai_text = candidate.content.parts[0].text.strip()
        
        if ai_text:
            print(f"✅ Gemini API response ({language}): {ai_text[:100]}...")
        else:
            print(f"⚠️ Empty response from Gemini API")
            print(f"Response: {response}")
            if hasattr(response, 'candidates'):
                print(f"Candidates: {response.candidates}")
            ai_text = "Sorry, I couldn't generate a response. Please try again."
            
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        import traceback
        traceback.print_exc()
        ai_text = "I'm having trouble connecting right now. Please try again."
    
    save_conversation(session_id, text, ai_text, language)
    return ai_text, language

# ========== TEXT-TO-SPEECH ==========
def text_to_speech(text, language='en'):
    """Convert text to speech using gTTS"""
    try:
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
        "stt_configured": bool(GEMINI_STT_KEY),
        "api_configured": bool(GEMINI_API_KEY),
        "gemini_api": "gemini-2.5-flash",
        "multilingual": True,
        "supported_languages": list(LANGUAGE_MAPPING.keys()),
        "audio_support": True,
        "speech_recognition": "Gemini-native (STT)",
        "conversation_engine": "Gemini-native (API)",
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
        language_hint = data.get('language_hint', 'auto')
        
        if not audio_base64:
            return jsonify({"success": False, "error": "No audio data provided"}), 400
        
        print(f"📥 Received audio: {len(audio_base64)} chars base64")
        
        try:
            audio_bytes = base64.b64decode(audio_base64)
            print(f"✅ Decoded audio: {len(audio_bytes)} bytes")
        except Exception as e:
            return jsonify({"success": False, "error": f"Invalid base64 audio: {str(e)}"}), 400
        
        print("🎵 Preprocessing audio...")
        processed_audio, mime_type = preprocess_audio(audio_bytes)
        
        print("🎤 Step 1: Transcribing with Gemini STT...")
        user_text, detected_language = transcribe_with_gemini(processed_audio, mime_type)
        
        if not user_text or len(user_text.strip()) < 2:
            return jsonify({
                "success": False,
                "error": "Could not understand speech. Please try speaking more clearly.",
                "user_text": user_text,
                "detected_language": detected_language
            }), 400
        
        if language_hint != 'auto' and language_hint in LANGUAGE_MAPPING:
            detected_language = language_hint
        
        print(f"🌐 Detected language: {detected_language}")
        
        print(f"🤖 Step 2: Getting AI response in {detected_language}...")
        ai_response, response_language = get_ai_response(user_text, session_id, detected_language)
        
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
        
        detected_language = detect_language_from_text(text)
        
        if language_hint != 'auto' and language_hint in LANGUAGE_MAPPING:
            detected_language = language_hint
        
        print(f"🌐 Language: {detected_language}")
        
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
    """Homepage with embedded UI"""
    stt_status = "✅ Configured" if GEMINI_STT_KEY else "❌ Not Configured"
    api_status = "✅ Configured" if GEMINI_API_KEY else "❌ Not Configured"
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎙️ Gemini Voice AI Chatbot</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin-bottom: 10px;
            color: #333;
        }}
        .header p {{
            color: #666;
            font-size: 16px;
        }}
        .status-box {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }}
        .status {{
            padding: 15px;
            border-radius: 10px;
            font-size: 14px;
            line-height: 1.6;
        }}
        .status.success {{
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
        }}
        .status.warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
        }}
        .main-content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin-bottom: 20px;
            color: #333;
            font-size: 20px;
        }}
        .input-group {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        textarea, input {{
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            resize: vertical;
        }}
        textarea:focus, input:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}
        .button-group {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        button {{
            flex: 1;
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            min-width: 140px;
        }}
        .btn-primary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
        .btn-primary:active {{
            transform: translateY(0);
        }}
        .btn-secondary {{
            background: #f0f0f0;
            color: #333;
        }}
        .btn-secondary:hover {{
            background: #e0e0e0;
        }}
        .btn-danger {{
            background: #ff6b6b;
            color: white;
        }}
        .btn-danger:hover {{
            background: #ff5252;
        }}
        .output {{
            margin-top: 15px;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
            min-height: 60px;
            border-left: 4px solid #667eea;
        }}
        .output.hidden {{
            display: none;
        }}
        .output-label {{
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
        }}
        .output-text {{
            color: #666;
            word-wrap: break-word;
        }}
        .loading {{
            color: #667eea;
            font-style: italic;
        }}
        .error {{
            color: #d32f2f;
        }}
        .success {{
            color: #388e3c;
        }}
        audio {{
            width: 100%;
            margin-top: 10px;
        }}
        #recordingStatus {{
            display: inline-block;
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-top: 10px;
        }}
        #recordingStatus.recording {{
            background: #ffebee;
            color: #d32f2f;
        }}
        #recordingStatus.ready {{
            background: #e8f5e9;
            color: #388e3c;
        }}
        @media (max-width: 768px) {{
            .main-content {{
                grid-template-columns: 1fr;
            }}
            .status-box {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎙️ Gemini Voice AI Chatbot</h1>
            <p>Real-time multilingual voice and text chat powered by Gemini 2.5 Flash</p>
            <div class="status-box">
                <div class="status success">
                    <strong>Speech-to-Text (GEMINI_STT):</strong> {stt_status}<br>
                    <strong>Conversation (GEMINI_API_KEY):</strong> {api_status}<br>
                    <strong>Languages:</strong> 14 supported
                </div>
                <div class="status" id="connectionStatus">
                    <strong>Testing connection...</strong>
                </div>
            </div>
        </div>

        <div class="main-content">
            <div class="card">
                <h2>🎤 Voice Chat</h2>
                <div class="input-group">
                    <label>Click to record (auto-detect language):</label>
                    <button class="btn-primary" id="recordBtn" onclick="startRecording()">🎤 Start Recording</button>
                    <div id="recordingStatus" class="hidden"></div>
                    <div id="transcriptOutput" class="output hidden">
                        <div class="output-label">Transcribed Text:</div>
                        <div class="output-text" id="transcriptText"></div>
                    </div>
                    <div id="voiceResponseOutput" class="output hidden">
                        <div class="output-label">AI Response:</div>
                        <div class="output-text" id="voiceResponseText"></div>
                        <audio id="responseAudio" controls></audio>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>💬 Text Chat</h2>
                <div class="input-group">
                    <textarea id="userText" placeholder="Type your message here..." rows="4"></textarea>
                    <button class="btn-primary" onclick="sendText()">Send Message</button>
                    <div id="textResponseOutput" class="output hidden">
                        <div class="output-label">AI Response:</div>
                        <div class="output-text" id="textResponseText"></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="card" style="margin-top: 20px;">
            <h2>⚙️ Settings</h2>
            <div class="button-group">
                <button class="btn-secondary" onclick="testHealth()">🔗 Test Connection</button>
                <button class="btn-secondary" onclick="clearHistory()">🗑️ Clear History</button>
                <button class="btn-danger" onclick="location.reload()">🔄 Reload Page</button>
            </div>
        </div>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let sessionId = Math.random().toString(36).substr(2, 9);

        // Test connection on load
        window.addEventListener('load', testHealth);

        async function testHealth() {{
            try {{
                const response = await fetch('/health');
                const data = await response.json();
                const statusEl = document.getElementById('connectionStatus');
                const sttStatus = data.stt_configured ? '✅' : '❌';
                const apiStatus = data.api_configured ? '✅' : '❌';
                statusEl.innerHTML = `
                    <strong>✅ Backend Connected</strong><br>
                    STT Key: ${{sttStatus}}<br>
                    API Key: ${{apiStatus}}<br>
                    Languages: ${{data.supported_languages.length}}
                `;
                statusEl.className = 'status success';
            }} catch (error) {{
                const statusEl = document.getElementById('connectionStatus');
                statusEl.innerHTML = `<strong>❌ Connection Failed</strong><br>${{error.message}}`;
                statusEl.className = 'status warning';
            }}
        }}

        async function startRecording() {{
            if (!isRecording) {{
                const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = (event) => {{
                    audioChunks.push(event.data);
                }};

                mediaRecorder.onstop = sendAudio;

                mediaRecorder.start();
                isRecording = true;

                const recordBtn = document.getElementById('recordBtn');
                recordBtn.textContent = '⏹️ Stop Recording';
                recordBtn.style.background = '#ff6b6b';

                const statusEl = document.getElementById('recordingStatus');
                statusEl.textContent = '🔴 Recording...';
                statusEl.className = 'recording';
                statusEl.classList.remove('hidden');
            }} else {{
                mediaRecorder.stop();
                isRecording = false;

                const recordBtn = document.getElementById('recordBtn');
                recordBtn.textContent = '🎤 Start Recording';
                recordBtn.style.background = '';

                const statusEl = document.getElementById('recordingStatus');
                statusEl.textContent = '⏳ Processing...';
                statusEl.className = '';
            }}
        }}

        async function sendAudio() {{
            const audioBlob = new Blob(audioChunks, {{ type: 'audio/webm' }});
            const reader = new FileReader();

            reader.onload = async () => {{
                const audioBase64 = reader.result.split(',')[1];

                try {{
                    const response = await fetch('/api/voice-chat', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            audio: audioBase64,
                            session_id: sessionId
                        }})
                    }});

                    const data = await response.json();

                    if (data.success) {{
                        document.getElementById('transcriptText').textContent = data.user_text;
                        document.getElementById('transcriptOutput').classList.remove('hidden');

                        document.getElementById('voiceResponseText').textContent = data.ai_response;
                        document.getElementById('responseAudio').src = 'data:audio/mp3;base64,' + data.audio;
                        document.getElementById('voiceResponseOutput').classList.remove('hidden');
                    }} else {{
                        alert('Error: ' + data.error);
                    }}
                }} catch (error) {{
                    alert('Network error: ' + error.message);
                }} finally {{
                    const statusEl = document.getElementById('recordingStatus');
                    statusEl.classList.add('hidden');
                }}
            }};

            reader.readAsDataURL(audioBlob);
        }}

        async function sendText() {{
            const text = document.getElementById('userText').value.trim();
            if (!text) {{
                alert('Please enter some text');
                return;
            }}

            try {{
                const response = await fetch('/api/text-chat', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        text: text,
                        session_id: sessionId
                    }})
                }});

                const data = await response.json();

                if (data.success) {{
                    document.getElementById('textResponseText').textContent = data.ai_response;
                    document.getElementById('textResponseOutput').classList.remove('hidden');
                    document.getElementById('userText').value = '';
                }} else {{
                    alert('Error: ' + data.error);
                }}
            }} catch (error) {{
                alert('Network error: ' + error.message);
            }}
        }}

        async function clearHistory() {{
            if (!confirm('Clear conversation history?')) return;

            try {{
                const response = await fetch('/api/clear-history', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ session_id: sessionId }})
                }});

                const data = await response.json();
                alert(data.message);
            }} catch (error) {{
                alert('Error: ' + error.message);
            }}
        }}
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Gemini Voice Chatbot on port {port}")
    print(f"🎤 Using GEMINI_STT key for speech-to-text transcription")
    print(f"💬 Using GEMINI_API_KEY for conversation responses")
    print(f"🌍 Supports {len(LANGUAGE_MAPPING)} languages with auto-detection")
    print(f"⚠️ Dual API keys for independent quota management")
    app.run(host='0.0.0.0', port=port, debug=False)
