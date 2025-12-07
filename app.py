# Complete app_loadbalanced.py with Context-Aware Coaching
import io
import os
import base64
from datetime import datetime
import sqlite3
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from gtts import gTTS
import random

app = Flask(__name__)
CORS(app)

# ========== CONFIG - MULTI-KEY LOAD BALANCING ==========
GEMINI_KEYS = {
    'key1': os.getenv("GEMINI_KEY1"),
    'key2': os.getenv("GEMINI_KEY2"),
    'key3': os.getenv("GEMINI_KEY3"),
    'key4': os.getenv("GEMINI_KEY4"),
    'key5': os.getenv("GEMINI_KEY5"),
}

ACTIVE_KEYS = {k: v for k, v in GEMINI_KEYS.items() if v}

DATABASE_FILE = "conversations.db"

LANGUAGE_MAPPING = {
    'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it',
    'pt': 'pt', 'ru': 'ru', 'ja': 'ja', 'ko': 'ko', 'zh': 'zh',
    'zh-cn': 'zh', 'zh-tw': 'zh-tw', 'ar': 'ar', 'hi': 'hi',
}

gemini_clients = {}
for key_name, api_key in ACTIVE_KEYS.items():
    gemini_clients[key_name] = genai.Client(api_key=api_key)
    print(f"✅ {key_name.upper()} configured")

if not gemini_clients:
    print("⚠️ WARNING: No Gemini API keys configured!")
else:
    print(f"🎯 Total active keys: {len(gemini_clients)}")
    print(f"📊 Total RPD capacity: {len(gemini_clients) * 20} requests/day")

# ========== DATABASE INITIALIZATION ==========
def init_db():
    """Initialize database with all tables"""
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    
    # Conversations table
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (session_id TEXT, timestamp DATETIME, user_input TEXT, ai_response TEXT, language TEXT)''')
    
    # Key usage tracking
    c.execute('''CREATE TABLE IF NOT EXISTS key_usage
                 (key_name TEXT, timestamp DATETIME, endpoint TEXT, status TEXT)''')
    
    # Session context (NEW)
    c.execute('''CREATE TABLE IF NOT EXISTS session_context
                 (session_id TEXT PRIMARY KEY, user_details TEXT, offer_details TEXT, timestamp DATETIME)''')
    
    conn.commit()
    conn.close()

init_db()

# ========== CONTEXT BUILDER FUNCTIONS ==========
def format_list(items):
    """Format list items for prompt"""
    if not items:
        return "Not specified"
    if isinstance(items, list):
        return '\n'.join([f"• {item}" for item in items])
    return str(items)


def build_system_prompt(user_details, offer_details, chat_history):
    """Build comprehensive system prompt with full context"""
    
    base_prompt = """You are an expert job interview coach for "Talleb 5edma" (طلب خدمة - Interview Preparation Service).

CORE ROLE:
- Help job seekers prepare for interviews and improve their interview skills
- Provide targeted advice based on the job position they're applying for
- Give constructive feedback on their responses and interview techniques
- Build confidence and professionalism

EXPERTISE AREAS:
- Technical interview preparation
- Behavioral interview (STAR method)
- Company research and cultural fit
- Salary negotiation
- Body language and communication skills
- Common interview questions and techniques

COACHING STYLE:
1. Be encouraging and supportive (builds confidence)
2. Give specific, actionable advice (practical tips)
3. Correct mistakes diplomatically (respectful feedback)
4. Ask clarifying questions when needed (understand their situation)
5. Provide example answers when helpful (concrete guidance)
6. Refer back to previous advice (show continuity)
7. Tailor all advice to their specific job opportunity (personalization)

RESPONSE GUIDELINES:
- Always respond in the language the user is using
- Keep responses concise but thorough (2-4 sentences typically, max 8)
- Be specific to their job opportunity
- Reference their experience level in advice
- Consider their education background
- Account for cultural context"""

    # Add user context
    if user_details:
        user_context = f"""

========== ABOUT THE USER ==========
Name: {user_details.get('name', 'Unknown')}
Experience Level: {user_details.get('experience_level', 'Not specified')}
Education: {user_details.get('education', 'Not specified')}
Languages: {', '.join(user_details.get('languages', ['English'])) if isinstance(user_details.get('languages'), list) else user_details.get('languages', 'English')}
Country/Region: {user_details.get('country', 'Not specified')}
Current Role: {user_details.get('current_role', 'Not specified')}
Career Goal: {user_details.get('career_goal', 'Not specified')}"""
        base_prompt += user_context

    # Add job offer context
    if offer_details:
        offer_context = f"""

========== ABOUT THE JOB OPPORTUNITY ==========
Position: {offer_details.get('position', 'Not specified')}
Company: {offer_details.get('company', 'Not specified')}
Industry: {offer_details.get('industry', 'Not specified')}
Job Level: {offer_details.get('job_level', 'Not specified')}

KEY REQUIREMENTS:
{format_list(offer_details.get('required_skills', []))}

PREFERRED QUALIFICATIONS:
{format_list(offer_details.get('preferred_qualifications', []))}

PRIMARY RESPONSIBILITIES:
{format_list(offer_details.get('responsibilities', []))}

COMPENSATION & BENEFITS:
Salary Range: {offer_details.get('salary_range', 'Not specified')}
Benefits: {', '.join(offer_details.get('benefits', [])) if offer_details.get('benefits') else 'Not specified'}

COMPANY INFO:
Size: {offer_details.get('company_size', 'Not specified')}
Culture Focus: {', '.join(offer_details.get('culture_values', [])) if offer_details.get('culture_values') else 'Not specified'}"""
        base_prompt += offer_context

    # Add chat context
    if chat_history and len(chat_history) > 0:
        chat_context = f"""

========== CONVERSATION CONTEXT ==========
Messages Exchanged: {len(chat_history)}
Topics Covered: Interview preparation, technical skills, soft skills, company research"""
        base_prompt += chat_context

    final_note = """

========== IMPORTANT INSTRUCTIONS ==========
1. Remember EVERYTHING the user has told you in this conversation
2. Refer back to previous messages and advice
3. Build upon previous coaching and examples
4. Provide highly personalized coaching based on:
   - Their specific job opportunity
   - Their experience level and background
   - The company they're interviewing for
   - The market and industry context
5. When they practice answers, give SPECIFIC feedback
6. Always tie your advice back to how it will help them in THIS interview
7. If they mention concerns or challenges, address them directly
8. Celebrate their progress and build momentum

FIRST MESSAGE SPECIAL:
If this is their first message, warmly welcome them and briefly summarize:
- The position they're preparing for
- Your understanding of where they need most help
- Your plan to coach them"""

    return base_prompt + final_note


# ========== KEY ROTATION & LOAD BALANCING ==========
def get_random_key():
    """Get a random active key for load balancing"""
    if not gemini_clients:
        return None
    return random.choice(list(gemini_clients.keys()))


def get_round_robin_key():
    """Round-robin key selection"""
    if not gemini_clients:
        return None
    keys = list(gemini_clients.keys())
    return keys[hash(datetime.now().timestamp()) % len(keys)]


def log_key_usage(key_name, endpoint, status="success"):
    """Log which key was used for monitoring"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO key_usage VALUES (?, ?, ?, ?)",
                  (key_name, datetime.now(), endpoint, status))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error logging key usage: {e}")


# ========== AUDIO PREPROCESSING ==========
def preprocess_audio(audio_bytes):
    """Simple audio preprocessing"""
    try:
        print(f"🎵 Processing audio: {len(audio_bytes)} bytes")
        return audio_bytes, "audio/webm"
    except Exception as e:
        print(f"❌ Audio processing error: {e}")
        return audio_bytes, "audio/webm"


# ========== SPEECH-TO-TEXT (Load Balanced) ==========
def transcribe_with_gemini(audio_bytes, mime_type="audio/webm"):
    """Transcribe speech to text using Gemini 2.5 Flash"""
    if not gemini_clients:
        return "Speech-to-text service is not configured.", "en"
    
    key_name = get_random_key()
    client = gemini_clients[key_name]
    
    try:
        print(f"🎤 Transcribing with {key_name.upper()} ({len(audio_bytes)} bytes)...")
        
        prompt = """Listen to this audio and transcribe the speech to text.
Detect the language and return ONLY a valid JSON response:
{"text": "transcribed text", "language": "en", "confidence": 0.95}

Language codes: en, es, fr, ar, zh, ja, ko, ru, de, it, pt, hi
If you cannot understand, return: {"text": "", "language": "unknown", "confidence": 0.0}"""
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                prompt,
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type,)
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                ]
            )
        )
        
        log_key_usage(key_name, "/api/voice-chat", "success")
        
        if not response or not hasattr(response, 'candidates') or not response.candidates:
            print(f"⚠️ Empty response from Gemini STT")
            return "Could not transcribe audio. Please try again.", "en"
        
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
        
        if "json" in result_text.lower() and '`' in result_text:
            result_text = result_text.replace('```json', '').replace('```', '').strip()
        
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
            import re
            text_match = re.search(r'"text"\s*:\s*"([^"]+)"', result_text)
            lang_match = re.search(r'"language"\s*:\s*"([^"]+)"', result_text)
            
            if text_match:
                text = text_match.group(1)
                language = lang_match.group(1) if lang_match else detect_language_from_text(text)
                print(f"✅ Extracted via regex: {text} ({language})")
                return text, language
            
            return "Could not understand the audio. Please try again.", "en"
            
    except Exception as e:
        print(f"❌ Gemini STT error on {key_name}: {e}")
        log_key_usage(key_name, "/api/voice-chat", "error")
        import traceback
        traceback.print_exc()
        return "Error processing audio. Please try again.", "en"


# ========== LANGUAGE DETECTION ==========
def detect_language_from_text(text):
    """Improved language detection from text"""
    if not text or not text.strip():
        return 'en'
    
    text_lower = text.lower().strip()
    
    arabic_chars = set('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
    if any(char in arabic_chars for char in text):
        return 'ar'
    
    if any('\u4e00' <= c <= '\u9fff' for c in text):
        return 'zh'
    
    if any('\u3040' <= c <= '\u309f' for c in text) or any('\u30a0' <= c <= '\u30ff' for c in text):
        return 'ja'
    
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
    """Save conversation to database"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO conversations VALUES (?, ?, ?, ?, ?)",
                  (session_id, datetime.now(), user_input, ai_response, language))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")


def get_conversation_history(session_id, limit=10):
    """Get conversation history for a session"""
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


# ========== CONTEXT-AWARE AI RESPONSE ==========
def get_context_aware_response(text, session_id, language, system_prompt, key_name):
    """Get AI response with full context"""
    if not gemini_clients or key_name not in gemini_clients:
        return "AI service is not configured."
    
    client = gemini_clients[key_name]
    
    full_prompt = f"{system_prompt}\n\n--- USER MESSAGE ---\n{text}"
    
    try:
        print(f"📤 Sending context-aware prompt to {key_name.upper()}...")
        print(f"📝 Prompt length: {len(full_prompt)} chars")
        
        response = client.models.generate_content(
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
        
        log_key_usage(key_name, "/api/voice-chat", "success")
        
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
            print(f"✅ Context-aware response: {ai_text[:100]}...")
            save_conversation(session_id, text, ai_text, language)
            return ai_text
        else:
            print(f"⚠️ Empty response from Gemini")
            return "Sorry, I couldn't generate a response. Please try again."
            
    except Exception as e:
        print(f"❌ Context-aware response error: {e}")
        log_key_usage(key_name, "/api/voice-chat", "error")
        import traceback
        traceback.print_exc()
        return "I'm having trouble connecting right now. Please try again."


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


# ========== CONTEXT ENDPOINTS ==========
@app.route('/api/context/save', methods=['POST'])
def save_context():
    """Save user and offer context to session"""
    try:
        data = request.json
        session_id = data.get('session_id')
        user_details = data.get('user_details', {})
        offer_details = data.get('offer_details', {})
        
        if not session_id:
            return jsonify({"success": False, "error": "Session ID required"}), 400
        
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        c.execute("INSERT OR REPLACE INTO session_context VALUES (?, ?, ?, ?)",
                  (session_id, json.dumps(user_details), json.dumps(offer_details), datetime.now()))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Context saved for session {session_id}")
        print(f"   User: {user_details.get('name', 'Unknown')}")
        print(f"   Position: {offer_details.get('position', 'Not specified')}")
        
        return jsonify({
            "success": True,
            "message": "Context saved successfully",
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"❌ Save context error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/context/get/<session_id>', methods=['GET'])
def get_context(session_id):
    """Retrieve context for a session"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("SELECT user_details, offer_details FROM session_context WHERE session_id=?", (session_id,))
        result = c.fetchone()
        conn.close()
        
        if result:
            return jsonify({
                "success": True,
                "user_details": json.loads(result[0]),
                "offer_details": json.loads(result[1])
            })
        else:
            return jsonify({
                "success": False,
                "error": "Context not found"
            }), 404
            
    except Exception as e:
        print(f"❌ Get context error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ========== MAIN API ENDPOINTS ==========
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "active",
        "total_keys": len(gemini_clients),
        "active_keys": list(gemini_clients.keys()),
        "gemini_api": "gemini-2.5-flash",
        "multilingual": True,
        "supported_languages": list(LANGUAGE_MAPPING.keys()),
        "audio_support": True,
        "speech_recognition": "Gemini-native (Load Balanced)",
        "conversation_engine": "Gemini-native (Load Balanced)",
        "context_awareness": "Enabled",
        "total_rpd_capacity": len(gemini_clients) * 20,
        "load_balancing": "Random rotation across all keys",
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/voice-chat', methods=['POST'])
def voice_chat():
    """Main endpoint - process voice with CONTEXT"""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No JSON data received"}), 400
        
        audio_base64 = data.get('audio')
        session_id = data.get('session_id', 'default_session')
        language_hint = data.get('language_hint', 'auto')
        
        # CONTEXT DATA
        user_details = data.get('user_details', {})
        offer_details = data.get('offer_details', {})
        chat_details = data.get('chat_details', {})
        
        if not audio_base64:
            return jsonify({"success": False, "error": "No audio data provided"}), 400
        
        print(f"📥 Received audio: {len(audio_base64)} chars base64")
        if user_details:
            print(f"👤 User: {user_details.get('name', 'Unknown')}")
        if offer_details:
            print(f"💼 Position: {offer_details.get('position', 'Not specified')}")
        
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
        
        print(f"🤖 Step 2: Getting context-aware AI response...")
        
        # Get chat history
        history = get_conversation_history(session_id, limit=10)
        
        # Build system prompt WITH CONTEXT
        system_prompt = build_system_prompt(user_details, offer_details, history)
        
        # Select a random key for load balancing
        key_name = get_random_key()
        
        # Get AI response with context
        ai_response = get_context_aware_response(
            text=user_text,
            session_id=session_id,
            language=detected_language,
            system_prompt=system_prompt,
            key_name=key_name
        )
        
        print(f"🔊 Step 3: Generating speech in {detected_language}...")
        audio_response = text_to_speech(ai_response, detected_language)
        
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
            "response_language": detected_language,
            "session_id": session_id,
            "context_used": {
                "user_name": user_details.get('name', 'Unknown'),
                "position": offer_details.get('position', 'Not specified'),
                "company": offer_details.get('company', 'Not specified'),
                "api_key_used": key_name
            }
        })
        
    except Exception as e:
        print(f"❌ Voice chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e), "message": "Internal server error"}), 500


@app.route('/api/text-chat', methods=['POST'])
def text_chat():
    """Text-only endpoint with context"""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No JSON data received"}), 400
        
        text = data.get('text')
        session_id = data.get('session_id', 'default_session')
        language_hint = data.get('language_hint', 'auto')
        
        # CONTEXT DATA
        user_details = data.get('user_details', {})
        offer_details = data.get('offer_details', {})
        chat_details = data.get('chat_details', {})
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        print(f"💬 Text chat: {text}")
        if user_details:
            print(f"👤 User: {user_details.get('name', 'Unknown')}")
        
        detected_language = detect_language_from_text(text)
        
        if language_hint != 'auto' and language_hint in LANGUAGE_MAPPING:
            detected_language = language_hint
        
        print(f"🌐 Language: {detected_language}")
        
        # Get chat history
        history = get_conversation_history(session_id, limit=10)
        
        # Build system prompt WITH CONTEXT
        system_prompt = build_system_prompt(user_details, offer_details, history)
        
        # Select a random key for load balancing
        key_name = get_random_key()
        
        # Get context-aware response
        ai_response = get_context_aware_response(
            text=text,
            session_id=session_id,
            language=detected_language,
            system_prompt=system_prompt,
            key_name=key_name
        )
        
        print(f"✅ AI response ({detected_language}): {ai_response[:100]}...")
        
        return jsonify({
            "success": True,
            "ai_response": ai_response,
            "detected_language": detected_language,
            "response_language": detected_language,
            "session_id": session_id,
            "context_used": {
                "user_name": user_details.get('name', 'Unknown'),
                "position": offer_details.get('position', 'Not specified'),
                "company": offer_details.get('company', 'Not specified'),
                "api_key_used": key_name
            }
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


@app.route('/api/key-usage', methods=['GET'])
def key_usage():
    """Get API key usage statistics"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("""
            SELECT key_name, COUNT(*) as total_requests, 
                   SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) as successful,
                   SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as errors
            FROM key_usage
            WHERE timestamp > datetime('now', '-1 day')
            GROUP BY key_name
        """)
        results = c.fetchall()
        conn.close()
        
        usage = {}
        for key_name, total, success, errors in results:
            usage[key_name] = {
                "total_requests": total,
                "successful": success or 0,
                "errors": errors or 0,
                "success_rate": f"{(success or 0) / total * 100:.1f}%" if total > 0 else "0%"
            }
        
        return jsonify({
            "active_keys": len(gemini_clients),
            "total_capacity_rpd": len(gemini_clients) * 20,
            "usage_last_24h": usage,
            "load_balancing_status": "active"
        })
        
    except Exception as e:
        print(f"❌ Key usage error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/')
def index():
    """Homepage with context-aware coaching"""
    keys_status = "<br>".join([f"✅ {k.upper()}" for k in gemini_clients.keys()])
    total_rpd = len(gemini_clients) * 20
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎙️ Talleb 5edma - Interview Coaching (Load Balanced)</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        .header h1 {{ margin-bottom: 10px; color: #333; }}
        .session-banner {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .session-banner.new {{ background: #e8f5e9; border-left-color: #4caf50; }}
        .session-banner.existing {{ background: #fff3e0; border-left-color: #ff9800; }}
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
            line-height: 1.8;
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
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
        .card h2 {{ margin-bottom: 20px; color: #333; font-size: 20px; }}
        textarea {{
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            resize: vertical;
        }}
        textarea:focus {{ outline: none; border-color: #667eea; }}
        button {{
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            transition: all 0.3s ease;
            margin-top: 10px;
        }}
        button:hover {{ transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }}
        button.secondary {{ background: #f0f0f0; color: #333; }}
        .output {{
            margin-top: 15px;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
            min-height: 60px;
            border-left: 4px solid #667eea;
        }}
        .output.hidden {{ display: none; }}
        audio {{ width: 100%; margin-top: 10px; }}
        @media (max-width: 768px) {{
            .main-content {{ grid-template-columns: 1fr; }}
            .status-box {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎙️ Talleb 5edma - Interview Coaching</h1>
            <p>Context-Aware AI Interview Coach with Multi-Key Load Balancing</p>
            
            <div id="sessionBanner" class="session-banner new">
                <div>
                    <span id="sessionIcon">🆕</span>
                    <span id="sessionText">New Conversation</span> | 
                    <span id="sessionId" style="font-family: monospace; font-size: 12px;"></span>
                </div>
                <button class="secondary" onclick="startNewSession()" style="margin: 0;">New Chat</button>
            </div>
            
            <div class="status-box">
                <div class="status">
                    <strong>🔑 Active API Keys:</strong><br>
                    {keys_status}<br><br>
                    <strong>📊 Total RPD:</strong> {total_rpd}/day
                </div>
                <div class="status" id="connectionStatus">
                    <strong>Testing connection...</strong>
                </div>
            </div>
        </div>

        <div class="main-content">
            <div class="card">
                <h2>🎤 Voice Chat</h2>
                <button onclick="startRecording()" id="recordBtn">🎤 Start Recording</button>
                <div id="voiceResponseOutput" class="output hidden">
                    <div id="voiceResponseText"></div>
                    <audio id="responseAudio" controls></audio>
                </div>
            </div>

            <div class="card">
                <h2>💬 Text Chat</h2>
                <textarea id="userText" placeholder="Type your message..." rows="4"></textarea>
                <button onclick="sendText()">Send Message</button>
                <div id="textResponseOutput" class="output hidden">
                    <div id="textResponseText"></div>
                </div>
            </div>
        </div>

        <div class="card" style="margin-top: 20px;">
            <h2>📊 System Status</h2>
            <button class="secondary" onclick="checkKeyUsage()">🔑 View Key Usage</button>
            <button class="secondary" onclick="loadSessions()">📋 View Sessions</button>
            <div id="statsOutput"></div>
        </div>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let sessionId = localStorage.getItem('sessionId') || Math.random().toString(36).substr(2, 9);
        let isNewSession = !localStorage.getItem('sessionId');

        window.addEventListener('load', () => {{
            localStorage.setItem('sessionId', sessionId);
            updateSessionDisplay();
            testHealth();
        }});

        function updateSessionDisplay() {{
            const banner = document.getElementById('sessionBanner');
            const icon = document.getElementById('sessionIcon');
            const text = document.getElementById('sessionText');
            const idEl = document.getElementById('sessionId');
            
            idEl.textContent = sessionId;
            
            if (isNewSession) {{
                banner.className = 'session-banner new';
                icon.textContent = '🆕';
                text.textContent = 'New Conversation';
            }} else {{
                banner.className = 'session-banner existing';
                icon.textContent = '📦';
                text.textContent = 'Existing Conversation';
            }}
        }}

        function startNewSession() {{
            if (!confirm('Start a new conversation?')) return;
            sessionId = Math.random().toString(36).substr(2, 9);
            localStorage.setItem('sessionId', sessionId);
            isNewSession = true;
            location.reload();
        }}

        async function testHealth() {{
            try {{
                const response = await fetch('/health');
                const data = await response.json();
                document.getElementById('connectionStatus').innerHTML = `
                    <strong>✅ Backend Connected</strong><br>
                    Active Keys: ${{data.total_keys}}<br>
                    Total RPD: ${{data.total_rpd_capacity}}<br>
                    Context Aware: Yes
                `;
            }} catch (error) {{
                document.getElementById('connectionStatus').innerHTML = '❌ Connection Failed<br>' + error.message;
            }}
        }}

        async function startRecording() {{
            if (!isRecording) {{
                try {{
                    const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
                    mediaRecorder.onstop = sendAudio;
                    mediaRecorder.start();
                    isRecording = true;
                    document.getElementById('recordBtn').textContent = '⏹️ Stop Recording';
                }} catch (error) {{
                    alert('Microphone error: ' + error.message);
                }}
            }} else {{
                mediaRecorder.stop();
                isRecording = false;
                document.getElementById('recordBtn').textContent = '🎤 Start Recording';
            }}
        }}

        async function sendAudio() {{
            const blob = new Blob(audioChunks, {{ type: 'audio/webm' }});
            const reader = new FileReader();
            reader.onload = async () => {{
                const base64 = reader.result.split(',')[1];
                try {{
                    const response = await fetch('/api/voice-chat', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            audio: base64,
                            session_id: sessionId
                        }})
                    }});
                    const data = await response.json();
                    if (data.success) {{
                        document.getElementById('voiceResponseText').textContent = data.ai_response;
                        document.getElementById('responseAudio').src = 'data:audio/mp3;base64,' + data.audio;
                        document.getElementById('voiceResponseOutput').classList.remove('hidden');
                        isNewSession = false;
                        updateSessionDisplay();
                    }} else {{
                        alert('Error: ' + data.error);
                    }}
                }} catch (error) {{
                    alert('Error: ' + error.message);
                }}
            }};
            reader.readAsDataURL(blob);
        }}

        async function sendText() {{
            const text = document.getElementById('userText').value.trim();
            if (!text) return alert('Enter text');
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
                    isNewSession = false;
                    updateSessionDisplay();
                }} else {{
                    alert('Error: ' + data.error);
                }}
            }} catch (error) {{
                alert('Error: ' + error.message);
            }}
        }}

        async function checkKeyUsage() {{
            try {{
                const response = await fetch('/api/key-usage');
                const data = await response.json();
                let html = '<table style="width:100%; border-collapse: collapse; margin-top: 15px;"><tr style="background:#f0f0f0;"><th style="border:1px solid #ddd; padding:10px;">Key</th><th style="border:1px solid #ddd; padding:10px;">Requests</th><th style="border:1px solid #ddd; padding:10px;">Success</th><th style="border:1px solid #ddd; padding:10px;">Rate</th></tr>';
                for (const [key, stats] of Object.entries(data.usage_last_24h)) {{
                    html += `<tr><td style="border:1px solid #ddd; padding:10px;">${{key.toUpperCase()}}</td><td style="border:1px solid #ddd; padding:10px;">${{stats.total_requests}}</td><td style="border:1px solid #ddd; padding:10px;">${{stats.successful}}</td><td style="border:1px solid #ddd; padding:10px;">${{stats.success_rate}}</td></tr>`;
                }}
                html += '</table>';
                document.getElementById('statsOutput').innerHTML = html;
            }} catch (error) {{
                alert('Error: ' + error.message);
            }}
        }}

        async function loadSessions() {{
            try {{
                const response = await fetch('/api/sessions');
                if (response.ok) {{
                    const data = await response.json();
                    alert('Sessions: ' + JSON.stringify(data, null, 2));
                }} else {{
                    alert('Sessions endpoint not found');
                }}
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
    print(f"🚀 Starting Talleb 5edma - Interview Coaching (Load Balanced + Context-Aware)")
    print(f"🚀 Port: {port}")
    print(f"🔑 Active API Keys: {len(gemini_clients)}")
    print(f"📊 Total RPD Capacity: {len(gemini_clients) * 20}")
    print(f"⚖️ Load Balancing: Enabled (Random Rotation)")
    print(f"🧠 Context Awareness: Enabled")
    print(f"💾 Session Management: Enabled")
    print(f"🌍 Languages Supported: {len(LANGUAGE_MAPPING)}")
    app.run(host='0.0.0.0', port=port, debug=False)