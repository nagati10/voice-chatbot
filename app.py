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
    
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (session_id TEXT, timestamp DATETIME, user_input TEXT, ai_response TEXT, language TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS key_usage
                 (key_name TEXT, timestamp DATETIME, endpoint TEXT, status TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS session_context
                 (session_id TEXT PRIMARY KEY, user_details TEXT, offer_details TEXT, chat_history TEXT, timestamp DATETIME)''')
    
    conn.commit()
    conn.close()

init_db()

# ========== HELPER FUNCTIONS ==========
def format_list(items):
    """Format list items for prompt"""
    if not items:
        return "Not specified"
    if isinstance(items, list):
        return '\n'.join([f"• {item}" for item in items])
    return str(items)

def get_random_key():
    """Get a random active key for load balancing"""
    if not gemini_clients:
        return None
    return random.choice(list(gemini_clients.keys()))

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

# ========== SYSTEM PROMPT BUILDERS ==========
def build_system_prompt(user_details, offer_details, chat_history):
    """Build coaching system prompt"""
    base_prompt = """You are an expert job interview coach for "Talleb 5edma" (طلب خدمة - Interview Preparation Service).

CORE ROLE:
- Help job seekers prepare for interviews
- Provide targeted advice based on the job position
- Give constructive feedback on responses
- Build confidence and professionalism

RESPONSE GUIDELINES:
- Always respond in the language the user is using
- Keep responses concise but thorough (2-4 sentences)
- Be specific to their job opportunity
- Reference their experience level"""

    if user_details:
        user_context = f"""

========== ABOUT THE USER ==========
Name: {user_details.get('name', 'Unknown')}
Experience: {user_details.get('experience_level', 'Not specified')}
Education: {user_details.get('education', 'Not specified')}
Skills: {', '.join(user_details.get('skills', [])) if isinstance(user_details.get('skills'), list) else user_details.get('skills', 'Not specified')}"""
        base_prompt += user_context

    if offer_details:
        offer_context = f"""

========== ABOUT THE JOB OPPORTUNITY ==========
Position: {offer_details.get('position', 'Not specified')}
Company: {offer_details.get('company', 'Not specified')}
Requirements: {', '.join(offer_details.get('required_skills', [])) if isinstance(offer_details.get('required_skills'), list) else offer_details.get('required_skills', 'Not specified')}
Salary: {offer_details.get('salary_range', 'Not specified')}"""
        base_prompt += offer_context

    return base_prompt

def build_employer_interview_prompt(user_details, offer_details, chat_history):
    """Build employer role play system prompt"""
    prompt = f"""You are a professional hiring manager conducting a MOCK INTERVIEW.

CANDIDATE INFORMATION:
Name: {user_details.get('name', 'Candidate')}
Experience: {user_details.get('experience_level', 'Not specified')}
Education: {user_details.get('education', 'Not specified')}
Skills: {', '.join(user_details.get('skills', [])) if isinstance(user_details.get('skills'), list) else user_details.get('skills', 'Not specified')}

JOB POSITION:
Title: {offer_details.get('position', 'Not specified')}
Company: {offer_details.get('company', 'Not specified')}
Requirements: {', '.join(offer_details.get('required_skills', [])) if isinstance(offer_details.get('required_skills'), list) else offer_details.get('required_skills', 'Not specified')}

INTERVIEW GUIDELINES:
1. Start with greeting if first message
2. Ask 1-2 relevant questions per turn
3. Reference their specific experience
4. Ask technical questions
5. Probe for gaps or areas that need clarification
6. Provide constructive feedback
7. Evaluate fit for role
8. Remember everything they said

IMPORTANT:
- Be professional but approachable
- Ask open-ended questions (not yes/no)
- React naturally to their answers
- Make it feel like a REAL interview
- Don't give away answers
- Focus on assessing fit for THIS specific role"""

    return prompt

# ========== AI RESPONSE FUNCTIONS ==========
def transcribe_with_gemini(audio_bytes, mime_type="audio/webm"):
    """Transcribe speech to text using Gemini"""
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
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
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
        
        result_text = None
        if hasattr(response, 'text') and response.text:
            result_text = response.text.strip()
        elif response and hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    result_text = candidate.content.parts[0].text.strip()
        
        if not result_text:
            return "Could not transcribe audio. Please try again.", "en"
        
        if "json" in result_text.lower() and '`' in result_text:
            result_text = result_text.replace('```json', '').replace('```', '').strip()
        
        try:
            result = json.loads(result_text)
            text = result.get("text", "").strip()
            language = result.get("language", "unknown")
            confidence = result.get("confidence", 0)
            
            if confidence > 0.1 and text:
                if language == "unknown" and text:
                    language = detect_language_from_text(text)
                return text, language
            else:
                return "Could you please repeat that more clearly?", "en"
                
        except json.JSONDecodeError:
            import re
            text_match = re.search(r'"text"\s*:\s*"([^"]+)"', result_text)
            lang_match = re.search(r'"language"\s*:\s*"([^"]+)"', result_text)
            
            if text_match:
                text = text_match.group(1)
                language = lang_match.group(1) if lang_match else detect_language_from_text(text)
                return text, language
            
            return "Could not understand the audio. Please try again.", "en"
            
    except Exception as e:
        print(f"❌ Gemini STT error on {key_name}: {e}")
        log_key_usage(key_name, "/api/voice-chat", "error")
        return "Error processing audio. Please try again.", "en"

def get_context_aware_response(text, session_id, language, system_prompt, key_name):
    """Get AI response with full context"""
    if not gemini_clients or key_name not in gemini_clients:
        return "AI service is not configured."
    
    client = gemini_clients[key_name]
    full_prompt = f"{system_prompt}\n\n--- USER MESSAGE ---\n{text}"
    
    try:
        print(f"📤 Sending to {key_name.upper()}...")
        
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
        
        log_key_usage(key_name, "/api/text-chat", "success")
        
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
            print(f"✅ Response: {ai_text[:100]}...")
            save_conversation(session_id, text, ai_text, language)
            return ai_text
        else:
            return "Sorry, I couldn't generate a response. Please try again."
            
    except Exception as e:
        print(f"❌ Error: {e}")
        log_key_usage(key_name, "/api/text-chat", "error")
        return "I'm having trouble connecting right now. Please try again."

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
        return base64.b64encode(audio_bytes).decode('utf-8')
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return None

# ========== API ENDPOINTS ==========
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "active",
        "total_keys": len(gemini_clients),
        "active_keys": list(gemini_clients.keys()),
        "gemini_api": "gemini-2.5-flash",
        "endpoints": ["/api/text-chat", "/api/voice-chat", "/api/context/save"],
        "total_rpd_capacity": len(gemini_clients) * 20,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/text-chat', methods=['POST'])
def text_chat():
    """Text chat endpoint - supports both coaching and employer_interview modes"""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No JSON data received"}), 400
        
        text = data.get('text')
        session_id = data.get('session_id', 'default_session')
        mode = data.get('mode', 'coaching')
        
        user_details = data.get('user_details', {})
        offer_details = data.get('offer_details', {})
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        print(f"💬 {mode.upper()} Mode: {text}")
        
        detected_language = detect_language_from_text(text)
        history = get_conversation_history(session_id, limit=10)
        
        if mode == 'employer_interview':
            system_prompt = build_employer_interview_prompt(user_details, offer_details, history)
        else:
            system_prompt = build_system_prompt(user_details, offer_details, history)
        
        key_name = get_random_key()
        
        ai_response = get_context_aware_response(
            text=text,
            session_id=session_id,
            language=detected_language,
            system_prompt=system_prompt,
            key_name=key_name
        )
        
        return jsonify({
            "success": True,
            "ai_response": ai_response,
            "mode": mode,
            "detected_language": detected_language,
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/voice-chat', methods=['POST'])
def voice_chat():
    """Voice chat endpoint - transcribes audio and responds"""
    try:
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        session_id = request.form.get('session_id', 'default_session')
        mode = request.form.get('mode', 'coaching')
        
        user_details = json.loads(request.form.get('user_details', '{}'))
        offer_details = json.loads(request.form.get('offer_details', '{}'))
        
        if not audio_file:
            return jsonify({"success": False, "error": "No audio data"}), 400
        
        print(f"🎤 Voice chat received ({len(audio_file.read())} bytes)")
        audio_file.seek(0)
        audio_bytes = audio_file.read()
        
        # Step 1: Transcribe audio
        transcribed_text, detected_language = transcribe_with_gemini(audio_bytes, "audio/webm")
        print(f"✅ Transcribed: {transcribed_text}")
        
        if not transcribed_text or "Error" in transcribed_text or "Could" in transcribed_text:
            return jsonify({
                "success": False,
                "error": transcribed_text,
                "detected_language": detected_language
            }), 400
        
        # Step 2: Build system prompt based on mode
        history = get_conversation_history(session_id, limit=10)
        
        if mode == 'employer_interview':
            system_prompt = build_employer_interview_prompt(user_details, offer_details, history)
        else:
            system_prompt = build_system_prompt(user_details, offer_details, history)
        
        # Step 3: Get AI response
        key_name = get_random_key()
        ai_response = get_context_aware_response(
            text=transcribed_text,
            session_id=session_id,
            language=detected_language,
            system_prompt=system_prompt,
            key_name=key_name
        )
        
        # Step 4: Convert response to speech
        audio_response_b64 = text_to_speech(ai_response, detected_language)
        
        return jsonify({
            "success": True,
            "transcribed_text": transcribed_text,
            "ai_response": ai_response,
            "audio_response": audio_response_b64,
            "mode": mode,
            "detected_language": detected_language,
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"❌ Voice chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

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
        
        c.execute("INSERT OR REPLACE INTO session_context VALUES (?, ?, ?, ?, ?)",
                  (session_id, json.dumps(user_details), json.dumps(offer_details), 
                   '[]', datetime.now()))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Context saved for session {session_id}")
        
        return jsonify({
            "success": True,
            "message": "Context saved successfully",
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"❌ Save context error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/')
def index():
    """Simple home page"""
    return """
    <html>
    <head><title>Talleb 5edma - Interview Coach API</title></head>
    <body style="font-family: Arial; margin: 40px;">
        <h1>🎙️ Talleb 5edma - Interview Coaching API</h1>
        <p>✅ API is running and ready to receive requests</p>
        <h3>Endpoints:</h3>
        <ul>
            <li><strong>GET /health</strong> - Check API status</li>
            <li><strong>POST /api/text-chat</strong> - Send text message</li>
            <li><strong>POST /api/voice-chat</strong> - Send audio file</li>
            <li><strong>POST /api/context/save</strong> - Save user context</li>
        </ul>
        <p><strong>Modes supported:</strong> coaching, employer_interview</p>
    </body>
    </html>
    """

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Talleb 5edma - Interview Coaching")
    print(f"🚀 Port: {port}")
    print(f"🔑 Active API Keys: {len(gemini_clients)}")
    print(f"📊 Total RPD Capacity: {len(gemini_clients) * 20}")
    print(f"🧠 Features: Text Chat + Voice Chat + Employer Mode")
    app.run(host='0.0.0.0', port=port, debug=False)
