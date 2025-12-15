import io
import os
import base64
from datetime import datetime
import sqlite3
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from google import genai
from google.genai import types
from gtts import gTTS
import requests
import random

app = Flask(__name__)
CORS(app)

# ========== WEBSOCKET SETUP (No monkey patching needed for production) ==========
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    engineio_logger=False,
    socketio_logger=False
)

# Track connected users
connected_users = {}  # {user_id: session_id}

# ========== CONFIG ==========
GEMINI_KEYS = {
    'key1': os.getenv("GEMINI_KEY1"),
    'key2': os.getenv("GEMINI_KEY2"),
    'key3': os.getenv("GEMINI_KEY3"),
    'key4': os.getenv("GEMINI_KEY4"),
    'key5': os.getenv("GEMINI_KEY5"),
}

ACTIVE_KEYS = {k: v for k, v in GEMINI_KEYS.items() if v}
DATABASE_FILE = "conversations.db"

NESTJS_BACKEND_URL = os.getenv("NESTJS_BACKEND_URL", "https://talleb-5edma.onrender.com")  # Update this!

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

# ========== DATABASE ==========
def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (session_id TEXT, timestamp DATETIME, user_input TEXT, ai_response TEXT, language TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS key_usage
                 (key_name TEXT, timestamp DATETIME, endpoint TEXT, status TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS interview_invitations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  chat_id TEXT NOT NULL,
                  from_user_id TEXT NOT NULL,
                  to_user_id TEXT NOT NULL,
                  from_user_name TEXT,
                  offer_id TEXT,
                  status TEXT DEFAULT 'pending',
                  created_at DATETIME,
                  responded_at DATETIME,
                  UNIQUE(chat_id, from_user_id, to_user_id))''')
    # ⭐ Interview sessions table
    c.execute('''CREATE TABLE IF NOT EXISTS interview_sessions
                 (session_id TEXT PRIMARY KEY,
                  current_question INTEGER DEFAULT 0,
                  questions_asked TEXT,
                  candidate_strengths TEXT,
                  candidate_weaknesses TEXT,
                  interview_notes TEXT,
                  status TEXT DEFAULT 'in_progress',
                  started_at DATETIME,
                  completed_at DATETIME)''')
    conn.commit()
    conn.close()

init_db()

# ========== WEBSOCKET EVENTS ==========

@socketio.on('connect')
def handle_connect():
    """Client connects"""
    print(f"👤 Client connected: {request.sid}")
    emit('connect_response', {'data': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnects"""
    user_id = None
    for uid, sid in list(connected_users.items()):
        if sid == request.sid:
            user_id = uid
            del connected_users[uid]
            break
    if user_id:
        print(f"👤 User {user_id} disconnected")
    else:
        print(f"👤 Client disconnected: {request.sid}")

@socketio.on('join')
def on_join(data):
    """User joins their personal room to receive notifications"""
    user_id = data.get('userId')
    if not user_id:
        emit('error', {'message': 'userId required'})
        return
    
    join_room(f"user_{user_id}")
    connected_users[user_id] = request.sid
    
    print(f"✅ User {user_id} joined room user_{user_id}")
    emit('join_response', {
        'success': True,
        'message': f'Joined room user_{user_id}',
        'user_id': user_id
    })

@socketio.on('leave')
def on_leave(data):
    """User leaves their room"""
    user_id = data.get('userId')
    if user_id and user_id in connected_users:
        leave_room(f"user_{user_id}")
        del connected_users[user_id]
        print(f"⬅️ User {user_id} left room")
    emit('leave_response', {'success': True})

@socketio.on('ping')
def on_ping():
    """Heartbeat"""
    emit('pong')

# ========== HELPER FUNCTIONS ==========
def get_random_key():
    if not gemini_clients:
        return None
    return random.choice(list(gemini_clients.keys()))

def detect_language_from_text(text):
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
        'en': ['hello', 'hi', 'thank', 'please', 'how', 'what'],
        'es': ['hola', 'gracias', 'por favor', 'cómo', 'qué'],
        'fr': ['bonjour', 'merci', 's\'il', 'comment', 'quoi'],
        'de': ['hallo', 'danke', 'bitte', 'wie', 'was'],
        'ar': ['مرحبا', 'شكرا', 'من فضلك', 'كيف', 'ما'],
    }
    
    for lang, keywords in language_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                return lang
    
    return 'en'

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

def get_conversation_history(session_id, limit=10):
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("SELECT user_input, ai_response FROM conversations WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
                  (session_id, limit))
        history = c.fetchall()
        conn.close()
        return list(reversed(history))
    except Exception as e:
        print(f"Database error: {e}")
        return []

# ⭐ Interview session helpers
def get_or_create_interview_session(session_id):
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute("SELECT session_id, current_question, status FROM interview_sessions WHERE session_id=?",
              (session_id,))
    row = c.fetchone()
    if row:
        conn.close()
        return row[1], row[2]

    c.execute(
        "INSERT INTO interview_sessions (session_id, current_question, status, started_at) "
        "VALUES (?, ?, ?, ?)",
        (session_id, 0, 'in_progress', datetime.now())
    )
    conn.commit()
    conn.close()
    return 0, 'in_progress'

def update_interview_session_progress(session_id, current_question, status=None):
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    if status and status == 'completed':
        c.execute(
            "UPDATE interview_sessions SET current_question=?, status=?, completed_at=? WHERE session_id=?",
            (current_question, status, datetime.now(), session_id)
        )
    else:
        c.execute(
            "UPDATE interview_sessions SET current_question=? WHERE session_id=?",
            (current_question, session_id)
        )
    conn.commit()
    conn.close()

# ========== INTERVIEW INVITATION HELPERS ==========
def save_interview_invitation(chat_id, from_user_id, to_user_id, from_user_name, offer_id):
    """Save or update an interview invitation"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        c.execute('''SELECT id, status FROM interview_invitations 
                     WHERE chat_id=? AND from_user_id=? AND to_user_id=?''',
                  (chat_id, from_user_id, to_user_id))
        existing = c.fetchone()
        
        if existing:
            c.execute('''UPDATE interview_invitations 
                         SET status='pending', created_at=?, responded_at=NULL
                         WHERE id=?''',
                      (datetime.now(), existing[0]))
        else:
            c.execute('''INSERT INTO interview_invitations 
                         (chat_id, from_user_id, to_user_id, from_user_name, offer_id, status, created_at)
                         VALUES (?, ?, ?, ?, ?, 'pending', ?)''',
                      (chat_id, from_user_id, to_user_id, from_user_name, offer_id, datetime.now()))
        
        conn.commit()
        invitation_id = existing[0] if existing else c.lastrowid
        conn.close()
        
        return {"success": True, "invitation_id": invitation_id}
    except Exception as e:
        print(f"❌ Error saving invitation: {e}")
        return {"success": False, "error": str(e)}

def get_pending_invitations(user_id):
    """Get all pending invitations for a user"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute('''SELECT id, chat_id, from_user_id, from_user_name, offer_id, created_at
                     FROM interview_invitations 
                     WHERE to_user_id=? AND status='pending'
                     ORDER BY created_at DESC''',
                  (user_id,))
        invitations = c.fetchall()
        conn.close()
        
        return [{
            "invitation_id": inv[0],
            "chat_id": inv[1],
            "from_user_id": inv[2],
            "from_user_name": inv[3],
            "offer_id": inv[4],
            "created_at": inv[5]
        } for inv in invitations]
    except Exception as e:
        print(f"❌ Error getting invitations: {e}")
        return []

def update_invitation_status(invitation_id, status):
    """Update invitation status (accepted/rejected)"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute('''UPDATE interview_invitations 
                     SET status=?, responded_at=?
                     WHERE id=?''',
                  (status, datetime.now(), invitation_id))
        conn.commit()
        conn.close()
        return {"success": True}
    except Exception as e:
        print(f"❌ Error updating invitation: {e}")
        return {"success": False, "error": str(e)}

def get_invitation_by_id(invitation_id):
    """Get invitation details by ID"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute('''SELECT id, chat_id, from_user_id, to_user_id, from_user_name, 
                            offer_id, status, created_at, responded_at
                     FROM interview_invitations WHERE id=?''',
                  (invitation_id,))
        inv = c.fetchone()
        conn.close()
        
        if inv:
            return {
                "invitation_id": inv[0],
                "chat_id": inv[1],
                "from_user_id": inv[2],
                "to_user_id": inv[3],
                "from_user_name": inv[4],
                "offer_id": inv[5],
                "status": inv[6],
                "created_at": inv[7],
                "responded_at": inv[8]
            }
        return None
    except Exception as e:
        print(f"❌ Error getting invitation: {e}")
        return None

# ========== SYSTEM PROMPT BUILDERS ==========
def build_system_prompt(user_details, offer_details, chat_history):
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
- Reference their experience level
- Remember all previous conversation context"""

    if user_details:
        user_context = f"""

CANDIDATE PROFILE:
Name: {user_details.get('name', 'Unknown')}
Experience: {user_details.get('experience_level', 'Not specified')}
Education: {user_details.get('education', 'Not specified')}
Skills: {', '.join(user_details.get('skills', [])) if isinstance(user_details.get('skills'), list) else user_details.get('skills', 'Not specified')}
Country: {user_details.get('country', 'Not specified')}
Languages: {', '.join(user_details.get('languages', [])) if isinstance(user_details.get('languages'), list) else user_details.get('languages', 'Not specified')}"""
        base_prompt += user_context

    if offer_details:
        offer_context = f"""

JOB OPPORTUNITY:
Position: {offer_details.get('position', 'Not specified')}
Company: {offer_details.get('company', 'Not specified')}
Requirements: {', '.join(offer_details.get('required_skills', [])) if isinstance(offer_details.get('required_skills'), list) else offer_details.get('required_skills', 'Not specified')}
Salary: {offer_details.get('salary_range', 'Not specified')}
Location: {offer_details.get('location', 'Not specified')}"""
        base_prompt += offer_context

    if chat_history:
        base_prompt += f"""

CONVERSATION HISTORY (Remember this context):
"""
        for user_msg, ai_msg in chat_history[-5:]:
            base_prompt += f"User: {user_msg}\nCoach: {ai_msg}\n"

    return base_prompt

# ⭐ Professional 5-question structured interview
def build_employer_interview_prompt(user_details, offer_details, chat_history, current_question_index=0):
    """Professional structured mock interview with 5 ordered questions and active listening"""
    
    position_title = offer_details.get('position', 'this position')
    required_skills = offer_details.get('required_skills', [])
    main_skill = required_skills[0] if isinstance(required_skills, list) and required_skills else "your key skills"

    must_ask_questions = [
        f"To start, could you please tell me about yourself and why you are interested in {position_title}?",
        f"Can you describe a challenging project where you used {main_skill}? Please use the STAR method: Situation, Task, Action, Result.",
        "What is one technical or job-specific problem you solved recently? How did you approach it and what was the result?",
        "Tell me about a time you faced a conflict or disagreement at work. How did you handle it and what did you learn?",
        "Where do you see yourself in the next 3–5 years, and how does this role fit into your career goals?"
    ]

    idx = max(0, min(current_question_index, len(must_ask_questions) - 1))

    prompt = f"""You are a professional hiring manager conducting a STRUCTURED MOCK INTERVIEW for "Talleb 5edma".

CANDIDATE INFORMATION:
Name: {user_details.get('name', 'Candidate')}
Experience: {user_details.get('experience_level', 'Not specified')}
Education: {user_details.get('education', 'Not specified')}
Skills: {', '.join(user_details.get('skills', [])) if isinstance(user_details.get('skills'), list) else user_details.get('skills', 'Not specified')}
Country: {user_details.get('country', 'Not specified')}

JOB POSITION:
Title: {offer_details.get('position', 'Not specified')}
Company: {offer_details.get('company', 'Not specified')}
Requirements: {', '.join(required_skills) if isinstance(required_skills, list) else required_skills}

INTERVIEW GUIDELINES:
1. Always greet politely at the beginning of the interview.
2. Ask ONE main question per message (sometimes a short follow-up).
3. Use the 5 MUST-ASK questions IN ORDER.
4. After each answer, briefly react (active listening) in 1 short sentence, then ask the next question.
5. Use a professional, friendly tone.
6. Refer to specific details from the candidate's previous answers when possible.
7. If an answer is vague, ask ONE clarifying follow-up.
8. End the interview after all 5 questions with a brief closing.

5 MUST-ASK QUESTIONS (IN ORDER):
1) Opening / Motivation
2) Project / STAR example
3) Technical / Skills depth
4) Behavioral / Conflict
5) Motivation & Future goals

ACTIVE LISTENING EXAMPLES:
- "Thank you for sharing that."
- "That is a clear example, thank you."
- "I appreciate your honesty about that situation."
- "That shows good problem-solving skills."
Use only ONE short reaction, then continue with the next question.

CURRENT QUESTION INDEX: {idx + 1} of {len(must_ask_questions)}
CURRENT MAIN QUESTION TO ASK NOW:
\"\"\"{must_ask_questions[idx]}\"\"\""""

    if chat_history:
        prompt += """

RECENT CONVERSATION (use this to react naturally):
"""
        for user_msg, ai_msg in chat_history[-5:]:
            prompt += f"Candidate: {user_msg}\nYou: {ai_msg}\n"

    return prompt

# ========== AI FUNCTIONS ==========
def transcribe_with_gemini(audio_bytes, mime_type="audio/webm"):
    if not gemini_clients:
        return "Speech-to-text service not configured.", "en"
    
    available_keys = list(gemini_clients.keys())
    random.shuffle(available_keys)
    
    for key_name in available_keys:
        try:
            client = gemini_clients[key_name]
            print(f"🎤 Transcribing with {key_name.upper()}...")
            
            prompt = """Listen to this audio and transcribe the speech to text.
Detect the language and return ONLY a valid JSON response:
{"text": "transcribed text", "language": "en", "confidence": 0.95}"""
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)],
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=2048)
            )
            
            result_text = response.text.strip() if hasattr(response, 'text') else ""
            if "json" in result_text.lower() and '`' in result_text:
                result_text = result_text.replace('```json', '').replace('```', '').strip()
            
            try:
                result = json.loads(result_text)
                text = result.get("text", "").strip()
                language = result.get("language", "en")
                if not text:
                    return "Could not hear that clearly. Please repeat.", "en"
                print(f"✅ Transcribed: {text}")
                return text, language
            except:
                return "Could not understand audio.", "en"
                
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                print(f"⚠️ {key_name.upper()} quota exceeded")
                continue
            else:
                print(f"❌ {key_name.upper()} error: {e}")
                continue
    
    return "❌ All API keys have hit their quota.", "en"

def get_ai_response(text, user_details, offer_details, chat_history, mode, language, session_id=None):
    if not gemini_clients:
        return "AI service not configured."
    
    available_keys = list(gemini_clients.keys())
    random.shuffle(available_keys)
    
    for key_name in available_keys:
        try:
            client = gemini_clients[key_name]
            
            if mode == 'employer_interview':
                current_q, status = get_or_create_interview_session(session_id or "default")
                system_prompt = build_employer_interview_prompt(user_details, offer_details, chat_history, current_question_index=current_q)
            else:
                system_prompt = build_system_prompt(user_details, offer_details, chat_history)
            
            full_prompt = f"{system_prompt}\n\n--- CURRENT MESSAGE ---\n{text}"
            print(f"📤 Trying {key_name.upper()}...")
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=2048)
            )
            
            ai_text = response.text.strip() if hasattr(response, 'text') else ""
            if ai_text:
                print(f"✅ Response from {key_name.upper()}")
                
                # ⭐ Advance question index for interview mode
                if mode == 'employer_interview':
                    new_q = current_q + 1
                    if new_q >= 5:
                        update_interview_session_progress(session_id or "default", 5, status='completed')
                    else:
                        update_interview_session_progress(session_id or "default", new_q)
                
                return ai_text
            else:
                return "Sorry, I couldn't generate a response."
                
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                print(f"⚠️ {key_name.upper()} quota exceeded")
                continue
            else:
                print(f"❌ {key_name.upper()} error: {e}")
                continue
    
    return "❌ All API keys have hit their quota."

def text_to_speech(text, language='en'):
    try:
        tts_lang = LANGUAGE_MAPPING.get(language, 'en')
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_bytes = audio_buffer.read()
        print(f"✅ Generated TTS: {len(audio_bytes)} bytes")
        return base64.b64encode(audio_bytes).decode('utf-8')
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return None

# ========== ROUTES ==========
@app.route('/')
def index():
    return jsonify({
        "status": "active",
        "message": "Talleb 5edma - Interview Coaching API",
        "features": ["text-chat", "voice-chat", "interview-invitations", "websocket", "structured-interview"],
        "websocket_enabled": True,
        "async_mode": "threading"
    })

@app.route('/api/text-chat', methods=['POST'])
def text_chat():
    try:
        data = request.json
        text = data.get('text', '').strip()
        session_id = data.get('session_id', 'default')
        mode = data.get('mode', 'coaching')
        user_details = data.get('user_details', {})
        offer_details = data.get('offer_details', {})
        chat_history = data.get('chat_history', [])
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        print(f"💬 {mode}: {text}")
        language = detect_language_from_text(text)
        
        question_number = None
        progress = None
        if mode == 'employer_interview':
            current_q, status = get_or_create_interview_session(session_id)
            question_number = current_q + 1
            progress = f"{min(question_number, 5)}/5"
            print(f"📊 Interview Progress: Question {question_number}/5 - Status: {status}")
        
        ai_response = get_ai_response(text, user_details, offer_details, chat_history, mode, language, session_id)
        save_conversation(session_id, text, ai_response, language)
        
        return jsonify({
            "success": True,
            "ai_response": ai_response,
            "mode": mode,
            "language": language,
            "session_id": session_id,
            "question_number": question_number,
            "progress": progress
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/voice-chat', methods=['POST'])
def voice_chat():
    try:
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "No audio file"}), 400
        
        audio_file = request.files['audio']
        if not audio_file or audio_file.filename == '':
            return jsonify({"success": False, "error": "Audio file is empty"}), 400
        
        audio_bytes = audio_file.read()
        if len(audio_bytes) < 500:
            return jsonify({"success": False, "error": "Audio too short"}), 400
        
        session_id = request.form.get('session_id', 'default')
        mode = request.form.get('mode', 'coaching')
        
        try:
            user_details = json.loads(request.form.get('user_details', '{}'))
            offer_details = json.loads(request.form.get('offer_details', '{}'))
        except json.JSONDecodeError:
            return jsonify({"success": False, "error": "Invalid JSON"}), 400
        
        transcribed_text, detected_language = transcribe_with_gemini(audio_bytes, "audio/webm")
        if "error" in transcribed_text.lower():
            return jsonify({"success": False, "error": transcribed_text}), 400
        
        chat_history = get_conversation_history(session_id)
        
        question_number = None
        progress = None
        if mode == 'employer_interview':
            current_q, status = get_or_create_interview_session(session_id)
            question_number = current_q + 1
            progress = f"{min(question_number, 5)}/5"
            print(f"🎧 Interview (voice) Progress: Question {question_number}/5 - Status: {status}")
        
        ai_response = get_ai_response(transcribed_text, user_details, offer_details, chat_history, mode, detected_language, session_id)
        save_conversation(session_id, transcribed_text, ai_response, detected_language)
        
        audio_response = text_to_speech(ai_response, detected_language)
        
        return jsonify({
            "success": True,
            "transcribed_text": transcribed_text,
            "ai_response": ai_response,
            "audio_response": audio_response,
            "language": detected_language,
            "session_id": session_id,
            "question_number": question_number,
            "progress": progress
        })
        
    except Exception as e:
        print(f"❌ Voice Chat Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ========== INTERVIEW INVITATION ENDPOINTS ==========
@app.route('/api/send-interview-invitation', methods=['POST'])
def send_interview_invitation():
    try:
        data = request.json
        chat_id = data.get('chat_id')
        from_user_id = data.get('from_user_id')
        to_user_id = data.get('to_user_id')
        from_user_name = data.get('from_user_name', 'Company')
        offer_id = data.get('offer_id')
        
        if not all([chat_id, from_user_id, to_user_id]):
            return jsonify({"success": False, "error": "Missing required fields"}), 400
        
        print(f"📨 Interview Invitation: {from_user_name} → Student")
        
        result = save_interview_invitation(chat_id, from_user_id, to_user_id, from_user_name, offer_id)
        
        if result["success"]:
            invitation_id = result["invitation_id"]
            invitation = get_invitation_by_id(invitation_id)
            
            socketio.emit('invitation_received', {
                'invitation_id': invitation_id,
                'chat_id': chat_id,
                'from_user_id': from_user_id,
                'from_user_name': from_user_name,
                'offer_id': offer_id,
                'created_at': str(invitation['created_at']) if invitation else '',
                'status': 'pending'
            }, room=f"user_{to_user_id}")
            
            print(f"📤 WebSocket: Sent to user_{to_user_id}")
            
            return jsonify({
                "success": True,
                "message": "Interview invitation sent",
                "invitation_id": invitation_id,
                "websocket_notified": True
            })
        else:
            return jsonify({"success": False, "error": result["error"]}), 500
            
    except Exception as e:
        print(f"❌ Send Invitation Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/get-pending-invitations/<user_id>', methods=['GET'])
def get_user_pending_invitations(user_id):
    try:
        invitations = get_pending_invitations(user_id)
        return jsonify({"success": True, "invitations": invitations, "count": len(invitations)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/accept-interview-invitation', methods=['POST'])
def accept_interview_invitation():
    try:
        data = request.json
        invitation_id = data.get('invitation_id')
        
        if not invitation_id:
            return jsonify({"success": False, "error": "Missing invitation_id"}), 400
        
        invitation = get_invitation_by_id(invitation_id)
        if not invitation:
            return jsonify({"success": False, "error": "Invitation not found"}), 404
        
        if invitation["status"] != "pending":
            return jsonify({"success": False, "error": f"Already {invitation['status']}"}), 400
        
        result = update_invitation_status(invitation_id, "accepted")
        
        if result["success"]:
            socketio.emit('invitation_accepted', {
                'invitation_id': invitation_id,
                'from_user_id': invitation['from_user_id'],
                'to_user_id': invitation['to_user_id']
            }, room=f"user_{invitation['from_user_id']}")
            
            return jsonify({"success": True, "message": "Invitation accepted", "invitation": invitation})
        else:
            return jsonify({"success": False, "error": result["error"]}), 500
            
    except Exception as e:
        print(f"❌ Accept Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/reject-interview-invitation', methods=['POST'])
def reject_interview_invitation():
    try:
        data = request.json
        invitation_id = data.get('invitation_id')
        
        if not invitation_id:
            return jsonify({"success": False, "error": "Missing invitation_id"}), 400
        
        invitation = get_invitation_by_id(invitation_id)
        if not invitation:
            return jsonify({"success": False, "error": "Invitation not found"}), 404
        
        result = update_invitation_status(invitation_id, "rejected")
        
        if result["success"]:
            socketio.emit('invitation_rejected', {
                'invitation_id': invitation_id,
                'from_user_id': invitation['from_user_id']
            }, room=f"user_{invitation['from_user_id']}")
            
            return jsonify({"success": True, "message": "Invitation rejected"})
        else:
            return jsonify({"success": False, "error": result["error"]}), 500
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/quota-status', methods=['GET'])
def quota_status():
    status = {}
    for key_name in gemini_clients.keys():
        try:
            client = gemini_clients[key_name]
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="ping",
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=10)
            )
            status[key_name] = "✅ Available"
        except Exception as e:
            status[key_name] = "❌ Quota Exceeded" if "quota" in str(e).lower() else "⚠️ Error"
    
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "keys": status,
        "total_keys": len(gemini_clients),
        "available_keys": sum(1 for v in status.values() if "Available" in v)
    })

@app.route('/api/ws-status', methods=['GET'])
def ws_status():
    return jsonify({
        "websocket_enabled": True,
        "connected_users": len(connected_users),
        "timestamp": datetime.now().isoformat(),
        "connected_user_ids": list(connected_users.keys())
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "active",
        "websocket_enabled": True,
        "async_mode": "threading",
        "connected_users": len(connected_users),
        "keys_configured": len(gemini_clients)
    })

@app.route('/api/analyze-interview', methods=['POST'])
def analyze_interview():
    """
    Analyze completed interview and send results to NestJS backend
    Expected payload:
    {
      "session_id": "...",
      "user_details": {...},
      "offer_details": {...},
      "chat_id": "...",
      "duration_seconds": 600
    }
    """
    try:
        print("=" * 80)
        print("🔍 INTERVIEW_ANALYSIS: Endpoint called")
        print("=" * 80)
        
        data = request.json
        print(f"📥 INTERVIEW_ANALYSIS: Request data keys: {list(data.keys()) if data else 'None'}")
        
        session_id = data.get('session_id')
        chat_id = data.get('chat_id')
        user_details = data.get('user_details', {})
        offer_details = data.get('offer_details', {})
        duration_seconds = data.get('duration_seconds', 600)
        print(f"📊 INTERVIEW_ANALYSIS: session_id={session_id}")
        print(f"📊 INTERVIEW_ANALYSIS: chat_id={chat_id}")
        print(f"📊 INTERVIEW_ANALYSIS: user_name={user_details.get('name', 'N/A')}")
        print(f"📊 INTERVIEW_ANALYSIS: position={offer_details.get('position', 'N/A')}")
        print(f"📊 INTERVIEW_ANALYSIS: duration={duration_seconds}s")
        if not session_id or not chat_id:
            print("❌ INTERVIEW_ANALYSIS: Missing required fields")
            return jsonify({"success": False, "error": "Missing required fields"}), 400
        print(f"🔍 INTERVIEW_ANALYSIS: Fetching conversation history...")
        
        # Get conversation history
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute(
            "SELECT user_input, ai_response FROM conversations WHERE session_id=? ORDER BY timestamp ASC",
            (session_id,)
        )
        history = c.fetchall()
        conn.close()
        print(f"📚 INTERVIEW_ANALYSIS: Found {len(history)} conversation exchanges")
        if not history or len(history) < 2:
            print(f"⚠️ INTERVIEW_ANALYSIS: Insufficient data ({len(history)} exchanges)")
            return jsonify({
                "success": False,
                "error": "Insufficient interview data"
            }), 400
        # Calculate completion percentage
        completion_percentage = min(100, int((len(history) / 5) * 100))
        print(f"📈 INTERVIEW_ANALYSIS: Completion: {completion_percentage}%")
        # Build analysis prompt
        candidate_name = user_details.get('name', 'Candidate')
        position = offer_details.get('position', 'Position')
        print(f"🤖 INTERVIEW_ANALYSIS: Building AI analysis prompt...")
        analysis_prompt = f"""You are an expert HR analyst. Analyze this interview transcript comprehensively.
CANDIDATE: {candidate_name}
POSITION: {position}
COMPANY: {offer_details.get('company', 'Company')}
INTERVIEW TRANSCRIPT:
"""
        for i, (user_msg, ai_msg) in enumerate(history, 1):
            analysis_prompt += f"\nQ{i} (Interviewer): {ai_msg}\nA{i} (Candidate): {user_msg}\n"
        analysis_prompt += """
Provide a detailed analysis in STRICT JSON format (no markdown, no code blocks):
{
  "overall_score": <0-100>,
  "strengths": ["strength1", "strength2", "strength3"],
  "weaknesses": ["weakness1", "weakness2"],
  "question_analysis": [
    {
      "question": "question text",
      "answer": "answer summary",
      "score": <0-10>,
      "feedback": "specific feedback"
    }
  ],
  "recommendation": "HIRE",
  "summary": "2-3 sentence overall assessment"
}
CRITICAL: 
- Return ONLY valid JSON
- No markdown code blocks
- Escape all quotes in strings
- recommendation must be: STRONG_HIRE, HIRE, MAYBE, or NO_HIRE
- overall_score must be 0-100
- Keep all strings concise and properly escaped"""
        print(f"🔑 INTERVIEW_ANALYSIS: Prompt length: {len(analysis_prompt)} chars")
        # Call Gemini AI
        if not gemini_clients:
            print("❌ INTERVIEW_ANALYSIS: No Gemini clients configured")
            return jsonify({"success": False, "error": "AI service not available"}), 500
        key_name = get_random_key()
        if not key_name:
            print("❌ INTERVIEW_ANALYSIS: No API keys available")
            return jsonify({"success": False, "error": "No AI keys available"}), 500
        print(f"🔑 INTERVIEW_ANALYSIS: Using key: {key_name}")
        client = gemini_clients[key_name]
        print(f"🤖 INTERVIEW_ANALYSIS: Calling Gemini AI...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=analysis_prompt,
            config=types.GenerateContentConfig(
                temperature=0.3, 
                max_output_tokens=4096,
                response_mime_type="application/json"  # Request JSON format
            )
        )
        result_text = response.text.strip() if hasattr(response, 'text') else ""
        print(f"📤 INTERVIEW_ANALYSIS: AI response length: {len(result_text)} chars")
        print(f"📤 INTERVIEW_ANALYSIS: AI response preview: {result_text[:200]}...")
        # Clean and parse JSON with robust error handling
        if "```json" in result_text:
            print(f"🔧 INTERVIEW_ANALYSIS: Cleaning JSON markdown")
            result_text = result_text.replace('```json', '').replace('```', '').strip()
        if "```" in result_text:
            print(f"🔧 INTERVIEW_ANALYSIS: Removing remaining code fence")
            result_text = result_text.replace('```', '').strip()
        print(f"🔍 INTERVIEW_ANALYSIS: Parsing JSON...")
        analysis = None
        
        try:
            analysis = json.loads(result_text)
            print(f"✅ INTERVIEW_ANALYSIS: JSON parsed successfully")
        except json.JSONDecodeError as json_err:
            print(f"⚠️ INTERVIEW_ANALYSIS: Initial JSON parse failed: {json_err}")
            print(f"🔧 INTERVIEW_ANALYSIS: Attempting to fix malformed JSON...")
            
            # Try to truncate at last valid closing brace
            try:
                last_brace = result_text.rfind('}')
                if last_brace > 0:
                    truncated = result_text[:last_brace + 1]
                    print(f"🔧 INTERVIEW_ANALYSIS: Truncated to {len(truncated)} chars")
                    analysis = json.loads(truncated)
                    print(f"✅ INTERVIEW_ANALYSIS: Successfully parsed truncated JSON")
            except Exception as truncate_err:
                print(f"❌ INTERVIEW_ANALYSIS: Truncation failed: {truncate_err}")
            
            # If still failed, create fallback analysis
            if not analysis:
                print(f"⚠️ INTERVIEW_ANALYSIS: Using fallback analysis")
                analysis = {
                    "overall_score": 50,
                    "strengths": ["Participated in interview", "Responded to questions"],
                    "weaknesses": ["Limited detailed responses", "Could provide more specific examples"],
                    "question_analysis": [
                        {
                            "question": f"Question {i+1}",
                            "answer": "Response provided",
                            "score": 5,
                            "feedback": "More detail would strengthen the response"
                        } for i in range(min(len(history), 5))
                    ],
                    "recommendation": "MAYBE",
                    "summary": "Interview completed. Candidate showed basic engagement but could benefit from more detailed and structured responses."
                }
        # Add metadata
        analysis['candidate_name'] = candidate_name
        analysis['position'] = position
        analysis['completion_percentage'] = completion_percentage
        analysis['interview_duration'] = f"{len(history)} exchanges"
        print(f"✅ INTERVIEW_ANALYSIS: Analysis complete")
        print(f"📊 INTERVIEW_ANALYSIS: Score={analysis.get('overall_score', 0)}")
        print(f"📊 INTERVIEW_ANALYSIS: Recommendation={analysis.get('recommendation', 'N/A')}")
        print(f"📊 INTERVIEW_ANALYSIS: Strengths={len(analysis.get('strengths', []))}")
        print(f"📊 INTERVIEW_ANALYSIS: Weaknesses={len(analysis.get('weaknesses', []))}")
        # Send to NestJS backend
        print(f"🌐 INTERVIEW_ANALYSIS: Sending to NestJS backend...")
        print(f"🌐 INTERVIEW_ANALYSIS: NestJS URL: {NESTJS_BACKEND_URL}")
        
        message_sent = False
        try:
            nestjs_payload = {
                "chat_id": chat_id,
                "analysis": analysis,
            }
            print(f"📦 INTERVIEW_ANALYSIS: NestJS payload keys: {list(nestjs_payload.keys())}")
            print(f"📦 INTERVIEW_ANALYSIS: chat_id={chat_id}")
            
            nestjs_response = requests.post(
                f"{NESTJS_BACKEND_URL}/chat/interview-result",
                json=nestjs_payload,
                timeout=10
            )
            print(f"📡 INTERVIEW_ANALYSIS: NestJS response status: {nestjs_response.status_code}")
            print(f"📡 INTERVIEW_ANALYSIS: NestJS response body: {nestjs_response.text[:500]}")
            if nestjs_response.status_code in [200, 201]:
                print(f"✅ INTERVIEW_ANALYSIS: Successfully saved to NestJS")
                message_sent = True
            else:
                print(f"⚠️ INTERVIEW_ANALYSIS: NestJS save failed")
                print(f"⚠️ INTERVIEW_ANALYSIS: Status: {nestjs_response.status_code}")
                print(f"⚠️ INTERVIEW_ANALYSIS: Response: {nestjs_response.text}")
                message_sent = False
        except requests.exceptions.Timeout as timeout_err:
            print(f"⏱️ INTERVIEW_ANALYSIS: NestJS request timeout: {timeout_err}")
            message_sent = False
        except requests.exceptions.ConnectionError as conn_err:
            print(f"🔌 INTERVIEW_ANALYSIS: NestJS connection error: {conn_err}")
            message_sent = False
        except requests.exceptions.RequestException as req_err:
            print(f"⚠️ INTERVIEW_ANALYSIS: NestJS request error: {req_err}")
            message_sent = False
        # Return analysis
        print(f"✅ INTERVIEW_ANALYSIS: Returning response to Android")
        print("=" * 80)
        
        return jsonify({
            "success": True,
            "analysis": analysis,
            "message_sent": message_sent,
            "nestjs_backend": NESTJS_BACKEND_URL
        })
    except Exception as e:
        print("=" * 80)
        print(f"❌ INTERVIEW_ANALYSIS: CRITICAL ERROR")
        print(f"❌ INTERVIEW_ANALYSIS: Error type: {type(e).__name__}")
        print(f"❌ INTERVIEW_ANALYSIS: Error message: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        print("=" * 80)
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Talleb 5edma - Interview Coaching")
    print(f"🚀 Port: {port}")
    print(f"🔑 Active Keys: {len(gemini_clients)}")
    print(f"🧠 Features: Text + Voice Chat + WebSocket + Interview Invitations + Structured Interview")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)