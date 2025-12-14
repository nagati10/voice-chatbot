import io
import os
import base64
from datetime import datetime
import sqlite3
import json
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from google import genai
from google.genai import types
from gtts import gTTS
import random

app = Flask(__name__)
CORS(app)

# ========== WEBSOCKET SETUP ==========
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

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
    # ========== INTERVIEW INVITATIONS TABLE ==========
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
    # Remove user from tracking
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
    """
    User joins their personal room to receive notifications
    
    Expected data: {
        "userId": "student_123"
    }
    """
    user_id = data.get('userId')
    if not user_id:
        emit('error', {'message': 'userId required'})
        return
    
    # Join room named user_{user_id}
    join_room(f"user_{user_id}")
    
    # Track connection
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
    """Heartbeat - keep connection alive"""
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

# ========== INTERVIEW INVITATION HELPERS ==========
def save_interview_invitation(chat_id, from_user_id, to_user_id, from_user_name, offer_id):
    """Save or update an interview invitation"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        # Check if invitation already exists
        c.execute('''SELECT id, status FROM interview_invitations 
                     WHERE chat_id=? AND from_user_id=? AND to_user_id=?''',
                  (chat_id, from_user_id, to_user_id))
        existing = c.fetchone()
        
        if existing:
            # Update existing invitation
            c.execute('''UPDATE interview_invitations 
                         SET status='pending', created_at=?, responded_at=NULL
                         WHERE id=?''',
                      (datetime.now(), existing[0]))
        else:
            # Insert new invitation
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

def build_employer_interview_prompt(user_details, offer_details, chat_history):
    prompt = f"""You are a professional hiring manager conducting a MOCK INTERVIEW for "Talleb 5edma".

CANDIDATE INFORMATION:
Name: {user_details.get('name', 'Candidate')}
Experience: {user_details.get('experience_level', 'Not specified')}
Education: {user_details.get('education', 'Not specified')}
Skills: {', '.join(user_details.get('skills', [])) if isinstance(user_details.get('skills'), list) else user_details.get('skills', 'Not specified')}
Country: {user_details.get('country', 'Not specified')}

JOB POSITION:
Title: {offer_details.get('position', 'Not specified')}
Company: {offer_details.get('company', 'Not specified')}
Requirements: {', '.join(offer_details.get('required_skills', [])) if isinstance(offer_details.get('required_skills'), list) else offer_details.get('required_skills', 'Not specified')}

INTERVIEW GUIDELINES:
1. Start with greeting if first message
2. Ask 1-2 relevant questions per turn
3. Reference their specific experience
4. Ask technical & behavioral questions
5. React naturally to their answers
6. Make it feel like a REAL interview
7. Focus on assessing fit for THIS specific role

IMPORTANT:
- Be professional but approachable
- Ask open-ended questions
- Don't give away answers
- Remember everything they said
- React to previous answers"""

    if chat_history:
        prompt += f"""

CONVERSATION HISTORY (Remember this context):
"""
        for user_msg, ai_msg in chat_history[-5:]:
            prompt += f"Candidate: {user_msg}\nYou (Interviewer): {ai_msg}\n"

    return prompt

# ========== AI FUNCTIONS WITH QUOTA AWARENESS ==========
def transcribe_with_gemini(audio_bytes, mime_type="audio/webm"):
    if not gemini_clients:
        return "Speech-to-text service not configured.", "en"
    
    # Try each key
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
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                )
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
                print(f"✅ Transcribed with {key_name.upper()}: {text}")
                return text, language
            except:
                return "Could not understand audio.", "en"
                
        except Exception as e:
            error_msg = str(e)
            
            # If quota exceeded, try next key
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                print(f"⚠️ {key_name.upper()} quota exceeded, trying next key...")
                continue
            else:
                print(f"❌ {key_name.upper()} error: {e}")
                continue
    
    # If all keys failed
    return "❌ All API keys have hit their quota. STT unavailable.", "en"

def get_ai_response(text, user_details, offer_details, chat_history, mode, language):
    if not gemini_clients:
        return "AI service not configured."
    
    # Try each key - skip ones that hit quota
    available_keys = list(gemini_clients.keys())
    random.shuffle(available_keys)
    
    for key_name in available_keys:
        try:
            client = gemini_clients[key_name]
            
            if mode == 'employer_interview':
                system_prompt = build_employer_interview_prompt(user_details, offer_details, chat_history)
            else:
                system_prompt = build_system_prompt(user_details, offer_details, chat_history)
            
            full_prompt = f"{system_prompt}\n\n--- CURRENT MESSAGE ---\n{text}"
            
            print(f"📤 Trying {key_name.upper()}...")
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=2048,
                )
            )
            
            ai_text = response.text.strip() if hasattr(response, 'text') else ""
            
            if ai_text:
                print(f"✅ Response from {key_name.upper()}: {ai_text[:80]}...")
                return ai_text
            else:
                return "Sorry, I couldn't generate a response."
                
        except Exception as e:
            error_msg = str(e)
            
            # If quota exceeded, try next key
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                print(f"⚠️ {key_name.upper()} quota exceeded, trying next key...")
                continue
            else:
                print(f"❌ {key_name.upper()} error: {e}")
                continue
    
    # If all keys failed
    return "❌ All API keys have hit their quota. Please try again in a few hours or upgrade to a paid plan."

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
        "features": ["text-chat", "voice-chat", "interview-invitations", "websocket"],
        "websocket_enabled": True
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
        
        ai_response = get_ai_response(text, user_details, offer_details, chat_history, mode, language)
        
        save_conversation(session_id, text, ai_response, language)
        
        return jsonify({
            "success": True,
            "ai_response": ai_response,
            "mode": mode,
            "language": language,
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/voice-chat', methods=['POST'])
def voice_chat():
    try:
        print(f"📨 Voice Chat Request:")
        print(f"  Files: {list(request.files.keys())}")
        print(f"  Form data: {list(request.form.keys())}")
        
        if 'audio' not in request.files:
            print(f"❌ No audio file in request")
            return jsonify({
                "success": False,
                "error": "No audio file. Make sure microphone permission is granted."
            }), 400
        
        audio_file = request.files['audio']
        
        if not audio_file or audio_file.filename == '':
            print(f"❌ Audio file is empty")
            return jsonify({
                "success": False,
                "error": "Audio file is empty. Try recording again."
            }), 400
        
        audio_bytes = audio_file.read()
        print(f"📦 Audio size: {len(audio_bytes)} bytes")
        
        if len(audio_bytes) < 500:
            print(f"❌ Audio too short ({len(audio_bytes)} bytes)")
            return jsonify({
                "success": False,
                "error": f"Audio too short. Please record for at least 1 second. (Received: {len(audio_bytes)} bytes)"
            }), 400
        
        session_id = request.form.get('session_id', 'default')
        mode = request.form.get('mode', 'coaching')
        
        print(f"🎙️ Session: {session_id}, Mode: {mode}")
        
        try:
            user_details_str = request.form.get('user_details', '{}')
            offer_details_str = request.form.get('offer_details', '{}')
            print(f"  User details: {user_details_str[:50]}...")
            print(f"  Offer details: {offer_details_str[:50]}...")
            user_details = json.loads(user_details_str)
            offer_details = json.loads(offer_details_str)
        except json.JSONDecodeError as je:
            print(f"❌ Invalid JSON in form data: {je}")
            return jsonify({
                "success": False,
                "error": f"Invalid user/offer details format: {str(je)}"
            }), 400
        
        print(f"✅ Request validated successfully")
        
        print(f"🎤 Starting transcription...")
        transcribed_text, detected_language = transcribe_with_gemini(audio_bytes, "audio/webm")
        print(f"📝 Result: {transcribed_text}")
        
        if "error" in transcribed_text.lower() or "could" in transcribed_text.lower() or "quota" in transcribed_text.lower():
            print(f"⚠️ Transcription failed: {transcribed_text}")
            return jsonify({
                "success": False,
                "error": transcribed_text
            }), 400
        
        print(f"📖 Fetching conversation history...")
        chat_history = get_conversation_history(session_id)
        print(f"  Retrieved {len(chat_history)} previous messages")
        
        print(f"🤖 Generating AI response...")
        ai_response = get_ai_response(transcribed_text, user_details, offer_details, chat_history, mode, detected_language)
        print(f"💬 AI: {ai_response[:80]}...")
        
        save_conversation(session_id, transcribed_text, ai_response, detected_language)
        
        print(f"🔊 Generating audio response...")
        audio_response = text_to_speech(ai_response, detected_language)
        print(f"✅ TTS generated: {len(audio_response) if audio_response else 0} chars")
        
        return jsonify({
            "success": True,
            "transcribed_text": transcribed_text,
            "ai_response": ai_response,
            "audio_response": audio_response,
            "language": detected_language,
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"❌ Voice Chat Error: {e}")
        import traceback
        error_trace = traceback.format_exc()
        print(error_trace)
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

# ========== INTERVIEW INVITATION ENDPOINTS ==========
@app.route('/api/send-interview-invitation', methods=['POST'])
def send_interview_invitation():
    """
    Enterprise sends interview invitation to student
    
    Sends both HTTP response AND WebSocket notification
    """
    try:
        data = request.json
        
        chat_id = data.get('chat_id')
        from_user_id = data.get('from_user_id')
        to_user_id = data.get('to_user_id')
        from_user_name = data.get('from_user_name', 'Company')
        offer_id = data.get('offer_id')
        
        if not all([chat_id, from_user_id, to_user_id]):
            return jsonify({
                "success": False,
                "error": "Missing required fields: chat_id, from_user_id, to_user_id"
            }), 400
        
        print(f"📨 Interview Invitation: {from_user_name} → Student (Chat: {chat_id})")
        
        result = save_interview_invitation(
            chat_id=chat_id,
            from_user_id=from_user_id,
            to_user_id=to_user_id,
            from_user_name=from_user_name,
            offer_id=offer_id
        )
        
        if result["success"]:
            invitation_id = result["invitation_id"]
            
            # Get full invitation details
            invitation = get_invitation_by_id(invitation_id)
            
            # ✨ EMIT WEBSOCKET EVENT TO RECIPIENT ✨
            socketio.emit('invitation_received', {
                'invitation_id': invitation_id,
                'chat_id': chat_id,
                'from_user_id': from_user_id,
                'from_user_name': from_user_name,
                'offer_id': offer_id,
                'created_at': str(invitation['created_at']) if invitation else '',
                'status': 'pending'
            }, room=f"user_{to_user_id}")
            
            print(f"📤 WebSocket: Sent invitation event to user_{to_user_id}")
            
            return jsonify({
                "success": True,
                "message": "Interview invitation sent successfully",
                "invitation_id": invitation_id,
                "chat_id": chat_id,
                "status": "pending",
                "websocket_notified": True
            })
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500
            
    except Exception as e:
        print(f"❌ Send Invitation Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

@app.route('/api/get-pending-invitations/<user_id>', methods=['GET'])
def get_user_pending_invitations(user_id):
    """Get all pending interview invitations for a user (HTTP fallback)"""
    try:
        print(f"📬 Getting pending invitations for user: {user_id}")
        
        invitations = get_pending_invitations(user_id)
        
        return jsonify({
            "success": True,
            "invitations": invitations,
            "count": len(invitations)
        })
        
    except Exception as e:
        print(f"❌ Get Invitations Error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/accept-interview-invitation', methods=['POST'])
def accept_interview_invitation():
    """Student accepts interview invitation"""
    try:
        data = request.json
        invitation_id = data.get('invitation_id')
        
        if not invitation_id:
            return jsonify({
                "success": False,
                "error": "Missing invitation_id"
            }), 400
        
        print(f"✅ Accepting invitation: {invitation_id}")
        
        invitation = get_invitation_by_id(invitation_id)
        
        if not invitation:
            return jsonify({
                "success": False,
                "error": "Invitation not found"
            }), 404
        
        if invitation["status"] != "pending":
            return jsonify({
                "success": False,
                "error": f"Invitation already {invitation['status']}"
            }), 400
        
        result = update_invitation_status(invitation_id, "accepted")
        
        if result["success"]:
            # Emit WebSocket event to enterprise
            socketio.emit('invitation_accepted', {
                'invitation_id': invitation_id,
                'from_user_id': invitation['from_user_id'],
                'to_user_id': invitation['to_user_id'],
                'accepted_at': str(datetime.now())
            }, room=f"user_{invitation['from_user_id']}")
            
            return jsonify({
                "success": True,
                "message": "Interview invitation accepted",
                "invitation": {
                    "invitation_id": invitation_id,
                    "chat_id": invitation["chat_id"],
                    "from_user_id": invitation["from_user_id"],
                    "from_user_name": invitation["from_user_name"],
                    "offer_id": invitation["offer_id"]
                }
            })
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500
            
    except Exception as e:
        print(f"❌ Accept Invitation Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/reject-interview-invitation', methods=['POST'])
def reject_interview_invitation():
    """Student rejects interview invitation"""
    try:
        data = request.json
        invitation_id = data.get('invitation_id')
        
        if not invitation_id:
            return jsonify({
                "success": False,
                "error": "Missing invitation_id"
            }), 400
        
        print(f"❌ Rejecting invitation: {invitation_id}")
        
        invitation = get_invitation_by_id(invitation_id)
        
        if not invitation:
            return jsonify({
                "success": False,
                "error": "Invitation not found"
            }), 404
        
        result = update_invitation_status(invitation_id, "rejected")
        
        if result["success"]:
            # Emit WebSocket event to enterprise
            socketio.emit('invitation_rejected', {
                'invitation_id': invitation_id,
                'from_user_id': invitation['from_user_id'],
                'to_user_id': invitation['to_user_id'],
                'rejected_at': str(datetime.now())
            }, room=f"user_{invitation['from_user_id']}")
            
            return jsonify({
                "success": True,
                "message": "Interview invitation rejected"
            })
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500
            
    except Exception as e:
        print(f"❌ Reject Invitation Error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/quota-status', methods=['GET'])
def quota_status():
    """Check API quota status"""
    status = {}
    for key_name in gemini_clients.keys():
        try:
            client = gemini_clients[key_name]
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="ping",
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=10,
                )
            )
            status[key_name] = "✅ Available"
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                status[key_name] = "❌ Quota Exceeded"
            else:
                status[key_name] = f"⚠️ {str(e)[:50]}"
    
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "keys": status,
        "total_keys": len(gemini_clients),
        "available_keys": sum(1 for v in status.values() if "Available" in v)
    })

@app.route('/api/ws-status', methods=['GET'])
def ws_status():
    """WebSocket status endpoint"""
    return jsonify({
        "websocket_enabled": True,
        "connected_users": len(connected_users),
        "timestamp": datetime.now().isoformat(),
        "connected_user_ids": list(connected_users.keys())
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "active",
        "keys": len(gemini_clients),
        "websocket_enabled": True,
        "connected_users": len(connected_users),
        "endpoints": [
            "/",
            "/api/text-chat",
            "/api/voice-chat",
            "/api/quota-status",
            "/api/ws-status",
            "/api/send-interview-invitation",
            "/api/get-pending-invitations/<user_id>",
            "/api/accept-interview-invitation",
            "/api/reject-interview-invitation"
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Talleb 5edma - Interview Coaching")
    print(f"🚀 Port: {port}")
    print(f"🔑 Active Keys: {len(gemini_clients)}")
    print(f"📍 URL: https://voice-chatbot-k3fe.onrender.com")
    print(f"🧠 Features: Text + Voice Chat + Conversation Memory + Multi-Key Fallback + Interview Invitations + WebSocket")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)