import io
import os
import base64
from datetime import datetime
import sqlite3
import json
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from google import genai
from google.genai import types
from gtts import gTTS
import random

app = Flask(__name__)
CORS(app)

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
    'en': 'en',
    'es': 'es',
    'fr': 'fr',
    'de': 'de',
    'it': 'it',
    'pt': 'pt',
    'ru': 'ru',
    'ja': 'ja',
    'ko': 'ko',
    'zh': 'zh',
    'zh-cn': 'zh',
    'zh-tw': 'zh-tw',
    'ar': 'ar',
    'hi': 'hi',
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
                 (session_id TEXT, timestamp DATETIME, user_input TEXT,
                  ai_response TEXT, language TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS key_usage
                 (key_name TEXT, timestamp DATETIME, endpoint TEXT, status TEXT)''')
    conn.commit()
    conn.close()


init_db()


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
        c.execute("SELECT user_input, ai_response FROM conversations "
                  "WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
                  (session_id, limit))
        history = c.fetchall()
        conn.close()
        return list(reversed(history))
    except Exception as e:
        print(f"Database error: {e}")
        return []


# ========== SYSTEM PROMPT BUILDERS ==========
def build_system_prompt(user_details, offer_details, chat_history):
    base_prompt = """
You are an expert job interview coach for "Talleb 5edma" (طلب خدمة - Interview Preparation Service).
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
- Remember all previous conversation context
"""

    if user_details:
        user_context = f"""
CANDIDATE PROFILE:
Name: {user_details.get('name', 'Unknown')}
Experience: {user_details.get('experience_level', 'Not specified')}
Education: {user_details.get('education', 'Not specified')}
Skills: {', '.join(user_details.get('skills', [])) if isinstance(user_details.get('skills'), list) else user_details.get('skills', 'Not specified')}
Country: {user_details.get('country', 'Not specified')}
Languages: {', '.join(user_details.get('languages', [])) if isinstance(user_details.get('languages'), list) else user_details.get('languages', 'Not specified')}
"""
        base_prompt += user_context

    if offer_details:
        offer_context = f"""
JOB OPPORTUNITY:
Position: {offer_details.get('position', 'Not specified')}
Company: {offer_details.get('company', 'Not specified')}
Requirements: {', '.join(offer_details.get('required_skills', [])) if isinstance(offer_details.get('required_skills'), list) else offer_details.get('required_skills', 'Not specified')}
Salary: {offer_details.get('salary_range', 'Not specified')}
Location: {offer_details.get('location', 'Not specified')}
"""
        base_prompt += offer_context

    if chat_history:
        base_prompt += f"""
CONVERSATION HISTORY (Remember this context):
"""
        for user_msg, ai_msg in chat_history[-5:]:
            base_prompt += f"User: {user_msg}\nCoach: {ai_msg}\n"

    return base_prompt


def build_employer_interview_prompt(user_details, offer_details, chat_history):
    prompt = f"""
You are a professional hiring manager conducting a MOCK INTERVIEW for "Talleb 5edma".
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
- React to previous answers
"""
    if chat_history:
        prompt += """
CONVERSATION HISTORY (Remember this context):
"""
        for user_msg, ai_msg in chat_history[-5:]:
            prompt += f"Candidate: {user_msg}\nYou (Interviewer): {ai_msg}\n"

    return prompt


# ========== AI FUNCTIONS WITH QUOTA AWARENESS ==========
def transcribe_with_gemini(audio_bytes, mime_type="audio/webm"):
    if not gemini_clients:
        return "Speech-to-text service not configured.", "en"

    available_keys = list(gemini_clients.keys())
    random.shuffle(available_keys)

    for key_name in available_keys:
        try:
            client = gemini_clients[key_name]
            print(f"🎤 Transcribing with {key_name.upper()}...")

            prompt = """
Listen to this audio and transcribe the speech to text.
Detect the language and return ONLY a valid JSON response:
{"text": "transcribed text", "language": "en", "confidence": 0.95}
"""
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

                print(f"... ✅ Transcribed with {key_name.upper()}: {text}")
                return text, language
            except Exception:
                return "Could not understand audio.", "en"

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                print(f"⚠️ {key_name.upper()} quota exceeded, trying next key...")
                continue
            else:
                print(f"❌ {key_name.upper()} error: {e}")
                continue

    # FIX #1: If all keys failed - ALWAYS return tuple
    return "Could not transcribe audio. All API keys exhausted.", "en"


def get_ai_response(text, user_details, offer_details, chat_history, mode, language):
    if not gemini_clients:
        return "AI service not configured."

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
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                print(f"⚠️ {key_name.upper()} quota exceeded, trying next key...")
                continue
            else:
                print(f"❌ {key_name.upper()} error: {e}")
                continue

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
        return audio_bytes
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return None


# ========== ROUTES ==========

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Talleb 5edma - Interview Coach</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 30px;
        }
        
        h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #7f8c8d;
            margin-bottom: 30px;
            font-size: 14px;
        }
        
        .mode-selector {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 30px;
        }
        
        .mode-btn {
            padding: 12px 20px;
            border: 2px solid #ecf0f1;
            background: white;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s;
        }
        
        .mode-btn.active {
            border-color: #3498db;
            background: #3498db;
            color: white;
        }
        
        .section {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #2c3e50;
            font-weight: 500;
        }
        
        input, textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ecf0f1;
            border-radius: 6px;
            font-family: inherit;
            font-size: 14px;
        }
        
        textarea {
            resize: vertical;
            min-height: 80px;
        }
        
        .chat-area {
            background: #f9f9f9;
            border-radius: 8px;
            padding: 15px;
            min-height: 300px;
            max-height: 400px;
            overflow-y: auto;
            margin-bottom: 15px;
        }
        
        .message {
            margin-bottom: 12px;
            padding: 10px;
            border-radius: 6px;
        }
        
        .user-msg {
            background: #e3f2fd;
            text-align: right;
            color: #1565c0;
        }
        
        .ai-msg {
            background: #f3e5f5;
            color: #6a1b9a;
        }
        
        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: background 0.3s;
        }
        
        .btn:hover {
            background: #2980b9;
        }
        
        .btn-group {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 10px;
        }
        
        .status {
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .status.success {
            background: #d4edda;
            color: #155724;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
        }
        
        .status.info {
            background: #d1ecf1;
            color: #0c5460;
        }
        
        audio {
            width: 100%;
            margin: 10px 0;
        }
        
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Talleb 5edma - Interview Coach</h1>
        <p class="subtitle">Practice interviews with AI - Voice & Text Supported</p>
        
        <div class="mode-selector">
            <button class="mode-btn active" onclick="setMode('coach')">👨‍🏫 Coach Mode</button>
            <button class="mode-btn" onclick="setMode('employer_interview')">💼 Mock Interview</button>
        </div>
        
        <div class="section">
            <label>Your Name</label>
            <input type="text" id="userName" placeholder="Enter your name">
        </div>
        
        <div class="section">
            <label>Experience Level</label>
            <input type="text" id="experience" placeholder="e.g., 5 years in web development">
        </div>
        
        <div class="section">
            <label>Target Position</label>
            <input type="text" id="position" placeholder="e.g., Senior Developer">
        </div>
        
        <div class="section">
            <label>Company (Optional)</label>
            <input type="text" id="company" placeholder="Target company">
        </div>
        
        <div id="statusMsg"></div>
        
        <div class="chat-area" id="chatArea"></div>
        
        <div id="audioResponse" class="hidden">
            <label>AI Response (Audio):</label>
            <audio controls id="audioPlayer"></audio>
        </div>
        
        <div class="btn-group">
            <button class="btn" onclick="startVoiceInput()">🎙️ Start Recording</button>
            <button class="btn" onclick="stopVoiceInput()">⏹️ Stop Recording</button>
            <button class="btn" onclick="clearChat()">🗑️ Clear Chat</button>
        </div>
        
        <div class="section" style="margin-top: 20px;">
            <label>Or Type Your Message</label>
            <textarea id="userMessage" placeholder="Type your response here..."></textarea>
            <button class="btn" style="width: 100%; margin-top: 10px;" onclick="sendTextMessage()">Send Message</button>
        </div>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let currentMode = 'coach';
        let sessionId = 'session_' + Date.now();

        function setMode(mode) {
            currentMode = mode;
            document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            clearChat();
        }

        async function startVoiceInput() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    await sendVoiceMessage(audioBlob);
                };
                
                mediaRecorder.start();
                showStatus('🎙️ Recording... Click Stop to send', 'info');
            } catch (error) {
                showStatus('❌ Microphone access denied', 'error');
            }
        }

        function stopVoiceInput() {
            if (mediaRecorder) {
                mediaRecorder.stop();
                showStatus('⏳ Processing audio...', 'info');
            }
        }

        async function sendVoiceMessage(audioBlob) {
            try {
                const reader = new FileReader();
                reader.onload = async (event) => {
                    const base64Audio = event.target.result.split(',')[1];
                    
                    const response = await fetch('/api/voice-chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            audio: base64Audio,
                            session_id: sessionId,
                            mode: currentMode,
                            user_details: {
                                name: document.getElementById('userName').value || 'User',
                                experience_level: document.getElementById('experience').value || 'Not specified',
                                skills: []
                            },
                            offer_details: {
                                position: document.getElementById('position').value || 'Not specified',
                                company: document.getElementById('company').value || 'Not specified',
                                required_skills: []
                            }
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        addMessage('You', data.user_text);
                        addMessage('AI Coach', data.ai_response);
                        
                        if (data.audio) {
                            const audioData = 'data:audio/mp3;base64,' + data.audio;
                            document.getElementById('audioPlayer').src = audioData;
                            document.getElementById('audioResponse').classList.remove('hidden');
                        }
                        
                        showStatus('✅ Received response!', 'success');
                    } else {
                        showStatus('❌ ' + data.error, 'error');
                    }
                };
                reader.readAsDataURL(audioBlob);
            } catch (error) {
                showStatus('❌ Error: ' + error.message, 'error');
            }
        }

        async function sendTextMessage() {
            const message = document.getElementById('userMessage').value.trim();
            if (!message) {
                showStatus('⚠️ Please type a message', 'error');
                return;
            }

            try {
                const response = await fetch('/api/text-chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: message,
                        session_id: sessionId,
                        mode: currentMode,
                        user_details: {
                            name: document.getElementById('userName').value || 'User',
                            experience_level: document.getElementById('experience').value || 'Not specified',
                            skills: []
                        },
                        offer_details: {
                            position: document.getElementById('position').value || 'Not specified',
                            company: document.getElementById('company').value || 'Not specified',
                            required_skills: []
                        }
                    })
                });

                const data = await response.json();
                
                if (data.success) {
                    addMessage('You', message);
                    addMessage('AI Coach', data.ai_response);
                    document.getElementById('userMessage').value = '';
                    showStatus('✅ Response received!', 'success');
                } else {
                    showStatus('❌ ' + data.error, 'error');
                }
            } catch (error) {
                showStatus('❌ Error: ' + error.message, 'error');
            }
        }

        function addMessage(sender, text) {
            const chatArea = document.getElementById('chatArea');
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message ' + (sender === 'You' ? 'user-msg' : 'ai-msg');
            msgDiv.innerHTML = '<strong>' + sender + ':</strong> ' + text;
            chatArea.appendChild(msgDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        function showStatus(message, type) {
            const statusDiv = document.getElementById('statusMsg');
            statusDiv.className = 'status ' + type;
            statusDiv.textContent = message;
        }

        function clearChat() {
            document.getElementById('chatArea').innerHTML = '';
            document.getElementById('statusMsg').innerHTML = '';
            document.getElementById('audioResponse').classList.add('hidden');
            fetch('/api/clear-history', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId })
            });
        }

        // Initialize
        showStatus('✅ Ready! Choose a mode and start practicing', 'info');
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/voice-chat", methods=["POST"])
def voice_chat():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No JSON data received"}), 400

        audio_base64 = data.get("audio")
        session_id = data.get("session_id", "default_session")
        user_details = data.get("user_details", {}) or {}
        offer_details = data.get("offer_details", {}) or {}
        mode = data.get("mode", "coach")
        language_hint = data.get("language_hint", "auto")

        if not audio_base64:
            return jsonify({"success": False, "error": "No audio data provided"}), 400

        print(f"📥 Received audio: {len(audio_base64)} chars base64")

        try:
            audio_bytes = base64.b64decode(audio_base64)
            print(f"✅ Decoded audio: {len(audio_bytes)} bytes")
        except Exception as e:
            return jsonify({"success": False, "error": f"Invalid base64 audio: {str(e)}"}), 400

        print("🎵 Preprocessing audio...")
        processed_audio, mime_type = audio_bytes, "audio/webm"

        print("🎤 Step 1: Transcribing with Gemini STT...")
        user_text, detected_language = transcribe_with_gemini(processed_audio, mime_type)

        # FIX #2: Validate STT response
        if not user_text or len(user_text.strip()) < 2:
            return jsonify({
                "success": False,
                "error": "Could not understand speech. Please try speaking more clearly.",
                "user_text": user_text,
                "detected_language": detected_language
            }), 400

        if language_hint != "auto":
            detected_language = language_hint

        print(f"🌐 Detected language: {detected_language}")

        chat_history = get_conversation_history(session_id)
        print(f"🤖 Step 2: Getting AI response in {detected_language}...")

        ai_response = get_ai_response(
            user_text,
            user_details=user_details,
            offer_details=offer_details,
            chat_history=chat_history,
            mode=mode,
            language=detected_language,
        )

        save_conversation(session_id, user_text, ai_response, detected_language)

        print(f"🔊 Step 3: Generating speech in {detected_language}...")
        audio_response = text_to_speech(ai_response, detected_language)

        # FIX #3: Validate TTS response
        if not audio_response or len(audio_response) < 100:
            return jsonify({
                "success": False,
                "error": "Failed to generate audio response",
                "user_text": user_text,
                "ai_response": ai_response,
                "detected_language": detected_language
            }), 500

        audio_base64_response = base64.b64encode(audio_response).decode("utf-8")

        return jsonify({
            "success": True,
            "user_text": user_text,
            "ai_response": ai_response,
            "audio": audio_base64_response,
            "detected_language": detected_language,
            "session_id": session_id
        })
    except Exception as e:
        print(f"❌ Voice chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/text-chat", methods=["POST"])
def text_chat():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No JSON data received"}), 400

        text = data.get("text")
        session_id = data.get("session_id", "default_session")
        user_details = data.get("user_details", {}) or {}
        offer_details = data.get("offer_details", {}) or {}
        mode = data.get("mode", "coach")
        language_hint = data.get("language_hint", "auto")

        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400

        print(f"💬 Text chat: {text}")

        detected_language = detect_language_from_text(text)
        if language_hint != "auto":
            detected_language = language_hint

        chat_history = get_conversation_history(session_id)

        ai_response = get_ai_response(
            text,
            user_details=user_details,
            offer_details=offer_details,
            chat_history=chat_history,
            mode=mode,
            language=detected_language,
        )

        save_conversation(session_id, text, ai_response, detected_language)

        return jsonify({
            "success": True,
            "ai_response": ai_response,
            "detected_language": detected_language,
            "session_id": session_id
        })
    except Exception as e:
        print(f"❌ Text chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/clear-history", methods=["POST"])
def clear_history():
    try:
        data = request.json or {}
        session_id = data.get("session_id", "default_session")

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
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "active",
        "keys_configured": len(gemini_clients),
        "supported_languages": list(LANGUAGE_MAPPING.keys()),
        "timestamp": datetime.now().isoformat()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)