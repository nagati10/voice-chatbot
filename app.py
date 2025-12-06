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
import numpy as np

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

# Language code mapping
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
    'zh-cn': 'zh-CN', # Chinese Simplified
    'zh-tw': 'zh-TW', # Chinese Traditional
    'ar': 'ar',      # Arabic
    'hi': 'hi',      # Hindi
    'bn': 'bn',      # Bengali
    'tr': 'tr',      # Turkish
    'nl': 'nl',      # Dutch
    'pl': 'pl',      # Polish
    'vi': 'vi',      # Vietnamese
    'th': 'th',      # Thai
    'id': 'id',      # Indonesian
    'el': 'el',      # Greek
    'he': 'he',      # Hebrew
}

# Supported languages for speech recognition
SPEECH_LANGUAGES = {
    'en': 'en-US',    # English (US)
    'en-uk': 'en-GB', # English (UK)
    'es': 'es-ES',    # Spanish
    'fr': 'fr-FR',    # French
    'de': 'de-DE',    # German
    'it': 'it-IT',    # Italian
    'pt': 'pt-PT',    # Portuguese
    'ru': 'ru-RU',    # Russian
    'ja': 'ja-JP',    # Japanese
    'ko': 'ko-KR',    # Korean
    'zh-cn': 'zh-CN', # Chinese (Simplified)
    'zh-tw': 'zh-TW', # Chinese (Traditional)
    'ar': 'ar-SA',    # Arabic
    'hi': 'hi-IN',    # Hindi
    'bn': 'bn-BD',    # Bengali
    'tr': 'tr-TR',    # Turkish
    'nl': 'nl-NL',    # Dutch
    'pl': 'pl-PL',    # Polish
    'vi': 'vi-VN',    # Vietnamese
    'th': 'th-TH',    # Thai
    'id': 'id-ID',    # Indonesian
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
                  user_input TEXT, ai_response TEXT,
                  language TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ========== HELPER FUNCTIONS ==========
def filter_false_positives(text):
    """Filter out common false positives and nonsensical text"""
    if not text:
        return text
    
    text_lower = text.lower().strip()
    
    # Common false positives to filter
    false_positives = [
        "ok google", "hey google", "ok google ok google", "ok google ok google ok google",
        "alexa", "siri", "cortana", "hey siri",
        "test test", "testing testing", "testing 1 2 3",
        "hello hello", "hi hi", "hey hey",
        "microphone test", "mic test"
    ]
    
    for fp in false_positives:
        if fp in text_lower:
            print(f"⚠️ Filtered false positive: {fp}")
            return "I didn't catch that. Could you please speak clearly and try again?"
    
    # Check for repeated single words
    words = text_lower.split()
    if len(words) <= 3:
        # Check if it's just repeated words
        if len(set(words)) == 1 and len(words) > 1:
            print(f"⚠️ Filtered repeated word: {words[0]}")
            return "I heard you say something, but it wasn't clear. Please speak a complete sentence."
    
    # Check for extremely short responses
    if len(text.strip()) < 3:
        return "That was too short. Please speak a complete sentence."
    
    # Check for nonsense combinations
    nonsense_patterns = ["asdf", "qwerty", "zxcv", "123", "abc"]
    for pattern in nonsense_patterns:
        if pattern in text_lower:
            print(f"⚠️ Filtered nonsense pattern: {pattern}")
            return "I didn't understand that. Could you please rephrase?"
    
    return text

# ========== LANGUAGE DETECTION ==========
def detect_language_text(text):
    """Detect language from text with fallback"""
    try:
        if not text or len(text.strip()) < 3:
            return 'en'  # Default to English for short text
        
        # Filter text first
        filtered_text = filter_false_positives(text)
        if filtered_text != text:
            return 'en'
        
        # Detect language
        detected = detect(text)
        
        # Get confidence scores
        lang_probabilities = detect_langs(text)
        
        # Check if we have reasonable confidence (>0.5)
        for lang_prob in lang_probabilities:
            if lang_prob.prob > 0.5:
                detected_lang = lang_prob.lang
                print(f"🌐 Detected language: {detected_lang} (confidence: {lang_prob.prob:.2f})")
                
                # Map to our supported language codes
                if detected_lang in LANGUAGE_MAPPING:
                    return LANGUAGE_MAPPING[detected_lang]
                elif detected_lang.split('-')[0] in LANGUAGE_MAPPING:
                    return LANGUAGE_MAPPING[detected_lang.split('-')[0]]
        
        return 'en'  # Default fallback
    except Exception as e:
        print(f"⚠️ Language detection error: {e}")
        return 'en'  # Default to English on error

def detect_language_audio(text, session_id):
    """Detect language from text or get from session history"""
    try:
        # Try to get language from session history
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("SELECT language FROM conversations WHERE session_id=? ORDER BY timestamp DESC LIMIT 1", 
                  (session_id,))
        result = c.fetchone()
        conn.close()
        
        if result and result[0]:
            print(f"🌐 Using language from history: {result[0]}")
            return result[0]
        
        # Otherwise detect from text
        return detect_language_text(text)
        
    except Exception as e:
        print(f"⚠️ Language detection from history error: {e}")
        return detect_language_text(text)

# ========== AUDIO PREPROCESSING ==========
def preprocess_audio(audio_bytes, target_format='wav'):
    """Preprocess audio data to improve recognition rate and reduce noise"""
    try:
        print(f"🎵 Preprocessing audio: {len(audio_bytes)} bytes")
        
        # Check if audio is too short
        if len(audio_bytes) < 1000:  # Less than ~60ms at 16kHz
            print("⚠️ Audio too short, likely noise")
            return audio_bytes
        
        # Method 1: Try using pydub conversion with noise reduction
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            # Load and convert with pydub
            try:
                audio = AudioSegment.from_file(tmp_path, format="webm")
            except:
                # Try wav format
                audio = AudioSegment.from_file(tmp_path, format="wav")
            
            # Convert to mono
            audio = audio.set_channels(1)
            
            # Check audio volume
            dbfs = audio.dBFS
            print(f"📊 Original audio: {len(audio)}ms, {dbfs:.1f} dBFS")
            
            if dbfs < -40:  # Very quiet audio
                print(f"⚠️ Audio is very quiet: {dbfs:.1f} dBFS")
                # Try to boost volume
                audio = audio + 20  # Boost by 20dB
            
            # Normalize volume to -20 dBFS (good for speech)
            audio = audio.normalize(headroom=10)
            
            # Apply noise gate (remove very quiet parts)
            try:
                from pydub.silence import detect_nonsilent
                
                # Detect non-silent parts
                min_silence_len = 100  # ms
                silence_thresh = -35   # dBFS
                
                nonsilent_ranges = detect_nonsilent(
                    audio, 
                    min_silence_len=min_silence_len,
                    silence_thresh=silence_thresh,
                    seek_step=10
                )
                
                if len(nonsilent_ranges) > 0:
                    # Get start and end of speech
                    start = min([start for start, end in nonsilent_ranges])
                    end = max([end for start, end in nonsilent_ranges])
                    
                    # Add padding
                    padding = 100  # ms
                    start = max(0, start - padding)
                    end = min(len(audio), end + padding)
                    
                    # Extract speech portion
                    if end - start > 300:  # At least 300ms of audio
                        audio = audio[start:end]
                        print(f"✅ Noise gated: kept {len(audio)}ms of speech")
                    else:
                        print("⚠️ Speech too short after noise gating")
                else:
                    print("⚠️ No speech detected")
                    
            except Exception as e:
                print(f"⚠️ Noise gating failed: {e}")
                # Continue without noise gating
            
            # Convert to 16kHz sample rate
            audio = audio.set_frame_rate(16000)
            
            # Apply simple high-pass filter to remove low-frequency noise
            try:
                # Simple DC bias removal
                samples = np.frombuffer(audio.raw_data, dtype=np.int16)
                
                # Remove DC offset
                samples = samples - np.mean(samples)
                
                # Simple high-pass filter (remove frequencies below 80Hz)
                alpha = 0.95
                filtered = np.zeros_like(samples)
                filtered[0] = samples[0]
                for i in range(1, len(samples)):
                    filtered[i] = alpha * filtered[i-1] + alpha * (samples[i] - samples[i-1])
                
                # Convert back to bytes
                import array
                samples_array = array.array('h', filtered.astype(np.int16))
                raw_data = samples_array.tobytes()
                
                audio = AudioSegment(
                    data=raw_data,
                    sample_width=audio.sample_width,
                    frame_rate=audio.frame_rate,
                    channels=audio.channels
                )
                
            except Exception as e:
                print(f"⚠️ Audio filtering failed: {e}")
            
            # Export as wav with optimal settings
            output = io.BytesIO()
            audio.export(output, format="wav", parameters=[
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1"
            ])
            processed_bytes = output.getvalue()
            
            print(f"✅ Audio preprocessed: {len(processed_bytes)} bytes, duration: {len(audio)}ms")
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return processed_bytes
            
        except Exception as e:
            print(f"⚠️ Pydub preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            # If pydub fails, return original audio
            return audio_bytes
            
    except Exception as e:
        print(f"❌ Audio preprocessing error: {e}")
        return audio_bytes

# ========== CONVERSATION MEMORY ==========
def save_conversation(session_id, user_input, ai_response, language):
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
def transcribe_audio(audio_bytes):
    """Speech-to-text with improved filtering for false positives"""
    if not SR_AVAILABLE:
        return "Speech recognition is temporarily unavailable. Please use text mode."
    
    try:
        r = sr.Recognizer()
        
        print(f"🔊 Audio bytes length: {len(audio_bytes)}")
        
        # Check if audio is too short (likely noise)
        if len(audio_bytes) < 2000:  # Less than ~125ms at 16kHz
            print("⚠️ Audio too short, likely noise")
            return "I didn't hear anything. Please speak clearly."
        
        # Try with auto-detection first
        languages_to_try = [None]  # None = Google's auto-detection
        
        # Add common languages if auto-detection fails
        common_languages = ['en-US', 'es-ES', 'fr-FR', 'de-DE', 'ja-JP', 'ko-KR', 'zh-CN']
        languages_to_try.extend(common_languages)
        
        transcripts = []  # Store all successful transcripts
        
        for language_code in languages_to_try:
            try:
                print(f"🌐 Trying language: {language_code if language_code else 'auto'}")
                
                # Create AudioData
                audio_data = sr.AudioData(audio_bytes, 
                                         sample_rate=16000, 
                                         sample_width=2)
                
                # Configure recognizer for better accuracy
                r.energy_threshold = 300  # Increased to filter noise
                r.dynamic_energy_threshold = True
                r.pause_threshold = 0.8
                r.phrase_threshold = 0.1
                r.non_speaking_duration = 0.3
                
                # Try recognition with show_all to get alternatives
                results = r.recognize_google(audio_data, 
                                           language=language_code,
                                           show_all=True)
                
                if results and 'alternative' in results:
                    for alt in results['alternative']:
                        if 'transcript' in alt and alt['transcript'].strip():
                            text = alt['transcript'].strip()
                            confidence = alt.get('confidence', 0.5)
                            
                            # Filter false positives
                            filtered_text = filter_false_positives(text)
                            if filtered_text != text:
                                print(f"⚠️ Filtered suspicious text: {text}")
                                continue
                            
                            # Check for quality
                            words = text.lower().split()
                            word_count = len(words)
                            
                            # Skip very short or nonsense
                            if word_count == 0:
                                continue
                            
                            # Check for actual speech (not just filler)
                            filler_words = ['uh', 'um', 'ah', 'er', 'hmm', 'oh']
                            actual_words = [w for w in words if w not in filler_words and len(w) > 2]
                            
                            if len(actual_words) == 0:
                                print(f"⚠️ Only filler words: {text}")
                                continue
                            
                            # Store transcript with confidence
                            transcripts.append({
                                'text': text,
                                'confidence': confidence,
                                'language': language_code if language_code else 'auto',
                                'word_count': word_count
                            })
                            
            except sr.UnknownValueError:
                continue  # Try next language
            except sr.RequestError as e:
                print(f"❌ Speech recognition service error: {e}")
                if not transcripts:  # Only return error if no transcripts yet
                    return "Speech service is temporarily unavailable. Please try again."
                else:
                    break  # We have some transcripts, use them
            except Exception as e:
                print(f"⚠️ Try {language_code} failed: {e}")
                continue
        
        # Select best transcript
        if transcripts:
            # Sort by confidence and word count
            transcripts.sort(key=lambda x: (x['confidence'], x['word_count']), reverse=True)
            best = transcripts[0]
            
            print(f"✅ Best transcript ({best['language']}, confidence: {best['confidence']:.2f}): {best['text'][:100]}...")
            return best['text']
        
        # If we get here, all attempts failed
        print("❌ All transcription attempts failed")
        
        # Save debug audio
        try:
            import wave
            debug_filename = f"debug_audio_{datetime.now().strftime('%H%M%S')}.wav"
            with wave.open(debug_filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_bytes[:8000])  # First 0.5s
            print(f"💾 Saved debug audio to {debug_filename}")
        except:
            pass
            
        return "I couldn't understand the audio. Please speak clearly and try again."
        
    except sr.UnknownValueError:
        print("⚠️ Speech recognition could not understand audio")
        return "I couldn't understand the audio. Please speak clearly and try again."
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
    """Get response from Gemini 2.5 Flash in the detected language"""
    
    if not GEMINI_API_KEY:
        return "AI service is not configured. Please set GEMINI_API_KEY environment variable.", 'en'
    
    # Filter text first
    filtered_text = filter_false_positives(text)
    if filtered_text != text:
        return filtered_text, language
    
    # Get conversation history
    history = get_conversation_history(session_id)
    
    # Build context from history
    context = ""
    if history:
        for user_msg, ai_msg, hist_lang in reversed(history):
            context += f"User: {user_msg}\nAssistant: {ai_msg}\n"
        context += "\n"
    
    # Language-specific instructions
    language_instructions = {
        'en': "Respond conversationally in 1-3 sentences as a helpful voice assistant.",
        'es': "Responde de manera conversacional en 1-3 frases como un asistente de voz útil.",
        'fr': "Répondez de manière conversationnelle en 1-3 phrases comme un assistant vocal utile.",
        'de': "Antworten Sie konversationell in 1-3 Sätzen als hilfreicher Sprachassistent.",
        'ja': "役立つ音声アシスタントとして、1〜3文で会話形式で答えてください。",
        'ko': "유용한 음성 비서로서 1~3문장으로 대화식으로 응답하세요.",
        'zh-CN': "以有用的语音助手身份，用1-3句话进行对话式回答。",
        'zh-TW': "以有用的語音助手身份，用1-3句話進行對話式回答。",
        'hi': "एक मददगार आवाज सहायक के रूप में 1-3 वाक्यों में बातचीत के तरीके से जवाब दें।",
        'ar': "رد بطريقة محادثة في 1-3 جملة كمساعد صوتي مفيد.",
        'ru': "Отвечайте разговорным тоном в 1-3 предложениях в качестве полезного голосового помощника.",
        'it': "Rispondi conversazionalmente in 1-3 frasi come un assistente vocale utile.",
        'pt': "Responda conversacionalmente em 1-3 frases como um assistente de voz útil.",
    }
    
    instruction = language_instructions.get(language, language_instructions['en'])
    
    # Combine with current input
    full_prompt = f"""{context}User: {text}

{instruction}

IMPORTANT: Respond in {language} language only. Keep response natural and conversational."""
    
    # Call Gemini API
    try:
        # Initialize client
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Generate content with Gemini 2.5 Flash
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "max_output_tokens": 150
            }
        )
        
        # Extract text from response
        if response and response.text:
            ai_text = response.text.strip()
            print(f"✅ Gemini 2.5 response ({language}): {ai_text[:100]}...")
        else:
            print("⚠️ Empty response from Gemini")
            ai_text = "I heard you, but I'm having trouble responding right now. Could you rephrase that?"
        
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        
        # Fallback response in detected language
        fallback_responses = {
            'en': "I'm having trouble connecting to my AI service right now. Please try again in a moment.",
            'es': "Estoy teniendo problemas para conectarme a mi servicio de IA en este momento. Por favor, inténtalo de nuevo en un momento.",
            'fr': "J'ai du mal à me connecter à mon service d'IA pour le moment. Veuillez réessayer dans un instant.",
            'de': "Ich habe gerade Probleme, mich mit meinem KI-Dienst zu verbinden. Bitte versuchen Sie es in einem Moment erneut.",
            'ja': "現在、AIサービスに接続するのに問題が発生しています。しばらくしてからもう一度お試しください。",
            'ko': "현재 AI 서비스에 연결하는 데 문제가 있습니다. 잠시 후 다시 시도해 주세요.",
            'zh-CN': "我目前连接AI服务时遇到问题。请稍后再试。",
            'zh-TW': "我目前連接AI服務時遇到問題。請稍後再試。",
        }
        
        ai_text = fallback_responses.get(language, fallback_responses['en'])
    
    # Save to memory with language
    save_conversation(session_id, text, ai_text, language)
    
    return ai_text, language

# ========== TEXT-TO-SPEECH ==========
def text_to_speech(text, language='en'):
    """Convert text to speech in the specified language"""
    try:
        # Map language code to gTTS compatible codes
        tts_language_map = {
            'en': 'en',
            'es': 'es',
            'fr': 'fr',
            'de': 'de',
            'it': 'it',
            'pt': 'pt',
            'ru': 'ru',
            'ja': 'ja',
            'ko': 'ko',
            'zh-CN': 'zh-CN',
            'zh-TW': 'zh-TW',
            'ar': 'ar',
            'hi': 'hi',
            'bn': 'bn',
            'tr': 'tr',
            'nl': 'nl',
            'pl': 'pl',
            'vi': 'vi',
            'th': 'th',
            'id': 'id',
            'el': 'el',
            'he': 'he',
        }
        
        tts_lang = tts_language_map.get(language, 'en')
        
        # Handle Chinese variants
        if language in ['zh-cn', 'zh-CN']:
            tts_lang = 'zh-CN'
        elif language in ['zh-tw', 'zh-TW']:
            tts_lang = 'zh-TW'
        
        print(f"🔊 Generating TTS in {tts_lang} language")
        tts = gTTS(text=text, lang=tts_lang, slow=False, lang_check=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        audio_bytes = audio_buffer.read()
        print(f"✅ Generated TTS audio ({tts_lang}): {len(audio_bytes)} bytes")
        return audio_bytes
        
    except Exception as e:
        print(f"❌ TTS error: {e}")
        
        # Fallback to English if language not supported
        if language != 'en':
            try:
                print("🔄 Falling back to English TTS")
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
    """Main endpoint - process voice to voice with language detection"""
    try:
        data = request.json
        
        if not data:
            return jsonify({"success": False, "error": "No JSON data received"}), 400
        
        audio_base64 = data.get('audio')
        session_id = data.get('session_id', 'default_session')
        language_hint = data.get('language_hint')  # Optional hint
        
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
        
        # Step 1: Speech to Text
        print("🎤 Step 1: Transcribing audio...")
        user_text = transcribe_audio(audio_bytes)
        
        # Check transcription result
        error_phrases = ["Speech recognition", "Error processing", "I couldn't understand", 
                        "service is temporarily", "I didn't hear", "Please speak", 
                        "That was too short"]
        
        if any(phrase in user_text for phrase in error_phrases):
            return jsonify({
                "success": False, 
                "error": user_text, 
                "user_text": user_text,
                "debug": "Transcription failed or filtered"
            }), 400
        
        # Step 2: Detect language from text
        print("🌐 Step 2: Detecting language...")
        detected_language = detect_language_text(user_text)
        print(f"🌐 Detected language: {detected_language}")
        
        # Step 3: Get AI Response in detected language
        print(f"🤖 Step 3: Getting AI response in {detected_language}...")
        ai_response, response_language = get_ai_response(user_text, session_id, detected_language)
        
        # Step 4: Text to Speech in same language
        print(f"🔊 Step 4: Generating speech in {response_language}...")
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
        print(f"✅ Response audio ({response_language}): {len(audio_base64_response)} chars base64")
        
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
    """Text-only endpoint with language detection"""
    try:
        data = request.json
        
        if not data:
            return jsonify({"success": False, "error": "No JSON data received"}), 400
        
        text = data.get('text')
        session_id = data.get('session_id', 'default_session')
        language_hint = data.get('language_hint')
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        print(f"💬 Text chat: {text}")
        
        # Filter text
        filtered_text = filter_false_positives(text)
        if filtered_text != text:
            return jsonify({
                "success": False,
                "error": filtered_text,
                "user_text": text
            }), 400
        
        # Detect language
        detected_language = detect_language_text(text)
        if language_hint and language_hint in LANGUAGE_MAPPING:
            detected_language = LANGUAGE_MAPPING[language_hint]
        
        print(f"🌐 Detected language: {detected_language}")
        
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

@app.route('/api/detect-language', methods=['POST'])
def detect_language_endpoint():
    """API endpoint to detect language from text"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        # Filter text first
        filtered_text = filter_false_positives(text)
        if filtered_text != text:
            return jsonify({
                "success": False,
                "error": "Text contains suspicious patterns",
                "filtered_text": filtered_text
            }), 400
        
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

@app.route('/api/debug-audio', methods=['POST'])
def debug_audio():
    """Debug endpoint to analyze audio issues"""
    try:
        data = request.json
        audio_base64 = data.get('audio')
        
        if not audio_base64:
            return jsonify({"success": False, "error": "No audio provided"}), 400
        
        audio_bytes = base64.b64decode(audio_base64)
        
        # Save for analysis
        import wave
        filename = f"debug_audio_{datetime.now().strftime('%H%M%S')}.wav"
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_bytes)
        
        # Analyze audio
        audio = AudioSegment.from_wav(filename)
        
        # Try transcription for debugging
        transcription = ""
        if SR_AVAILABLE:
            try:
                r = sr.Recognizer()
                audio_data = sr.AudioData(audio_bytes, sample_rate=16000, sample_width=2)
                transcription = r.recognize_google(audio_data, language='en-US', show_all=False)
            except:
                transcription = "Transcription failed"
        
        return jsonify({
            "success": True,
            "audio_length_ms": len(audio),
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "dbFS": audio.dBFS,
            "max_dBFS": audio.max_dBFS,
            "rms": audio.rms,
            "file_saved": filename,
            "transcription_attempt": transcription
        })
        
    except Exception as e:
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
        <title>🌍 Multilingual Voice AI Chatbot</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                padding: 40px;
                max-width: 1200px;
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
            .language-test {{
                margin-top: 30px;
                padding: 20px;
                background: #f0f4ff;
                border-radius: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌍 Multilingual Voice AI Chatbot</h1>
            <p style="color: #666;">Real-time Speech-to-Speech in 20+ Languages</p>
            
            <div class="status">
                <p><strong>Status:</strong> ✅ Active</p>
                <p><strong>Cost:</strong> $0/month</p>
                <p><strong>Multilingual:</strong> ✅ 20+ Languages Supported</p>
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
                <div class="language-tag">Russian (ru)</div>
                <div class="language-tag">Japanese (ja)</div>
                <div class="language-tag">Korean (ko)</div>
                <div class="language-tag">Chinese (zh)</div>
                <div class="language-tag">Arabic (ar)</div>
                <div class="language-tag">Hindi (hi)</div>
                <div class="language-tag">Turkish (tr)</div>
                <div class="language-tag">Dutch (nl)</div>
                <div class="language-tag">Polish (pl)</div>
                <div class="language-tag">Thai (th)</div>
                <div class="language-tag">Vietnamese (vi)</div>
                <div class="language-tag">Indonesian (id)</div>
            </div>
            
            <h2>API Endpoints</h2>
            <div class="endpoint"><strong>POST</strong> /api/voice-chat - Voice chat with auto language detection</div>
            <div class="endpoint"><strong>POST</strong> /api/text-chat - Text chat with language detection</div>
            <div class="endpoint"><strong>POST</strong> /api/detect-language - Detect language from text</div>
            <div class="endpoint"><strong>GET</strong> /api/supported-languages - List supported languages</div>
            <div class="endpoint"><strong>POST</strong> /api/debug-audio - Debug audio issues</div>
            <div class="endpoint"><strong>POST</strong> /api/clear-history - Clear conversation history</div>
            <div class="endpoint"><strong>GET</strong> /health - Health check</div>
            
            <div class="language-test">
                <h3>Test Multilingual API</h3>
                <button onclick="testTextChat('en')">Test English</button>
                <button onclick="testTextChat('es')">Test Spanish</button>
                <button onclick="testTextChat('fr')">Test French</button>
                <button onclick="testTextChat('de')">Test German</button>
                <button onclick="testTextChat('ja')">Test Japanese</button>
                <button onclick="testTextChat('hi')">Test Hindi</button>
                
                <div style="margin-top: 20px;">
                    <input type="text" id="customText" placeholder="Type text in any language..." style="width: 300px; padding: 10px;">
                    <button onclick="testCustomText()">Test Custom Text</button>
                </div>
                
                <div id="testResult"></div>
            </div>
            
            <script>
                async function testTextChat(lang) {{
                    const testTexts = {{
                        'en': 'Hello! How are you today?',
                        'es': '¡Hola! ¿Cómo estás hoy?',
                        'fr': 'Bonjour! Comment allez-vous aujourd\\\\'hui?',
                        'de': 'Hallo! Wie geht es dir heute?',
                        'ja': 'こんにちは！今日は元気ですか？',
                        'hi': 'नमस्ते! आप आज कैसे हैं?'
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
                                'Detected: ' + data.detected_language + '<br>' +
                                'Response: "' + data.ai_response + '"' +
                            '</div>';
                        }} else {{
                            result.innerHTML = '<div class="error"><strong>❌ Error:</strong> ' + data.error + '</div>';
                        }}
                    }} catch (error) {{
                        result.innerHTML = '<div class="error"><strong>❌ Network Error:</strong> ' + error.message + '</div>';
                    }}
                }}
                
                async function testCustomText() {{
                    const customText = document.getElementById('customText').value;
                    if (!customText) return;
                    
                    const result = document.getElementById('testResult');
                    result.innerHTML = '<div style="color: orange;">Detecting language...</div>';
                    
                    try {{
                        // First detect language
                        const detectResponse = await fetch('/api/detect-language', {{
                            method: 'POST',
                            headers: {{'Content-Type': 'application/json'}},
                            body: JSON.stringify({{text: customText}})
                        }});
                        
                        const detectData = await detectResponse.json();
                        
                        if (detectData.success) {{
                            result.innerHTML = '<div style="color: blue;">' +
                                '<strong>🌐 Detected Language:</strong> ' + detectData.detected_language + '<br>' +
                                'Now getting AI response...' +
                            '</div>';
                            
                            // Then get AI response
                            const chatResponse = await fetch('/api/text-chat', {{
                                method: 'POST',
                                headers: {{'Content-Type': 'application/json'}},
                                body: JSON.stringify({{
                                    text: customText, 
                                    session_id: 'test_' + Date.now(),
                                    language_hint: detectData.detected_language
                                }})
                            }});
                            
                            const chatData = await chatResponse.json();
                            
                            if (chatData.success) {{
                                result.innerHTML = '<div class="success">' +
                                    '<strong>✅ ' + detectData.detected_language.toUpperCase() + ' SUCCESS!</strong><br>' +
                                    'Detected: ' + chatData.detected_language + '<br>' +
                                    'Response: "' + chatData.ai_response + '"' +
                                '</div>';
                            }} else {{
                                result.innerHTML = '<div class="error"><strong>❌ Chat Error:</strong> ' + chatData.error + '</div>';
                            }}
                        }} else {{
                            result.innerHTML = '<div class="error"><strong>❌ Detection Error:</strong> ' + detectData.error + '</div>';
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
                                'Multilingual: ' + (data.multilingual ? '✅' : '❌') + '<br>' +
                                'Supported Languages: ' + data.supported_languages.length + '<br>' +
                                'Speech Recognition: ' + (data.speech_recognition ? '✅' : '❌') + '<br>' +
                                'Gemini: ' + (data.gemini_configured ? '✅' : '❌') +
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