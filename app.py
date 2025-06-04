import gradio as gr
import pdfplumber
from huggingface_hub import InferenceClient
import os
import logging
import traceback
import time
import random
import requests
import base64
from io import BytesIO

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize HF Inference Client
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    logging.warning("HF_TOKEN environment variable not set. Some features may be limited.")
    client = None
else:
    client = InferenceClient(token=hf_token)

# Local fallback questions
TECH_QUESTIONS = [
    "What programming languages are you most comfortable with?",
    "Describe a challenging technical problem you solved recently.",
    "How do you approach debugging complex systems?",
    "What cloud platforms have you worked with?",
    "Tell me about a time you improved system performance.",
    "How do you stay updated with new technologies?",
    "Describe your experience with version control systems.",
    "What's your preferred development methodology?",
    "How do you ensure code quality in your projects?",
    "Describe your experience with containerization."
]

HR_QUESTIONS = [
    "Tell me about yourself.",
    "What are your greatest strengths and weaknesses?",
    "Where do you see yourself in 5 years?",
    "Describe a time you handled workplace conflict.",
    "Why do you want to work for our company?",
    "What motivates you professionally?",
    "How do you handle tight deadlines?",
    "Describe your ideal work environment.",
    "What's your approach to teamwork?",
    "How do you handle constructive criticism?"
]

def parse_resume(file_path):
    """Extract text from PDF resume using pdfplumber"""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # Explicitly set crop area to avoid warnings
                page = page.crop(page.bbox)
                text += page.extract_text() or ""  # Handle pages with no text
        logging.info(f"Parsed resume: {text[:100]}...")
        return text
    except Exception as e:
        logging.error(f"PDF parsing error: {traceback.format_exc()}")
        raise gr.Error(f"❌ Error parsing PDF: {str(e)}")

def generate_question(conversation, resume_text, round_type):
    """Generate interview questions using Mistral-7B with fallback"""
    # Truncate resume text while preserving words
    truncated_resume = resume_text[:2000]
    if len(resume_text) > 2000:
        truncated_resume = truncated_resume[:truncated_resume.rfind(' ')] + " [...]"
    
    # Build prompt
    prompt = f"""<s>[INST] You are a professional {round_type} interviewer. Based on this resume:
{truncated_resume}

Recent conversation:
{conversation}

Ask one relevant, insightful question. [/INST]"""
    
    try:
        if client:
            response = client.text_generation(
                prompt,
                model="mistralai/Mistral-7B-Instruct-v0.1",
                max_new_tokens=80,
                temperature=0.7
            )
            return response.strip()
    except Exception as e:
        logging.error(f"Text generation failed: {traceback.format_exc()}")
    
    # Fallback to local questions
    pool = TECH_QUESTIONS if round_type == "Technical" else HR_QUESTIONS
    return random.choice(pool)

def text_to_speech(text):
    """Convert text to speech with multiple fallback options"""
    # Option 1: Hugging Face API
    try:
        if client:
            return client.text_to_speech(text, model="espnet/kan-bayashi_ljspeech_vits")
    except Exception as e:
        logging.error(f"Hugging Face TTS failed: {traceback.format_exc()}")
    
    # Option 2: Google TTS API (free)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        try:
            response = requests.post(
                "https://texttospeech.googleapis.com/v1/text:synthesize",
                params={"key": google_api_key},
                json={
                    "input": {"text": text},
                    "voice": {"languageCode": "en-US", "name": "en-US-Wavenet-D"},
                    "audioConfig": {"audioEncoding": "MP3"}
                },
                timeout=10
            )
            if response.status_code == 200:
                audio_content = base64.b64decode(response.json()['audioContent'])
                return audio_content
        except Exception as e:
            logging.error(f"Google TTS failed: {str(e)}")
    
    # Option 3: Local TTS (requires pyttsx3)
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.save_to_file(text, 'temp.wav')
        engine.runAndWait()
        with open('temp.wav', 'rb') as f:
            return f.read()
    except ImportError:
        logging.warning("pyttsx3 not installed for local TTS")
    except Exception as e:
        logging.error(f"Local TTS failed: {str(e)}")
    
    # Final fallback: Generate silent audio
    return generate_silent_audio(1.5)  # 1.5 seconds of silence

def generate_silent_audio(duration=1.0, sample_rate=16000):
    """Generate silent audio as placeholder"""
    num_samples = int(duration * sample_rate)
    return bytes(num_samples * 2)  # 16-bit samples

def process_interaction(resume_file, round_type, user_input, history, resume_text_state):
    # Initialize state
    history = history or []
    resume_text = resume_text_state or ""
    
    # Parse resume on first interaction
    if not resume_text and resume_file:
        try:
            resume_text = parse_resume(resume_file)
        except Exception as e:
            # For Gradio 3.x compatibility
            history.append(("system", f"⚠️ {str(e)}"))
            return history, None, "", history, resume_text
    
    # Handle empty resume case
    if not resume_text:
        history.append(("system", "⚠️ Please upload a resume first"))
        return history, None, "", history, resume_text
    
    # Build conversation history string
    conversation_str = "\n".join([f"{role}: {msg}" for role, msg in history])
    
    # Add current user input to conversation
    if user_input.strip():
        history.append(("user", user_input.strip()))
        conversation_str += f"\nuser: {user_input.strip()}"
    
    # Generate AI question
    start_time = time.time()
    ai_question = generate_question(conversation_str, resume_text, round_type)
    history.append(("assistant", ai_question))
    gen_time = time.time() - start_time
    logging.info(f"Generated question in {gen_time:.2f}s: {ai_question}")
    
    # Convert to speech
    start_time = time.time()
    audio = text_to_speech(ai_question)
    tts_time = time.time() - start_time
    logging.info(f"Generated audio in {tts_time:.2f}s")
    
    return history, audio, "", history, resume_text

def reset_state():
    """Reset conversation and resume text"""
    return [], None, "", [], ""

# Gradio Interface - Compatible with both 3.x and 4.x
with gr.Blocks(title="AI Interviewer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Cloud AI Interviewer")
    gr.Markdown("Upload your resume and start practicing interview questions!")
    
    with gr.Row():
        with gr.Column(scale=1):
            resume_input = gr.File(
                label="Upload CV (PDF)", 
                file_types=[".pdf"]
            )
            round_select = gr.Radio(
                ["HR", "Technical"], 
                label="Interview Type", 
                value="Technical"
            )
            clear_btn = gr.Button("🔄 Clear Conversation", variant="secondary")
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=400,
                # Gradio 4.x feature - comment out if using 3.x
                # avatar_images=[
                #     None,  # User avatar
                #     "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"  # AI avatar
                # ]
            )
            audio_output = gr.Audio(
                label="AI Question", 
                autoplay=True, 
                interactive=False
            )
            user_input = gr.Textbox(
                label="Your Answer", 
                placeholder="Type your response...",
                lines=2
            )
            submit_btn = gr.Button("📤 Send", variant="primary")
    
    # State management
    history_state = gr.State([])
    resume_text_state = gr.State("")
    
    # Event handling
    submit_btn.click(
        process_interaction,
        inputs=[resume_input, round_select, user_input, history_state, resume_text_state],
        outputs=[chatbot, audio_output, user_input, history_state, resume_text_state]
    )
    
    clear_btn.click(
        reset_state,
        inputs=[],
        outputs=[chatbot, audio_output, user_input, history_state, resume_text_state]
    )
    
    resume_input.change(
        reset_state,
        inputs=[],
        outputs=[chatbot, audio_output, user_input, history_state, resume_text_state]
    )
    
    # Additional keyboard shortcut
    user_input.submit(
        process_interaction,
        inputs=[resume_input, round_select, user_input, history_state, resume_text_state],
        outputs=[chatbot, audio_output, user_input, history_state, resume_text_state]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="localhost",
        server_port=7860,
        show_error=True
    )