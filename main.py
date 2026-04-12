from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import yt_dlp
import os

app = FastAPI()

# Allow the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PUT YOUR DEEPGRAM API KEY HERE ---
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
# --------------------------------------

# Helper function to map UI languages to Deepgram codes
def get_language_code(ui_lang: str) -> str:
    lang_map = {
        "English": "en", "Spanish": "es", "French": "fr", 
        "Arabic": "ar", "Portuguese": "pt", "German": "de", 
        "Hindi": "hi", "Japanese": "ja"
    }
    return lang_map.get(ui_lang, "en") # Defaults to English if not found

class LinkRequest(BaseModel):
    url: str
    language: str

@app.get("/")
def home():
    return {"message": "Engine is running!"}

# --- DOOR 1: FILE UPLOADS ---
@app.post("/transcribe-file")
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Form("auto")
):
    try:
        audio_data = await file.read()
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        payload: FileSource = {"buffer": audio_data}
        
        # Configure AI based on UI selections
        options = PrerecordedOptions(
            model="nova-3", 
            smart_format=True,
            detect_language=True if language == "auto" else False,
            language=get_language_code(language) if language != "auto" else None
        )
        
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        return {"transcript": transcript}
    except Exception as e:
        return {"error": str(e)}

# --- DOOR 2: LINKS ---
@app.post("/transcribe-link")
async def transcribe_link(request: LinkRequest):
    filename = "temp_audio.m4a"
    try:
        # 1. Download audio
        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'outtmpl': filename,
            'quiet': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([request.url])

        # 2. Read file
        with open(filename, "rb") as f:
            audio_data = f.read()

        # 3. Send to AI
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        payload: FileSource = {"buffer": audio_data}
        options = PrerecordedOptions(
            model="nova-3", 
            smart_format=True,
            detect_language=True if request.language == "auto" else False,
            language=get_language_code(request.language) if request.language != "auto" else None
        )
        
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]

        # 4. Clean up
        if os.path.exists(filename):
            os.remove(filename)

        return {"transcript": transcript}

    except Exception as e:
        if os.path.exists(filename):
            os.remove(filename)
        return {"error": str(e)}