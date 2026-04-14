from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import yt_dlp
import os
import uuid
import tempfile

# Needed for yt-dlp on Render
os.environ["PATH"] += os.pathsep + "/usr/bin"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Cookies file path — place cookies.txt in the same folder as main.py
COOKIES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cookies.txt")


def get_language_code(ui_lang: str) -> str:
    lang_map = {
        "English": "en", "Spanish": "es", "French": "fr",
        "Arabic": "ar", "Portuguese": "pt", "German": "de",
        "Hindi": "hi", "Japanese": "ja"
    }
    return lang_map.get(ui_lang, "en")


def build_deepgram_options(language: str) -> PrerecordedOptions:
    """Build PrerecordedOptions — translate is handled separately via extra_params."""
    return PrerecordedOptions(
        model="nova-3",
        smart_format=True,
        detect_language=True if language == "auto" else False,
        language=get_language_code(language) if language != "auto" else None,
    )


def get_extra_params(task: str) -> dict:
    """Return extra query params for Deepgram if translation is requested."""
    return {"translate": "en"} if task == "translate" else {}


def extract_transcript(alts: dict, task: str) -> str:
    """Pull the right transcript field depending on task."""
    if task == "translate" and "translations" in alts:
        return alts["translations"][0]["transcript"]
    return alts["transcript"]


class LinkRequest(BaseModel):
    url: str
    language: str = "auto"
    task: str = "transcribe"


@app.get("/")
def home():
    return {"message": "Transcribbr engine is running!"}


# --- DOOR 1: FILE UPLOADS ---
@app.post("/transcribe-file")
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    task: str = Form("transcribe")
):
    try:
        audio_data = await file.read()

        if not audio_data:
            return {"error": "Uploaded file is empty."}

        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        payload: FileSource = {"buffer": audio_data}
        options = build_deepgram_options(language)
        extra_params = get_extra_params(task)

        response = deepgram.listen.rest.v("1").transcribe_file(
            payload, options, **extra_params
        )

        alts = response["results"]["channels"][0]["alternatives"][0]
        transcript = extract_transcript(alts, task)

        return {"transcript": transcript}

    except Exception as e:
        return {"error": str(e)}


# --- DOOR 2: LINKS ---
@app.post("/transcribe-link")
async def transcribe_link(request: LinkRequest):
    # Use a unique filename per request to avoid collisions on concurrent users
    unique_id = uuid.uuid4().hex
    temp_dir = tempfile.gettempdir()
    filename = os.path.join(temp_dir, f"transcribbr_{unique_id}.m4a")

    try:
        # Build yt-dlp options
        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'outtmpl': filename,
            'quiet': True,
            'no_warnings': True,
            # Use cookies if the file exists — handles YouTube + Instagram bot detection
            'cookiefile': COOKIES_PATH if os.path.exists(COOKIES_PATH) else None,
            # Spoof a real browser User-Agent to reduce bot flags
            'http_headers': {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/120.0.0.0 Safari/537.36'
                )
            },
            # Safety limits — prevent huge files from crashing the server
            'max_filesize': 200 * 1024 * 1024,  # 200MB cap
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([request.url])

        # Confirm file was actually downloaded
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            return {"error": "Failed to download audio. The link may be private or unsupported."}

        with open(filename, "rb") as f:
            audio_data = f.read()

        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        payload: FileSource = {"buffer": audio_data}
        options = build_deepgram_options(request.language)
        extra_params = get_extra_params(request.task)

        response = deepgram.listen.rest.v("1").transcribe_file(
            payload, options, **extra_params
        )

        alts = response["results"]["channels"][0]["alternatives"][0]
        transcript = extract_transcript(alts, request.task)

        return {"transcript": transcript}

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Always clean up temp file, even if something crashes
        if os.path.exists(filename):
            os.remove(filename)