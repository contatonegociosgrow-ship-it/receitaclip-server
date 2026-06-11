from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yt_dlp
import httpx
import os
import tempfile
import asyncio
import imageio_ffmpeg

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
FFMPEG_DIR = os.path.dirname(FFMPEG_PATH)

class URLRequest(BaseModel):
    url: str

@app.get("/")
def health():
    return {
        "status": "ReceitaClip server rodando",
        "ffmpeg": FFMPEG_PATH
    }

@app.post("/transcrever")
async def transcrever(request: URLRequest):
    url = request.url

    if not url:
        raise HTTPException(status_code=400, detail="URL não fornecida")

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY não configurada")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:

            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "64",
                }],
                "ffmpeg_location": FFMPEG_DIR,
                "quiet": True,
                "no_warnings": True,
                "extract_flat": False,
            }

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: _download(url, ydl_opts))

            mp3_files = [f for f in os.listdir(tmpdir) if f.endswith(".mp3")]
            if not mp3_files:
                raise HTTPException(status_code=422, detail="Não foi possível extrair o áudio deste vídeo")

            audio_file = os.path.join(tmpdir, mp3_files[0])

            file_size = os.path.getsize(audio_file)
            if file_size > 25 * 1024 * 1024:
                raise HTTPException(status_code=422, detail="Vídeo muito longo. Use vídeos de até 10 minutos.")

            transcricao = await _transcrever_groq(audio_file)

            return {
                "sucesso": True,
                "transcricao": transcricao,
                "plataforma": _detectar_plataforma(url)
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar vídeo: {str(e)}")


def _download(url: str, opts: dict):
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])


def _detectar_plataforma(url: str) -> str:
    if "tiktok.com" in url:
        return "tiktok"
    elif "instagram.com" in url:
        return "instagram"
    elif "facebook.com" in url or "fb.watch" in url:
        return "facebook"
    return "outro"


async def _transcrever_groq(audio_path: str) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(audio_path, "rb") as f:
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": ("audio.mp3", f, "audio/mpeg")},
                data={
                    "model": "whisper-large-v3-turbo",
                    "language": "pt",
                    "response_format": "text"
                }
            )

        if response.status_code != 200:
            raise Exception(f"Erro no Whisper: {response.text}")

        return response.text
