from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yt_dlp
import httpx
import os
import tempfile
import asyncio
import imageio_ffmpeg
import base64
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
    expose_headers=["*"],
    max_age=3600,
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()


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

            # Extrair metadados do vídeo (título e descrição)
            descricao_video = ""
            titulo_video = ""
            try:
                ydl_opts_meta = {
                    "skip_download": True,
                    "quiet": True,
                    "no_warnings": True,
                }
                loop = asyncio.get_event_loop()
                info = await loop.run_in_executor(
                    None,
                    lambda: _get_info(url, ydl_opts_meta)
                )
                if info:
                    descricao_video = info.get('description', '') or ''
                    titulo_video = info.get('title', '') or ''
                    print(f"Título: {titulo_video}")
                    print(f"Descrição ({len(descricao_video)} chars): {descricao_video[:200]}")
            except Exception as e:
                print(f"Erro ao extrair metadados: {e}")

            # Extrair thumbnail
            thumbnail_base64 = await _extrair_thumbnail(url, tmpdir)

            # Download do áudio
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "128",
                }],
                "ffmpeg_location": FFMPEG_PATH,
                "quiet": True,
                "no_warnings": True,
                "extract_flat": False,
                "writethumbnail": True,
            }

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: _download(url, ydl_opts))

            # Pegar thumbnail do download se não conseguiu antes
            if not thumbnail_base64:
                img_extensions = [".jpg", ".jpeg", ".png", ".webp"]
                for ext in img_extensions:
                    thumb_files = [f for f in os.listdir(tmpdir) if f.endswith(ext)]
                    if thumb_files:
                        with open(os.path.join(tmpdir, thumb_files[0]), "rb") as img:
                            thumbnail_base64 = f"data:image/jpeg;base64,{base64.b64encode(img.read()).decode()}"
                        break

            # Verificar se o MP3 foi gerado
            mp3_files = [f for f in os.listdir(tmpdir) if f.endswith(".mp3")]
            if not mp3_files:
                raise HTTPException(
                    status_code=422,
                    detail="Não foi possível extrair o áudio deste vídeo"
                )

            audio_file = os.path.join(tmpdir, mp3_files[0])

            # Verificar tamanho
            file_size = os.path.getsize(audio_file)
            print(f"Tamanho do áudio: {file_size / 1024 / 1024:.1f}MB")

            if file_size > 25 * 1024 * 1024:
                raise HTTPException(
                    status_code=422,
                    detail="Vídeo muito longo. Use vídeos de até 10 minutos."
                )

            # Transcrever com Whisper
            transcricao = await _transcrever_groq(audio_file)
            print(f"Transcrição ({len(transcricao)} chars): {transcricao[:300]}")

            return {
                "sucesso": True,
                "transcricao": transcricao,
                "descricao": descricao_video,
                "titulo": titulo_video,
                "thumbnail": thumbnail_base64,
                "plataforma": _detectar_plataforma(url)
            }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Erro geral: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar vídeo: {str(e)}"
        )


async def _extrair_thumbnail(url: str, tmpdir: str) -> str | None:
    """Tenta extrair thumbnail via info extraction"""
    try:
        ydl_opts_info = {
            "skip_download": True,
            "writethumbnail": True,
            "outtmpl": os.path.join(tmpdir, "thumb.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }

        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(
            None,
            lambda: _get_info(url, ydl_opts_info)
        )

        # Tentar pegar thumbnail da URL direta
        if info and info.get('thumbnail'):
            thumb_url = info['thumbnail']
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(thumb_url)
                if response.status_code == 200:
                    b64 = base64.b64encode(response.content).decode()
                    return f"data:image/jpeg;base64,{b64}"

        # Tentar arquivo gerado pelo writethumbnail
        img_extensions = [".jpg", ".jpeg", ".png", ".webp"]
        for ext in img_extensions:
            thumb_files = [
                f for f in os.listdir(tmpdir)
                if f.startswith("thumb") and f.endswith(ext)
            ]
            if thumb_files:
                with open(os.path.join(tmpdir, thumb_files[0]), "rb") as img:
                    return f"data:image/jpeg;base64,{base64.b64encode(img.read()).decode()}"

    except Exception as e:
        print(f"Thumbnail extraction failed: {e}")

    return None


def _get_info(url: str, opts: dict):
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            return ydl.extract_info(url, download=True)
        except Exception as e:
            print(f"Erro _get_info: {e}")
            return None


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
    async with httpx.AsyncClient(timeout=120.0) as client:
        with open(audio_path, "rb") as f:
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": ("audio.mp3", f, "audio/mpeg")},
                data={
                    "model": "whisper-large-v3-turbo",
                    "language": "pt",
                    "response_format": "verbose_json",
                    "temperature": "0",
                    "prompt": "Esta é uma receita culinária brasileira. O apresentador lista ingredientes com quantidades em xícaras, colheres, gramas, unidades e explica o modo de preparo passo a passo. Preste atenção nas quantidades dos ingredientes."
                }
            )

        if response.status_code != 200:
            print(f"Erro Whisper: {response.text}")
            raise Exception(f"Erro no Whisper: {response.text}")

        # verbose_json retorna objeto com campo "text"
        resultado = response.json()
        if isinstance(resultado, dict):
            return resultado.get('text', '')
        return str(resultado)
