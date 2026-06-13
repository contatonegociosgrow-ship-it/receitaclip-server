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
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()


class URLRequest(BaseModel):
    url: str


def _extrair_video_id(url: str) -> str | None:
    """Extrai o ID do vídeo do YouTube de qualquer formato de URL"""
    import re
    patterns = [
        r'(?:v=|youtu\.be/)([^&?/\s]{11})',
        r'(?:embed/)([^&?/\s]{11})',
        r'(?:shorts/)([^&?/\s]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


@app.get("/")
def health():
    return {
        "status": "ReceitaClip server rodando",
        "ffmpeg": FFMPEG_PATH,
        "youtube_api": "configurada" if YOUTUBE_API_KEY else "não configurada"
    }


@app.post("/metadata")
async def get_metadata(request: URLRequest):
    """Extrai título e descrição completa do vídeo"""
    url = request.url

    # Verificar se é YouTube
    video_id = _extrair_video_id(url)
    is_youtube = video_id is not None

    if is_youtube and YOUTUBE_API_KEY:
        # Usar YouTube Data API v3 — mais confiável
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    "https://www.googleapis.com/youtube/v3/videos",
                    params={
                        "id": video_id,
                        "part": "snippet",
                        "key": YOUTUBE_API_KEY
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    items = data.get("items", [])

                    if items:
                        snippet = items[0].get("snippet", {})
                        titulo = snippet.get("title", "")
                        descricao = snippet.get("description", "")
                        canal = snippet.get("channelTitle", "")

                        print(f"YouTube API — Título: {titulo}")
                        print(f"Descrição ({len(descricao)} chars): {descricao[:300]}")

                        return {
                            "titulo": titulo,
                            "descricao": descricao,
                            "canal": canal,
                            "duracao": 0,
                            "fonte": "youtube_api"
                        }
                    else:
                        print("YouTube API — vídeo não encontrado")
                else:
                    print(f"YouTube API erro: {response.status_code} — {response.text}")

        except Exception as e:
            print(f"Erro YouTube API: {e}")

    # Fallback: yt-dlp para outras plataformas ou se API falhar
    try:
        ydl_opts = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
        }
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(
            None,
            lambda: _get_info(url, ydl_opts)
        )
        if not info:
            return {
                "titulo": "",
                "descricao": "",
                "canal": "",
                "duracao": 0
            }

        return {
            "titulo": info.get('title', '') or '',
            "descricao": info.get('description', '') or '',
            "canal": info.get('uploader', '') or '',
            "duracao": info.get('duration', 0) or 0,
            "fonte": "yt_dlp"
        }
    except Exception as e:
        print(f"Erro yt-dlp metadata: {e}")
        return {
            "titulo": "",
            "descricao": "",
            "canal": "",
            "duracao": 0,
            "erro": str(e)
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

            # Extrair metadados
            descricao_video = ""
            titulo_video = ""
            canal_video = ""

            try:
                video_id = _extrair_video_id(url)
                is_youtube = video_id is not None

                if is_youtube and YOUTUBE_API_KEY:
                    # YouTube — usar API oficial
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        response = await client.get(
                            "https://www.googleapis.com/youtube/v3/videos",
                            params={
                                "id": video_id,
                                "part": "snippet",
                                "key": YOUTUBE_API_KEY
                            }
                        )
                        if response.status_code == 200:
                            data = response.json()
                            items = data.get("items", [])
                            if items:
                                snippet = items[0].get("snippet", {})
                                titulo_video = snippet.get("title", "")
                                descricao_video = snippet.get("description", "")
                                canal_video = snippet.get("channelTitle", "")
                                print(f"YouTube API OK — Descrição: {len(descricao_video)} chars")
                else:
                    # Outras plataformas — yt-dlp
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
                        canal_video = info.get('uploader', '') or ''
                        print(f"yt-dlp metadata OK — Descrição: {len(descricao_video)} chars")

            except Exception as e:
                print(f"Erro ao extrair metadados: {e}")

            # Extrair thumbnail
            thumbnail_base64 = await _extrair_thumbnail(url, tmpdir)

            # Para YouTube — usar thumbnail da API oficial
            if _extrair_video_id(url) and not thumbnail_base64:
                vid = _extrair_video_id(url)
                thumb_url = f"https://img.youtube.com/vi/{vid}/maxresdefault.jpg"
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        r = await client.get(thumb_url)
                        if r.status_code == 200:
                            thumbnail_base64 = f"data:image/jpeg;base64,{base64.b64encode(r.content).decode()}"
                except Exception:
                    pass

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

            # Verificar MP3
            mp3_files = [f for f in os.listdir(tmpdir) if f.endswith(".mp3")]
            if not mp3_files:
                raise HTTPException(
                    status_code=422,
                    detail="Não foi possível extrair o áudio deste vídeo"
                )

            audio_file = os.path.join(tmpdir, mp3_files[0])
            file_size = os.path.getsize(audio_file)
            print(f"Tamanho do áudio: {file_size / 1024 / 1024:.1f}MB")

            if file_size > 25 * 1024 * 1024:
                raise HTTPException(
                    status_code=422,
                    detail="Vídeo muito longo. Use vídeos de até 10 minutos."
                )

            # Transcrever
            transcricao = await _transcrever_groq(audio_file)
            print(f"Transcrição ({len(transcricao)} chars): {transcricao[:200]}")

            return {
                "sucesso": True,
                "transcricao": transcricao,
                "descricao": descricao_video,
                "titulo": titulo_video,
                "canal": canal_video,
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


async def _extrair_thumbnail(url: str, tmpdir: str):
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

        if info and info.get('thumbnail'):
            thumb_url = info['thumbnail']
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(thumb_url)
                if response.status_code == 200:
                    b64 = base64.b64encode(response.content).decode()
                    return f"data:image/jpeg;base64,{b64}"

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

        resultado = response.json()
        if isinstance(resultado, dict):
            return resultado.get('text', '')
        return str(resultado)
