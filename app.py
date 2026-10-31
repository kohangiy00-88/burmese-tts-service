import os
import io
import re
import asyncio
import zipfile
import tempfile
import threading

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel
from pydub import AudioSegment
import edge_tts


app = FastAPI(title="Burmese TTS Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")

model = None
model_lock = threading.Lock()

VALID_VOICES = {
    "my-MM-ThihaNeural",
    "my-MM-NilarNeural"
}


class TTSRequest(BaseModel):
    text: str
    voice: str = "my-MM-ThihaNeural"


def get_voice(name):
    return name if name in VALID_VOICES else "my-MM-ThihaNeural"


def get_model():
    global model

    with model_lock:
        if model is None:
            model = WhisperModel(
                WHISPER_MODEL,
                device="cpu",
                compute_type="int8"
            )

    return model


async def create_speech(text, voice, output_path):
    text = str(text).strip()

    if not text:
        raise ValueError("စာသား မရှိပါ။")

    communicator = edge_tts.Communicate(
        text=text,
        voice=get_voice(voice)
    )

    await communicator.save(output_path)


def translate_to_burmese(text):
    text = text.strip()

    if not text:
        return ""

    try:
        result = GoogleTranslator(
            source="auto",
            target="my"
        ).translate(text)

        if not result:
            raise ValueError("ဘာသာပြန်စာသား မရရှိပါ။")

        return result.strip()

    except Exception as exc:
        raise RuntimeError(
            f"မြန်မာဘာသာပြန် မအောင်မြင်ပါ: {str(exc)}"
        )


def srt_time(ms):
    ms = max(0, int(ms))

    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    millis = ms % 1000

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d},"
        f"{millis:03d}"
    )


def build_srt(captions):
    result = []

    for index, item in enumerate(captions, start=1):
        result.append(
            f"{index}\n"
            f"{srt_time(item['start'])} --> "
            f"{srt_time(item['end'])}\n"
            f"{item['text']}\n"
        )

    return "\n".join(result)


@app.get("/")
@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "success": True,
        "service": "Burmese TTS Service",
        "status": "OK",
        "voices": [
            "my-MM-ThihaNeural",
            "my-MM-NilarNeural"
        ]
    }


@app.post("/api/tts")
def text_to_speech(data: TTSRequest):
    text = data.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="စာသား မထည့်ထားပါ။"
        )

    if len(text) > 30000:
        raise HTTPException(
            status_code=413,
            detail="စာသားအရှည် 30000 characters ထက် မကျော်ရပါ။"
        )

    try:
        with tempfile.TemporaryDirectory() as folder:
            output = os.path.join(folder, "voice.mp3")

            asyncio.run(
                create_speech(
                    text,
                    get_voice(data.voice),
                    output
                )
            )

            with open(output, "rb") as file:
                audio_bytes = file.read()

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": 'attachment; filename="voice.mp3"'
            }
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"အသံဖန်တီးမှု မအောင်မြင်ပါ: {str(exc)}"
        )


@app.post("/api/auto-dub")
def auto_dub(
    file: UploadFile = File(...),
    voice: str = Form("my-MM-ThihaNeural")
):
    """
    Input:
      - file: extracted video audio (WAV recommended)
      - voice: my-MM-ThihaNeural / my-MM-NilarNeural

    Output ZIP:
      - voice.mp3
      - captions.srt
    """

    voice = get_voice(voice)

    try:
        with tempfile.TemporaryDirectory() as folder:
            input_audio = os.path.join(folder, "source_audio.wav")

            with open(input_audio, "wb") as target:
                while True:
                    chunk = file.file.read(1024 * 1024)

                    if not chunk:
                        break

                    target.write(chunk)

            try:
                source_audio = AudioSegment.from_file(input_audio)
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"အသံဖိုင် ဖတ်မရပါ: {str(exc)}"
                )

            total_duration = len(source_audio)

            if total_duration <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="အသံဖိုင်ထဲမှာ အသံမတွေ့ပါ။"
                )

            whisper = get_model()

            try:
                segments, info = whisper.transcribe(
                    input_audio,
                    beam_size=1,
                    vad_filter=True
                )

                source_segments = list(segments)

            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"အသံစာသားဖော်ထုတ်မှု မအောင်မြင်ပါ: {str(exc)}"
                )

            if not source_segments:
                raise HTTPException(
                    status_code=422,
                    detail="မူရင်းအသံထဲမှာ ပြောဆိုထားတဲ့ စာသား မတွေ့ပါ။"
                )

            # Build the new Myanmar voice track.
            voice_track = AudioSegment.silent(
                duration=total_duration + 5000,
                frame_rate=24000
            ).set_channels(1)

            captions = []

            for index, segment in enumerate(source_segments):
                original_text = segment.text.strip()

                if not original_text:
                    continue

                try:
                    burmese_text = translate_to_burmese(
                        original_text
                    )
                except Exception as exc:
                    raise HTTPException(
                        status_code=502,
                        detail=str(exc)
                    )

                if not burmese_text:
                    continue

                segment_path = os.path.join(
                    folder,
                    f"segment_{index}.mp3"
                )

                try:
                    asyncio.run(
                        create_speech(
                            burmese_text,
                            voice,
                            segment_path
                        )
                    )

                    speech = AudioSegment.from_file(
                        segment_path,
                        format="mp3"
                    )

                except Exception as exc:
                    raise HTTPException(
                        status_code=500,
                        detail=f"မြန်မာအသံဖန်တီးမှု မအောင်မြင်ပါ: {str(exc)}"
                    )

                speech = speech.set_frame_rate(
                    voice_track.frame_rate
                ).set_channels(1)

                start_ms = max(
                    0,
                    int(float(segment.start) * 1000)
                )

                end_ms = start_ms + len(speech)

                # Overlay voice at the original speech position.
                voice_track = voice_track.overlay(
                    speech,
                    position=start_ms
                )

                captions.append({
                    "start": start_ms,
                    "end": end_ms,
                    "text": burmese_text
                })

            if not captions:
                raise HTTPException(
                    status_code=422,
                    detail="ဘာသာပြန်စာသား မရရှိပါ။"
                )

            voice_path = os.path.join(folder, "voice.mp3")

            voice_track.export(
                voice_path,
                format="mp3",
                bitrate="128k"
            )

            srt_path = os.path.join(folder, "captions.srt")

            with open(srt_path, "w", encoding="utf-8") as file:
                file.write(build_srt(captions))

            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(
                zip_buffer,
                "w",
                compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.write(
                    voice_path,
                    arcname="voice.mp3"
                )
                archive.write(
                    srt_path,
                    arcname="captions.srt"
                )

            zip_buffer.seek(0)

            return StreamingResponse(
                zip_buffer,
                media_type="application/zip",
                headers={
                    "Content-Disposition": 'attachment; filename="burmese-dub.zip"'
                }
            )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Auto-Dub မအောင်မြင်ပါ: {str(exc)}"
    )
