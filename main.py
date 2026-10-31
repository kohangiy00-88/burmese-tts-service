import os
import uuid
import tempfile

import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Burmese TTS Service")

VOICES = {
    "male": "my-MM-ThihaNeural",
    "female": "my-MM-NilarNeural",
}


class TTSRequest(BaseModel):
    text: str
    voice: str = "male"


@app.get("/")
def home():
    return {
        "success": True,
        "service": "Burmese TTS",
        "voices": VOICES
    }


@app.get("/voices")
def get_voices():
    return {
        "success": True,
        "voices": VOICES
    }


@app.post("/tts")
async def create_tts(request: TTSRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="စာသားထည့်ပါ။"
        )

    if len(text) > 5000:
        raise HTTPException(
            status_code=413,
            detail="စာလုံး 5000 ထက် မကျော်ရပါ။"
        )

    voice = VOICES.get(request.voice)

    if not voice:
        raise HTTPException(
            status_code=400,
            detail="Voice ရွေးချယ်မှု မမှန်ပါ။"
        )

    filename = os.path.join(
        tempfile.gettempdir(),
        f"{uuid.uuid4()}.mp3"
    )

    try:
        communicate = edge_tts.Communicate(
            text,
            voice
        )

        await communicate.save(filename)

        return FileResponse(
            filename,
            media_type="audio/mpeg",
            filename="burmese-voice.mp3"
        )

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"TTS မအောင်မြင်ပါ: {str(error)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
        )
