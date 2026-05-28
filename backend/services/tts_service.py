"""
TTS Service — facebook/mms-tts (Massively Multilingual Speech)
HuggingFace Task: Text-to-Speech

MMS-TTS covers 1,100+ languages including regional Indian languages.
Returns raw WAV bytes that can be streamed directly to the client.
"""
import io
import logging
import struct
import wave
from functools import lru_cache

import numpy as np
import torch
from transformers import VitsModel, AutoTokenizer

logger = logging.getLogger(__name__)

# Language → MMS-TTS HuggingFace model ID
MMS_MODELS = {
    "en": "facebook/mms-tts-eng",
    "hi": "facebook/mms-tts-hin",
    "te": "facebook/mms-tts-tel",
    "ta": "facebook/mms-tts-tam",
    "mr": "facebook/mms-tts-mar",
    "kn": "facebook/mms-tts-kan",
    "bn": "facebook/mms-tts-ben",
}
DEFAULT_LANG = "en"


@lru_cache(maxsize=4)
def _load_tts(lang: str):
    model_id = MMS_MODELS.get(lang, MMS_MODELS[DEFAULT_LANG])
    logger.info("Loading MMS-TTS model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


def synthesize(text: str, lang: str = "en") -> bytes:
    """Convert text to WAV bytes using MMS-TTS."""
    lang = lang if lang in MMS_MODELS else DEFAULT_LANG
    tokenizer, model = _load_tts(lang)

    # MMS-TTS has a ~500 char limit per call; chunk if needed
    max_chars = 400
    chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
    all_audio: list[np.ndarray] = []

    with torch.no_grad():
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt")
            output = model(**inputs).waveform.squeeze().numpy()
            all_audio.append(output)
            # Small silence between chunks
            all_audio.append(np.zeros(int(model.config.sampling_rate * 0.3)))

    audio = np.concatenate(all_audio).astype(np.float32)
    # Normalise
    audio = audio / (np.abs(audio).max() + 1e-6)
    pcm = (audio * 32767).astype(np.int16)

    sample_rate = model.config.sampling_rate
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()
