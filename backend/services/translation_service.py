"""
Translation Service — facebook/nllb-200-distilled-600M
HuggingFace Task: Text2Text Generation (seq2seq)

Supports 200 languages. Used to:
  1. Translate farmer queries from regional languages → English for model processing
  2. Translate AI responses from English → farmer's detected language
"""
import logging
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)

NLLB_MODEL = "facebook/nllb-200-distilled-600M"

LANG_MAP = {
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "mr": "mar_Deva",
    "kn": "kan_Knda",
    "bn": "ben_Beng",
    "sw": "swh_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "pt": "por_Latn",
}


@lru_cache(maxsize=1)
def _load_model():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    logger.info("Loading NLLB translation model: %s", NLLB_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).to(device)
    return tokenizer, model


def translate(text: str, source_lang: str = "en", target_lang: str = "en") -> str:
    """Translate text between ISO 639-1 language codes."""
    if source_lang == target_lang or not text.strip():
        return text

    src_tag = LANG_MAP.get(source_lang, "eng_Latn")
    tgt_tag = LANG_MAP.get(target_lang, "eng_Latn")

    try:
        tokenizer, model = _load_model()
        device = next(model.parameters()).device

        inputs = tokenizer(text, return_tensors="pt", src_lang=src_tag).to(device)
        target_id = tokenizer.convert_tokens_to_ids(tgt_tag)
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=target_id,
            max_new_tokens=256,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as exc:
        logger.error("Translation failed [%s→%s]: %s", source_lang, target_lang, exc)
        return text


def to_english(text: str, source_lang: str) -> str:
    return translate(text, source_lang=source_lang, target_lang="en")


def from_english(text: str, target_lang: str) -> str:
    return translate(text, source_lang="en", target_lang=target_lang)
