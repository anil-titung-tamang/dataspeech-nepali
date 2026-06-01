# Modified for Nepali (ne) language support
# Uses espeak-ng via phonemizer for Nepali phoneme counting
# Falls back to Devanagari character-based counting if phonemizer unavailable

try:
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend
    _backend = EspeakBackend('ne', preserve_punctuation=False, with_stress=False)
    NEPALI_PHONEMIZER_AVAILABLE = True
except Exception:
    NEPALI_PHONEMIZER_AVAILABLE = False


def _get_nepali_phonemes(text):
    """
    Convert Nepali Devanagari text to a phoneme string.
    Uses espeak-ng via phonemizer if available, otherwise falls back
    to counting Devanagari Unicode characters (each akshar ≈ 1 phonetic unit).
    """
    if NEPALI_PHONEMIZER_AVAILABLE:
        try:
            return phonemize(
                text,
                backend='espeak',
                language='ne',
                preserve_punctuation=False,
                with_stress=False,
            )
        except Exception:
            pass
    # Fallback: count Devanagari characters (U+0900–U+097F)
    return ''.join(c for c in text if '\u0900' <= c <= '\u097F')


def rate_apply(batch, rank=None, audio_column_name="audio", text_column_name="text"):
    if isinstance(batch[text_column_name], list):
        speaking_rates, phonemes_list = [], []
        if "speech_duration" in batch:
            for text, duration in zip(batch[text_column_name], batch["speech_duration"]):
                phonemes = _get_nepali_phonemes(text)
                duration = duration if duration != 0 else 0.01
                speaking_rates.append(len(phonemes) / duration)
                phonemes_list.append(phonemes)
        else:
            for text, audio in zip(batch[text_column_name], batch[audio_column_name]):
                phonemes = _get_nepali_phonemes(text)
                sample_rate = audio["sampling_rate"]
                audio_length = len(audio["array"].squeeze()) / sample_rate
                speaking_rates.append(len(phonemes) / audio_length)
                phonemes_list.append(phonemes)
        batch["speaking_rate"] = speaking_rates
        batch["phonemes"] = phonemes_list
    else:
        phonemes = _get_nepali_phonemes(batch[text_column_name])
        if "speech_duration" in batch:
            audio_length = batch["speech_duration"] if batch["speech_duration"] != 0 else 0.01
        else:
            sample_rate = batch[audio_column_name]["sampling_rate"]
            audio_length = len(batch[audio_column_name]["array"].squeeze()) / sample_rate
        batch["speaking_rate"] = len(phonemes) / audio_length
        batch["phonemes"] = phonemes
    return batch