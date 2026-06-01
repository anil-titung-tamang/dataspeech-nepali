"""
dataspeech/cpu_enrichments/rate.py  — Nepali-adapted version
=============================================================
Drop-in replacement for the upstream English rate.py.

Strategy (in priority order):
  1. espeak-ng with the Nepali backend ('ne') via subprocess — most accurate.
  2. Pure-Python Devanagari syllable counting — fast, zero-dependency fallback.

Both paths produce a `phonemes` string whose *length* (character count) is used
as the numerator for speaking_rate = phonemes / audio_duration, matching the
upstream API exactly so all downstream scripts (metadata_to_text.py, etc.) work
unchanged.

Install requirement for path 1:
    sudo apt-get install espeak-ng          # and the Nepali data package
    pip install phonemizer                  # Python wrapper (optional, we use subprocess)

If espeak-ng is unavailable the code silently falls back to path 2 — no crash.
"""

import re
import subprocess
import unicodedata

# ---------------------------------------------------------------------------
# Devanagari syllable-counting fallback
# ---------------------------------------------------------------------------
# Nepali is written in Devanagari. Each syllable is roughly one vowel nucleus.
# We count:
#   - Independent vowels (U+0904..U+0914, U+0960..U+0961)
#   - Vowel signs (matras) attached to consonants (U+093A..U+094C, U+0955..U+0957,
#     U+0962..U+0963)
#   - Consonants *not* followed by a virama or another consonant with halant
#     (i.e. bare consonants carry the inherent 'a' vowel)
# This is an approximation; espeak-ng is more accurate.

_DEVANAGARI_VOWEL_INDEPENDENT = re.compile(
    r"[\u0904-\u0914\u0960\u0961\u0972]"
)
_DEVANAGARI_VOWEL_SIGN = re.compile(
    r"[\u093A-\u094C\u0955-\u0957\u0962\u0963]"
)
_DEVANAGARI_CONSONANT = re.compile(r"[\u0915-\u0939\u0958-\u095F\u0978-\u097F]")
_HALANT = "\u094D"  # virama — suppresses inherent vowel


def _devanagari_syllables(text: str) -> str:
    """
    Return a proxy 'phoneme' string for Devanagari text.
    Length of the returned string ≈ syllable count.
    """
    syllables = []
    chars = list(text)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if _DEVANAGARI_VOWEL_INDEPENDENT.match(ch):
            syllables.append(ch)
            i += 1
        elif _DEVANAGARI_CONSONANT.match(ch):
            # Consonant followed by virama → no inherent vowel, skip nucleus
            if i + 1 < len(chars) and chars[i + 1] == _HALANT:
                i += 2  # consonant + virama consumed; next consonant will decide
            else:
                syllables.append(ch)  # consonant + inherent 'a'
                i += 1
                # absorb any following matra / nukta
                while i < len(chars) and _DEVANAGARI_VOWEL_SIGN.match(chars[i]):
                    i += 1
        else:
            i += 1  # punctuation, digits, spaces — skip
    return "".join(syllables) if syllables else text  # never return empty


# ---------------------------------------------------------------------------
# espeak-ng backend
# ---------------------------------------------------------------------------
_ESPEAK_AVAILABLE: bool | None = None  # cached after first check


def _check_espeak() -> bool:
    global _ESPEAK_AVAILABLE
    if _ESPEAK_AVAILABLE is not None:
        return _ESPEAK_AVAILABLE
    try:
        result = subprocess.run(
            ["espeak-ng", "--version"],
            capture_output=True,
            timeout=5,
        )
        _ESPEAK_AVAILABLE = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _ESPEAK_AVAILABLE = False
    return _ESPEAK_AVAILABLE


def _espeak_phonemize(text: str, lang: str = "ne") -> str:
    """
    Call espeak-ng and return IPA string.
    Returns empty string on failure so caller can fall back.
    """
    try:
        result = subprocess.run(
            ["espeak-ng", "-v", lang, "--ipa", "-q", "--", text],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            # espeak-ng inserts spaces between words; strip control chars
            ipa = result.stdout.strip()
            # Remove stress marks and syllable boundaries for clean length count
            ipa = re.sub(r"[ˈˌ._\n]", "", ipa)
            return ipa
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Public API (matches upstream rate.py exactly)
# ---------------------------------------------------------------------------

def _phonemize(text: str) -> str:
    """Return a phoneme/syllable string for the given Nepali text."""
    if _check_espeak():
        ipa = _espeak_phonemize(text, lang="ne")
        if ipa:
            return ipa
    # Fallback: Devanagari syllable counting
    return _devanagari_syllables(text)


def rate_apply(batch, rank=None, audio_column_name="audio", text_column_name="text"):
    """
    Compute speaking_rate and phonemes for a batch.
    API is identical to the upstream English rate.py so all callers work unchanged.
    """
    if isinstance(batch[text_column_name], list):
        speaking_rates = []
        phonemes_list = []

        if "speech_duration" in batch:
            for text, audio_duration in zip(
                batch[text_column_name], batch["speech_duration"]
            ):
                phonemes = _phonemize(text)
                audio_duration = audio_duration if audio_duration != 0 else 0.01
                speaking_rate = len(phonemes) / audio_duration
                speaking_rates.append(speaking_rate)
                phonemes_list.append(phonemes)
        else:
            for text, audio in zip(
                batch[text_column_name], batch[audio_column_name]
            ):
                phonemes = _phonemize(text)
                sample_rate = audio["sampling_rate"]
                audio_length = len(audio["array"].squeeze()) / sample_rate
                audio_length = audio_length if audio_length != 0 else 0.01
                speaking_rate = len(phonemes) / audio_length
                speaking_rates.append(speaking_rate)
                phonemes_list.append(phonemes)

        batch["speaking_rate"] = speaking_rates
        batch["phonemes"] = phonemes_list

    else:
        phonemes = _phonemize(batch[text_column_name])
        if "speech_duration" in batch:
            audio_length = batch["speech_duration"] if batch["speech_duration"] != 0 else 0.01
        else:
            sample_rate = batch[audio_column_name]["sampling_rate"]
            audio_length = len(batch[audio_column_name]["array"].squeeze()) / sample_rate
            audio_length = audio_length if audio_length != 0 else 0.01
        speaking_rate = len(phonemes) / audio_length
        batch["speaking_rate"] = speaking_rate
        batch["phonemes"] = phonemes

    return batch