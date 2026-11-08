"""
Kataria AI Copilot - Diarized Call Transcriber (v3 standalone)

Standalone monolith = the v2 transcriber (Qwen roles + script unification + full
number ITN) with the NEW v3 AUDIO INPUT SYSTEM: each diarized turn is sliced with
a small SEGMENT_PAD on both sides before ASR, matching prepare_data_v3 so the ASR
sees the same context padding at inference that it saw in training (train/infer
parity). Audio load is 16 kHz mono (unchanged). Nothing else differs from v2.

Same pipeline as "better diarized cli batch.py" with three changes:
  1. ASR = the fine-tuned code-switch model (NeMo .nemo), not the HF AutoModel.
  2. Long diarized turns are hard-split to <=MAX_SEGMENT_DURATION so the ASR
     never sees a clip longer than it trained on (~30s).
  3. A POST-PROCESSING stage on each segment's text:
       a) transliteration lexicon: replace known Gujarati-script English
          loanwords with their English spelling (translit_lexicon.json).
       b) number normalization (ITN): turn spoken number words (English AND
          Gujarati) into digits -> "nine nine seven five" -> 9975,
          "ચાર હજાર" -> 4000, "પંદરથી" -> "15થી".

Diarization (pyannote) and role labeling (Qwen3) are unchanged. The diarizer can be
switched with DIARIZER (see Config): pyannote by default, or a fine-tuned Sortformer.

Usage:
  python "better diarized cli batch v2.py" /path/to/call.mp3
  python "better diarized cli batch v2.py" /path/to/folder
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ── Suppress noisy third-party warnings before any imports that trigger them ──
import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning,  message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning,  message=".*torchaudio.*")
warnings.filterwarnings("ignore", category=UserWarning,  module="pyannote")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*ffmpeg.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*avconv.*")
# Silence NeMo's verbose INFO/WARNING logs (training config, Megatron, OneLogger)
logging.getLogger("nemo_logger").setLevel(logging.ERROR)
logging.getLogger("nemo").setLevel(logging.ERROR)
# Silence onnxruntime GPU device-discovery warnings
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")  # 3 = ERROR only

import re
import sys
import json
import time
import tempfile
import unicodedata

import torch
import librosa
import soundfile
from pyannote.audio import Pipeline
import ollama
import nemo.collections.asr as nemo_asr

# Re-apply after NeMo initializes its own logger (NeMo may reset levels on import)
for _ln in ["nemo_logger", "nemo", "nemo.collections", "nemo.core", "nemo.utils"]:
    logging.getLogger(_ln).setLevel(logging.ERROR)

torch.set_num_threads(os.cpu_count() or 4)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ----------------------------
# Config
# ----------------------------
start_time = time.time()

_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent

# ASR: models/asr.nemo = best_v3 continued on DATASET LLM (Gujarati-weighted, 2026-10-05).
# ASR_MODEL in .env picks another .nemo (e.g. models/best_v3.nemo to roll back).
MODEL_PATH   = str(_ROOT / os.environ.get("ASR_MODEL", "models/asr.nemo"))
LEXICON_PATH = str(_ROOT / "models" / "translit_lexicon.json")
OUTPUT_DIR   = str(_ROOT / "transcripts")

# Force a single Indic script so output is never a Hindi/Gujarati mixture.
#   "gu"   -> transliterate any Devanagari into Gujarati (recommended for these calls)
#   "hi"   -> transliterate any Gujarati into Devanagari
#   "auto" -> per-segment: whichever Indic script dominates wins
#   None   -> disabled (leave mixed)
# English (Latin) and digits are never touched. Runs LAST in postprocess, so Hindi
# numbers still convert from their native forms first (बीस->20, सौ->100).
TARGET_SCRIPT = "auto"

# --- optional n-gram LM beam decoding (fixes garbled/mixed-script real words) ---
USE_LM    = False                                              # set True to enable n-gram LM beam decoding
LM_PATH   = str(_ROOT / "models" / "lm.bin")               # KenLM binary (lmplz output)
LM_CORPUS = str(_ROOT / "models" / "lm_corpus.txt")        # transcripts -> unigram list for pyctcdecode
LM_ALPHA = 0.15                                        # LM weight (0.15 = WER/CER sweet spot)
LM_BETA = 1.0                                          # word-insertion bonus
NUM_RATIO_GREEDY = 0.4                                 # segment >= this fraction of number tokens -> greedy (protect digits); else LM

QWEN_MODEL = "qwen3:8b"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Diarizer: "pyannote" (community-1, default) or the path to a fine-tuned NeMo
# Sortformer .nemo (absolute, or relative to the project root). The fine-tuned
# Nemotron-3 model (nemotron3_kataria.nemo) needs NeMo 3.1+.
DIARIZER = os.environ.get("DIARIZER", "pyannote")

MIN_SEGMENT_DURATION = 0.25
MERGE_GAP_THRESHOLD = 1.0     # merge same-speaker turns <= this gap (sec). 1.0 = the v3-package
                             #   / scribe value: keeps a spoken plate/date/amount in ONE clip so
                             #   the ASR never sees half a number ("GJ 23 CE 3401" split -> "GJ2").
                             #   0.30 over-split and garbled numbers at segment edges. Safe for
                             #   roles now that role labeling is a per-speaker VOTE, not segment order.
MAX_SEGMENT_DURATION = 28.0   # split diarized turns longer than this (ASR trained on <=30s)
SEGMENT_PAD = 0.10            # v3 audio-input parity. VERIFIED against training:
                             #   prepare_data_v3.write_dataset(pad=0.10) slices
                             #   a=max(0,int((start-0.10)*sr)); b=min(total,int((end+0.10)*sr))
                             #   at sr=16000 -- _slice_wav below is byte-for-byte identical, so
                             #   the ASR sees the same edge context at inference as in training.
                             #   Clamped to audio bounds; does NOT change diarization/merge/role timing.

DEALERSHIP_NAME_VARIANTS = ["કટારીયા", "કટારિયા", "કટારી", "kataria", "katariya"]  # real spellings only
# A dealership mention that co-occurs with one of these is a real self-introduction
# ("...Automobiles", "...માંથી") -> counts double in the agent vote. This is what
# separates the AGENT (who introduces the dealership) from a CUSTOMER who merely
# echoes the name ("Kataria?"), so a stray customer mention no longer flips roles.
INTRODUCTION_MARKERS = ["automobiles", "ઓટોમોબાઈલ", "ઓટોમોબાઇલ", "માંથી", "માથી"]

# ============================================================================
# POST-PROCESSING 1: transliteration lexicon
# ============================================================================

def load_lexicon(path):
    """Load {gujarati_word: english_word}. Returns {} silently if the file is absent."""
    if not os.path.isfile(path):
        return {}
    lex = json.load(open(path, encoding="utf-8"))
    print(f"Loaded transliteration lexicon: {len(lex)} entries")
    return lex


_WORD_SPLIT = re.compile(r"(\s+)")


def apply_lexicon(text, lex):
    """Replace whole Gujarati-script tokens that appear in the lexicon with English.

    Whole-token match only (keeps surrounding punctuation), so it never rewrites
    a substring inside another word. Runs before number conversion.
    """
    if not lex:
        return text
    out = []
    for tok in _WORD_SPLIT.split(text):
        if not tok or tok.isspace():
            out.append(tok)
            continue
        core = tok.strip(".,?!।\"'()")
        lead = tok[:len(tok) - len(tok.lstrip(".,?!।\"'()"))]
        trail = tok[len(tok.rstrip(".,?!।\"'()")):]
        out.append(lead + lex[core] + trail if core in lex else tok)
    return "".join(out)


# ============================================================================
# POST-PROCESSING 2: number words -> digits (ITN)
# ============================================================================
# English is complete; Gujarati covers the common words in dealership calls.
# Extend GU_UNITS / GU_SCALES as you spot missed numbers.

EN_UNITS = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
            "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
            "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
            "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
EN_SCALES = {"hundred": 100, "thousand": 1000, "lakh": 100000, "lac": 100000, "crore": 10000000, "million": 1000000}

# Full 0-99, GROUNDED against the real call transcripts. Includes the older
# spelling variants ચાલીસ(40)/સાઠ(60) as aliases so both forms convert.
GU_UNITS = {
    "શૂન્ય": 0, "એક": 1, "બે": 2, "ત્રણ": 3, "ચાર": 4, "પાંચ": 5, "છ": 6, "સાત": 7,
    "આઠ": 8, "નવ": 9, "દસ": 10, "અગિયાર": 11, "બાર": 12, "તેર": 13, "ચૌદ": 14,
    "પંદર": 15, "સોળ": 16, "સત્તર": 17, "અઢાર": 18, "ઓગણીસ": 19, "વીસ": 20,
    "એકવીસ": 21, "બાવીસ": 22, "તેવીસ": 23, "ચોવીસ": 24, "પચ્ચીસ": 25, "છવ્વીસ": 26,
    "સત્તાવીસ": 27, "અઠ્ઠાવીસ": 28, "ઓગણત્રીસ": 29, "ત્રીસ": 30, "એકત્રીસ": 31,
    "બત્રીસ": 32, "તેત્રીસ": 33, "ચોત્રીસ": 34, "પાંત્રીસ": 35, "છત્રીસ": 36,
    "સડત્રીસ": 37, "અડત્રીસ": 38, "ઓગણચાળીસ": 39, "ચાળીસ": 40, "ચાલીસ": 40,
    "એકતાળીસ": 41, "બેતાળીસ": 42, "તેતાળીસ": 43, "ચુમાળીસ": 44, "પિસ્તાળીસ": 45,
    "છેતાળીસ": 46, "સુડતાળીસ": 47, "અડતાળીસ": 48, "ઓગણપચાસ": 49, "પચાસ": 50,
    "એકાવન": 51, "બાવન": 52, "ત્રેપન": 53, "ચોપન": 54, "પંચાવન": 55, "છપ્પન": 56,
    "સત્તાવન": 57, "અઠ્ઠાવન": 58, "ઓગણસાઠ": 59, "સાઈઠ": 60, "સાઠ": 60, "એકસઠ": 61,
    "બાસઠ": 62, "ત્રેસઠ": 63, "ચોસઠ": 64, "પાંસઠ": 65, "છાસઠ": 66, "સડસઠ": 67,
    "અડસઠ": 68, "અગણોસિત્તેર": 69, "સિત્તેર": 70, "એકોતેર": 71, "બોતેર": 72,
    "તોતેર": 73, "ચુમોતેર": 74, "પંચોતેર": 75, "છોતેર": 76, "સિત્યોતેર": 77,
    "ઇઠ્યોતેર": 78, "ઓગણાએંસી": 79, "એંસી": 80, "એક્યાસી": 81, "બ્યાસી": 82,
    "ત્યાસી": 83, "ચોર્યાસી": 84, "પંચ્યાસી": 85, "છ્યાસી": 86, "સિત્યાસી": 87,
    "અઠ્યાસી": 88, "નેવ્યાસી": 89, "નેવું": 90, "એકાણું": 91, "બાણું": 92,
    "ત્રાણું": 93, "ચોરાણું": 94, "પંચાણું": 95, "છન્નું": 96, "સત્તાણું": 97,
    "અઠ્ઠાણું": 98, "નવ્વાણું": 99,
    # model-spelling variants (your fine-tuned ASR spells these differently than the
    # grounded set) -- add more here as you spot them, or re-ground on model output.
    "તેસઠ": 63, "છવીસ": 26}
GU_SCALES = {"સો": 100, "હજાર": 1000, "લાખ": 100000, "કરોડ": 10000000}
# Fused Gujarati centuries (grounded in the real transcripts). Explicit list only --
# we deliberately do NOT strip a generic '<NN>સો', because બેસો = "sit", not 200.
GU_HUND = {"એકસો": 100, "બસ્સો": 200, "બસો": 200, "ત્રણસો": 300, "ચારસો": 400,
           "પાંચસો": 500, "છસો": 600, "સાતસો": 700, "આઠસો": 800, "નવસો": 900}

# ----------------------------------------------------------------------------
# Hindi (Devanagari) numbers. The model is code-switch (Guj/Hindi/Eng), so Hindi
# numbers can surface in Devanagari. We normalize spelling variants -- nuqta
# (ज़/ज, ढ़/ढ) and anusvara vs chandrabindu (ं/ँ) -- so पांच==पाँच, हजार==हज़ार.
# ----------------------------------------------------------------------------
def _hin(s):
    return s.replace("़", "").replace("ँ", "ं")

_HI_RAW = {"शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "छह": 6, "सात": 7,
    "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15,
    "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20, "इक्कीस": 21, "बाईस": 22,
    "तेईस": 23, "चौबीस": 24, "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29,
    "तीस": 30, "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35, "छत्तीस": 36,
    "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39, "चालीस": 40, "इकतालीस": 41, "बयालीस": 42,
    "तैंतालीस": 43, "चौवालीस": 44, "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48,
    "उनचास": 49, "पचास": 50, "इक्यावन": 51, "बावन": 52, "तिरेपन": 53, "चौवन": 54, "पचपन": 55,
    "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59, "साठ": 60, "इकसठ": 61, "बासठ": 62,
    "तिरेसठ": 63, "चौंसठ": 64, "पैंसठ": 65, "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69,
    "सत्तर": 70, "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75,
    "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79, "अस्सी": 80, "इक्यासी": 81,
    "बयासी": 82, "तिरासी": 83, "चौरासी": 84, "पचासी": 85, "छियासी": 86, "सत्तासी": 87,
    "अट्ठासी": 88, "नवासी": 89, "नब्बे": 90, "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93,
    "चौरानवे": 94, "पंचानवे": 95, "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99}
HI_UNITS = {_hin(k): v for k, v in _HI_RAW.items()}
HI_SCALES = {_hin(k): v for k, v in {"सौ": 100, "हज़ार": 1000, "लाख": 100000, "करोड़": 10000000}.items()}

# ----------------------------------------------------------------------------
# Ordinals (Gujarati + Hindi) -> digit + ordinal suffix (whole-token replace).
# ----------------------------------------------------------------------------
ORD = {
    "પહેલો":"1લો","પહેલી":"1લી","પહેલું":"1લું","બીજો":"2જો","બીજી":"2જી","બીજું":"2જું",
    "ત્રીજો":"3જો","ત્રીજી":"3જી","ત્રીજું":"3જું","ચોથો":"4થો","ચોથી":"4થી","ચોથું":"4થું",
    "પાંચમો":"5મો","પાંચમી":"5મી","છઠ્ઠો":"6ઠ્ઠો","છઠ્ઠી":"6ઠ્ઠી","સાતમો":"7મો","સાતમી":"7મી",
    "આઠમો":"8મો","આઠમી":"8મી","નવમો":"9મો","નવમી":"9મી","દસમો":"10મો","દસમી":"10મી",
    "पहला":"1ला","पहली":"1ली","दूसरा":"2रा","दूसरी":"2री","तीसरा":"3रा","तीसरी":"3री",
    "चौथा":"4था","चौथी":"4थी","पाँचवाँ":"5वाँ","पांचवां":"5वां","छठा":"6ठा","छठी":"6ठी",
    "सातवाँ":"7वाँ","आठवाँ":"8वाँ","नौवाँ":"9वाँ","दसवाँ":"10वाँ",
}

def apply_ordinals(text):
    """Whole-token replace ordinal words with digit+suffix (પહેલી -> 1લી)."""
    out = []
    for tok in _WORD_SPLIT.split(text):
        if not tok or tok.isspace():
            out.append(tok); continue
        core = tok.strip(".,?!।\"'()")
        lead = tok[:len(tok) - len(tok.lstrip(".,?!।\"'()"))]
        trail = tok[len(tok.rstrip(".,?!।\"'()")):]
        out.append(lead + ORD[core] + trail if core in ORD else tok)
    return "".join(out)

# ----------------------------------------------------------------------------
# double / triple markers -> repeated digit, concatenated (ડબલ પાંચ -> 55).
# Safe: the marker signals concatenation intent, so no digit-grouping ambiguity.
# ----------------------------------------------------------------------------
REPEAT_MARK = {"ડબલ": 2, "ટ્રિપલ": 3, "डबल": 2, "ट्रिपल": 3, "double": 2, "triple": 3}

def _num_word(core):
    """Return the integer value of a single number word (gu/hi/en), or None."""
    if core in GU_UNITS: return GU_UNITS[core]
    if _hin(core) in HI_UNITS: return HI_UNITS[_hin(core)]
    if core.lower() in EN_UNITS: return EN_UNITS[core.lower()]
    return None

def expand_repeat(tokens):
    """[ડબલ, પાંચ] -> ['55']; leaves 'ડબલ કેબિન' (double + non-digit) untouched."""
    out, i = [], 0
    while i < len(tokens):
        c = tokens[i].strip(_PUNCT)
        if c in REPEAT_MARK and i + 1 < len(tokens):
            v = _num_word(tokens[i + 1].strip(_PUNCT))
            if v is not None and 0 <= v <= 9:
                out.append(str(v) * REPEAT_MARK[c]); i += 2; continue
        out.append(tokens[i]); i += 1
    return out

# ----------------------------------------------------------------------------
# Clock times: fraction modifiers -> HH:MM. Runs BEFORE numbers_to_digits.
# ----------------------------------------------------------------------------
_QUARTER = {_hin(x) for x in ("સવા", "सवा")}
_HALF    = {_hin(x) for x in ("સાડા", "साढ़े")}
_TO      = {_hin(x) for x in ("પોણા", "पौने")}
_CLOCK_STANDALONE = {_hin(k): v for k, v in {"દોઢ": "1:30", "डेढ़": "1:30",
                                             "અઢી": "2:30", "ढाई": "2:30"}.items()}

def apply_clock_times(text):
    toks = text.split(); out = []; i = 0
    while i < len(toks):
        core = _hin(toks[i].strip(_PUNCT))
        if core in _CLOCK_STANDALONE:
            out.append(_CLOCK_STANDALONE[core]); i += 1; continue
        if (core in _QUARTER or core in _HALF or core in _TO) and i + 1 < len(toks):
            n = _num_word(toks[i + 1].strip(_PUNCT))
            if n is not None and 1 <= n <= 12:
                if core in _QUARTER:  out.append(f"{n}:15")
                elif core in _HALF:   out.append(f"{n}:30")
                else:                 out.append(f"{12 if n == 1 else n - 1}:45")
                i += 2; continue
        out.append(toks[i]); i += 1
    return " ".join(out)

_PUNCT = ".,?!।\"'()-"

# Fused postpositions we strip (and reattach to the digit), ONLY when the stem
# is a number word. Two tiers because of collisions:
#   ANY  = safe off units OR scales ("15થી", "20માં", "હજારમાં")
#   UNIT = safe off UNIT words only, NEVER off scale words -- because the oblique
#          forms of "સો"/100 are common real words ("સોના"/gold, "સોની"/Soni,
#          "સોને"). Restricting to units lets "પંદરની"->"15ની" while "સોના",
#          "સોની" stay intact. Bare "ે" is never stripped ("છે"/is -> 6).
_SUFFIX_ANY = ["થી", "માં"]
_SUFFIX_UNIT = ["ની", "ના", "ને", "નું"]


CONNECT = {"ને", "અને", "और"}          # "and" between number words (બસો ને વીસ = 220)

def _gu_norm(s):
    """Collapse common Gujarati number-spelling variants so the MODEL's spellings match
    the dictionary: ત્ર->ત (ત્રેસઠ==તેસઠ=63) and doubled conjuncts (છવ્વીસ==છવીસ=26,
    પચ્ચીસ==પચીસ=25). Verified collision-free across all 0-99 + hundred words."""
    s = s.replace("ત્ર", "ત")
    return re.sub(r"(.)્\1", r"\1", s)

# normalized lookup, built once from the canonical words (collision-free: 109 -> 109)
GU_NORM = {}
for _w, _v in list(GU_UNITS.items()) + list(GU_HUND.items()):
    GU_NORM.setdefault(_gu_norm(_w), _v)

def _gu_value(core):
    """Gujarati number word -> value: unit (0-99), fused century, or fused NN-hundred.
    Exact match first, then a spelling-variant fallback (તેસઠ=63, છવીસ=26)."""
    if core in GU_UNITS: return GU_UNITS[core]
    if core in GU_HUND:  return GU_HUND[core]
    if core.endswith("સો"):                       # fused NN-hundred (ચુમાળીસસો=4400);
        s = core[:-len("સો")]                     # stem must be TWO-digit -> never બેસો
        sv = GU_UNITS.get(s)
        if sv is None:
            sv = GU_NORM.get(_gu_norm(s))         # variant stem (તેસઠસો)
        if sv is not None and 11 <= sv <= 99:
            return sv * 100
    nk = _gu_norm(core)                            # spelling-variant fallback
    if nk in GU_NORM:
        return GU_NORM[nk]
    return None

def _split_hundred_ne(tokens):
    """બસોને -> [બસો, ને] (only when the stem is a HUNDRED). Leaves પંદરને (15+ને) alone."""
    out = []
    for t in tokens:
        core = t.strip(_PUNCT)
        if core.endswith("ને") and len(core) > 2:
            v = _gu_value(core[:-2])
            if v is not None and v >= 100 and v % 100 == 0:
                out.append(core[:-2]); out.append("ને"); continue
        out.append(t)
    return out

def _classify(core):
    """Return (kind, value, lang, suffix) for a number token, or None. Exact match,
    then a safe fused-suffix strip (suffix is reattached to the digit)."""
    low = core.lower()
    if low in EN_UNITS:
        return ("unit", EN_UNITS[low], "en", "")
    if low in EN_SCALES:
        return ("scale", EN_SCALES[low], "en", "")
    v = _gu_value(core)                            # unit / fused century / NN-hundred
    if v is not None:
        return ("unit", v, "gu", "")
    if core in GU_SCALES:
        return ("scale", GU_SCALES[core], "gu", "")
    if _hin(core) in HI_UNITS:                     # Hindi Devanagari cardinals
        return ("unit", HI_UNITS[_hin(core)], "hi", "")
    if _hin(core) in HI_SCALES:
        return ("scale", HI_SCALES[_hin(core)], "hi", "")
    for suf in _SUFFIX_ANY:
        if core.endswith(suf) and len(core) > len(suf):
            stem = core[:-len(suf)]
            sv = _gu_value(stem)
            if sv is not None:
                return ("unit", sv, "gu", suf)
            if stem in GU_SCALES:
                return ("scale", GU_SCALES[stem], "gu", suf)
    for suf in _SUFFIX_UNIT:                      # units only (protects સોના/સોની/સોને)
        if core.endswith(suf) and len(core) > len(suf):
            sv = _gu_value(core[:-len(suf)])
            if sv is not None:
                return ("unit", sv, "gu", suf)
    return None


def _resolve_run(run):
    """run: list of (kind, value, lang, suffix). Return the digit string."""
    suffix = run[-1][3]
    kinds = [k for k, _, _, _ in run]
    vals = [v for _, v, _, _ in run]
    langs = [l for _, _, l, _ in run]

    if "scale" in kinds:                       # arithmetic: ચાર હજાર -> 4000, one hundred twenty five -> 125
        total = cur = 0
        for k, v, _, _ in run:
            if k == "scale":
                cur = (cur or 1) * v
                total += cur
                cur = 0
            else:
                cur += v
        return str(total + cur) + suffix

    if all(l == "en" for l in langs) and all(0 <= v <= 9 for v in vals) and len(run) >= 2:
        return "".join(str(v) for v in vals) + suffix   # phone-style: nine nine seven five -> 9975

    if any(v >= 100 for v in vals):            # 200 and 20 -> 220 (બસો ને વીસ)
        return str(sum(vals)) + suffix

    # tens + unit merge (twenty five -> 25); otherwise keep separate (ત્રણ ચાર -> 3 4)
    merged, skip = [], False
    for i, v in enumerate(vals):
        if skip:
            skip = False
            continue
        if v >= 20 and v % 10 == 0 and i + 1 < len(vals) and vals[i + 1] < 10:
            merged.append(v + vals[i + 1]); skip = True
        else:
            merged.append(v)
    return " ".join(str(x) for x in merged) + suffix


def _split_scale_digits(tokens):
    """हज़ार975 -> [हज़ार, 975]: the ASR learned Scribe's '6 હજાર' style and can glue
    the remainder onto the scale word."""
    out = []
    for t in tokens:
        m = re.fullmatch(r"(\D+?)(\d+\W*)", t)
        if m and (_classify(m.group(1)) or ("",))[0] == "scale":
            out += [m.group(1), m.group(2)]
        else:
            out.append(t)
    return out


def numbers_to_digits(text):
    """Convert runs of spoken number words (English + Gujarati + Hindi) to digits."""
    tokens = _split_hundred_ne(expand_repeat(text.split()))   # ડબલ પાંચ->55 ; બસોને->બસો ને
    tokens = _split_scale_digits(tokens)                      # हज़ार975 -> हज़ार 975
    out, run = [], []

    def flush():
        if run:
            out.append(_resolve_run(run))
            run.clear()

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        core = tok.strip(_PUNCT)
        trail = tok[len(tok.rstrip(_PUNCT)):]
        # "ને"/"અને" sitting between two numbers = "and" -> connector, keep the run going
        if core in CONNECT and run and i + 1 < len(tokens) and _classify(tokens[i + 1].strip(_PUNCT)):
            i += 1
            continue
        info = None if not core else _classify(core)
        # Digits join a run only next to a scale word, so "6 हज़ार 975" -> 6975 and
        # "10 હજાર" -> 10000, while plain digits ("3 4 દિવસ", "2024") stay as they are.
        if info is None and core.isdigit():
            nxt = _classify(tokens[i + 1].strip(_PUNCT)) if i + 1 < len(tokens) else None
            after_scale = run and run[-1][0] == "scale" and int(core) < run[-1][1]
            if (nxt and nxt[0] == "scale") or after_scale:
                info = ("unit", int(core), "digit", "")
        # એક = article "a/an" (એક minute, એક ગાડી) UNLESS the next token is itself a
        # number/scale word -> then it's the numeral 1 (એક લાખ -> 100000, એક હજાર -> 1000).
        if info and core == "એક" and not run:
            _nxt = tokens[i + 1].strip(_PUNCT) if i + 1 < len(tokens) else ""
            if not (_nxt and _classify(_nxt)):
                info = None
        if info:
            run.append(info)
            # a fused suffix (15થી) or trailing punctuation ends this number run,
            # so "પંદરથી વીસ" -> "15થી 20" rather than being merged into one number
            if info[3] or trail:
                out.append(_resolve_run(run) + trail); run.clear()
        else:
            flush()
            out.append(tok)
        i += 1
    flush()
    return " ".join(t for t in out if t != "")


# ============================================================================
# SCRIPT UNIFICATION: force a single Indic script (no Hindi/Gujarati mixture).
# Devanagari <-> Gujarati is a fixed code-point offset; English/digits untouched.
# ============================================================================

def _build_translit(src_lo, src_hi, offset):
    m = {}
    for cp in range(src_lo, src_hi):
        try:
            unicodedata.name(chr(cp + offset))     # only map to an ASSIGNED target char
            m[cp] = cp + offset
        except ValueError:
            pass
    return m

_DEV2GUJ = _build_translit(0x0900, 0x0980, +0x180)   # Devanagari -> Gujarati
_GUJ2DEV = _build_translit(0x0A80, 0x0B00, -0x180)   # Gujarati -> Devanagari
_INDIC_DIGITS = {}
for _i in range(10):
    _INDIC_DIGITS[0x0966 + _i] = ord("0") + _i       # Devanagari digit -> ASCII
    _INDIC_DIGITS[0x0AE6 + _i] = ord("0") + _i       # Gujarati digit  -> ASCII
_DEV_RANGE = range(0x0900, 0x0980)
_GUJ_RANGE = range(0x0A80, 0x0B00)

def unify_script(text, target):
    """Merge Devanagari/Gujarati into one script. English + digits untouched."""
    if not target:
        return text
    if target == "auto":
        dev = sum(ord(c) in _DEV_RANGE for c in text)
        guj = sum(ord(c) in _GUJ_RANGE for c in text)
        target = "hi" if dev > guj else "gu"
    if target == "gu":
        text = text.translate(_DEV2GUJ).replace("઼", "")   # drop nukta (rare in Gujarati)
    elif target == "hi":
        text = text.translate(_GUJ2DEV)
    return text.translate(_INDIC_DIGITS)


def postprocess(text, lex):
    """lexicon -> ordinals -> clock times -> number ITN -> unify script.
    Script unification runs LAST so Hindi numbers convert from their native
    Devanagari forms first, then any leftover Devanagari words fold to Gujarati."""
    text = apply_lexicon(text, lex)
    text = apply_ordinals(text)        # પહેલી -> 1લી
    text = apply_clock_times(text)     # સાડા દસ -> 10:30  (before cardinals eat 'દસ')
    text = numbers_to_digits(text)
    text = unify_script(text, TARGET_SCRIPT)   # force one Indic script on the residual words
    return re.sub(r"\s+", " ", text).strip()


# ============================================================================
# Diarization + ASR
# ============================================================================

def load_diarizer():
    if DIARIZER == "pyannote":
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=HF_TOKEN)
        pipeline.to(DEVICE)
        return pipeline
    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.restore_from(str(_ROOT / DIARIZER), map_location=DEVICE)
    model.eval()
    # Offline streaming (bigger chunks): fewer mid-call speaker flips than the
    # default settings on the 100-call transcript test (5 vs 6 swapped calls).
    modules = model.sortformer_modules
    modules.chunk_len, modules.chunk_right_context = 340, 40
    modules.fifo_len, modules.spkcache_update_period = 40, 300
    check = getattr(modules, "_check_streaming_parameters", None) or model._check_streaming_parameters
    check()
    return model


def diarize_audio(pipeline, wav_full, sr):
    print("Running diarization...")
    if not isinstance(pipeline, Pipeline):
        return diarize_sortformer(pipeline, wav_full, sr)
    audio_input = {"waveform": wav_full, "sample_rate": sr}
    with torch.inference_mode():
        diarization_output = pipeline(audio_input, num_speakers=2)
    segments = []
    for turn, speaker in diarization_output.speaker_diarization:
        speaker_label = str(speaker)
        if not speaker_label.startswith("SPEAKER"):
            speaker_label = f"SPEAKER_{speaker_label.zfill(2)}"
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker_label})
    return segments


def diarize_sortformer(model, wav_full, sr):
    """Sortformer reads audio from a file. Only its two main speakers are kept,
    the way pyannote is forced to two (num_speakers=2)."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        soundfile.write(wav_path, wav_full.squeeze(0).cpu().numpy(), sr)
        output = model.diarize(audio=[wav_path], batch_size=1)
    finally:
        os.remove(wav_path)
    return sortformer_segments(output)


def sortformer_segments(output):
    """NeMo diarize() output for one file ("start end speaker_N" lines) -> segments."""
    if isinstance(output, tuple):  # (segments, tensors) when tensor outputs are requested
        output = output[0]
    turns = []
    for item in output[0]:
        start, end, speaker = item.split() if isinstance(item, str) else item
        turns.append((float(start), float(end), f"SPEAKER_{str(speaker).rsplit('_', 1)[-1].zfill(2)}"))
    talk = {}
    for start, end, speaker in turns:
        talk[speaker] = talk.get(speaker, 0.0) + end - start
    kept = sorted(talk, key=talk.get, reverse=True)[:2]
    return [{"start": start, "end": end, "speaker": speaker}
            for start, end, speaker in sorted(turns) if speaker in kept]


def merge_adjacent_segments(segments, gap_threshold=MERGE_GAP_THRESHOLD, min_duration=MIN_SEGMENT_DURATION):
    if not segments:
        return []
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        last = merged[-1]
        if seg["speaker"] == last["speaker"] and (seg["start"] - last["end"]) <= gap_threshold:
            last["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    return [seg for seg in merged if (seg["end"] - seg["start"]) >= min_duration]


def split_long_segments(segments, max_duration=MAX_SEGMENT_DURATION):
    """Split any segment longer than max_duration so the ASR sees <=30s clips."""
    out = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        if dur <= max_duration:
            out.append(seg)
            continue
        n = int(dur // max_duration) + 1
        step = dur / n
        for k in range(n):
            out.append({**seg,
                        "start": seg["start"] + k * step,
                        "end": min(seg["start"] + (k + 1) * step, seg["end"])})
    return out


def _slice_wav(wav_full, sr, start, end, pad=SEGMENT_PAD):
    """v3 audio-input parity: slice [start-pad, end+pad], clamped to the clip.
    Matches prepare_data_v3 so the ASR gets the same edge context at inference
    as it saw in training. The diarization timestamps (start/end) are untouched -
    only the audio HANDED TO THE ASR is padded; the transcript keeps raw times."""
    n = wav_full.shape[1]
    a = max(0, int((start - pad) * sr))
    b = min(n, int((end + pad) * sr))
    return wav_full[:, a:b]


def transcribe_segment(model, wav_full, sr, start, end):
    """Transcribe one time-slice with the fine-tuned NeMo model."""
    segment_wav = _slice_wav(wav_full, sr, start, end)
    if segment_wav.shape[1] == 0:
        return ""
    audio_np = segment_wav.squeeze(0).cpu().numpy().astype("float32")
    with torch.inference_mode():
        hyps = model.transcribe([audio_np], batch_size=1, verbose=False)
    h = hyps[0]
    if isinstance(h, (list, tuple)):
        h = h[0]
    return getattr(h, "text", h).strip()


def build_lm_decoder(model):
    """Build a pyctcdecode beam decoder backed by the KenLM binary. Returns
    (decoder, blank_index) or None if disabled/unavailable."""
    if not USE_LM:
        return None
    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError:
        print("[warn] pyctcdecode not installed - falling back to greedy decoding")
        return None
    if not os.path.isfile(LM_PATH):
        return None
    vocab = list(model.decoder.vocabulary)
    labels = vocab + [""]          # NeMo CTC blank is the last class
    unigrams = None
    if os.path.isfile(LM_CORPUS):
        unigrams = sorted({w for line in open(LM_CORPUS, encoding="utf-8") for w in line.split()})
    dec = build_ctcdecoder(labels, kenlm_model_path=LM_PATH, unigrams=unigrams,
                           alpha=LM_ALPHA, beta=LM_BETA)
    print(f"LM beam decoding ON (alpha={LM_ALPHA}, beta={LM_BETA}, unigrams={len(unigrams) if unigrams else 0})")
    return dec, len(vocab)


def _greedy_from_logprobs(model, lp, blank):
    """CTC-collapse argmax of a [T, classes] log-prob matrix -> detokenized text."""
    out, prev = [], -1
    for i in lp.argmax(-1):
        i = int(i)
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return model.tokenizer.ids_to_text(out)


def _number_ratio(text):
    """Fraction of tokens that are numbers (digit or spoken number word, Eng/Guj).
    A word-LM corrupts number sequences, so number-DENSE segments stay greedy;
    lightly-numbered wordy segments still go to the LM for garble cleanup."""
    toks = text.split()
    if not toks:
        return 0.0
    numc = 0
    for t in toks:
        core = t.strip(_PUNCT)
        if (any(c.isdigit() for c in core) or core.lower() in EN_UNITS
                or core.lower() in EN_SCALES or core in GU_UNITS or core in GU_SCALES):
            numc += 1
    return numc / len(toks)


def transcribe_with_lm(model, decoder_blank, wav_full, sr, segments):
    """HYBRID per-segment decode by number density: number-DENSE segments
    (>=NUM_RATIO_GREEDY of tokens are numbers -> phone/date/km/amount lists) use
    greedy to protect the digits; everything else uses LM beam to clean garbled
    real words. Returns {seg_idx: text}."""
    dec, blank = decoder_blank
    out = {}
    n = len(segments)
    n_lm = n_num = 0
    for i, seg in enumerate(segments):
        w = _slice_wav(wav_full, sr, seg["start"], seg["end"])
        if w.shape[1] == 0:
            continue
        sig = w.to(DEVICE)
        slen = torch.tensor([w.shape[1]], device=DEVICE)
        with torch.inference_mode():
            lp = model.forward(input_signal=sig, input_signal_length=slen)[0][0].float().cpu().numpy()
        greedy_txt = _greedy_from_logprobs(model, lp, blank)
        if _number_ratio(greedy_txt) >= NUM_RATIO_GREEDY:
            out[i] = greedy_txt; n_num += 1          # number-dense -> trust acoustics, not the LM
        else:
            out[i] = dec.decode(lp); n_lm += 1        # wordy -> let the LM fix garble
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"  decoded {i+1}/{n}  (LM:{n_lm}  greedy-numdense:{n_num})", flush=True)
    return out


def print_raw_segments(transcript_segments):
    print("\n" + "-" * 50)
    print("RAW SEGMENTS (post-processed text, raw SPEAKER_xx tags)")
    print("-" * 50)
    for i, seg in enumerate(transcript_segments):
        print(f"{i}: {seg['speaker']} [{seg['start']:.1f}s-{seg['end']:.1f}s]: {seg['text']}")
    print("-" * 50 + "\n")


# ============================================================================
# Role labeling (unchanged from the original pipeline)
# ============================================================================

def vote_dealership_agent(transcript_segments):
    """Weighted per-speaker vote over ALL dealership-name mentions (replaces the
    old naive first-hit anchor, which flipped roles whenever a customer said the
    name before the agent did). +1 per mention; +1 EXTRA when that segment also
    carries an introduction marker (a real "...Automobiles / ...માંથી" self-intro).
    The agent both mentions the name more and introduces it, so the weighted
    winner is the agent even if a customer echoes "Kataria?" once.

    Returns (agent_speaker, scores) on a clear winner, or (None, scores) when
    there are no mentions or the top score is tied (caller falls back to Qwen)."""
    scores = {}
    for seg in transcript_segments:
        low = seg["text"].lower()
        if not any(v.lower() in low for v in DEALERSHIP_NAME_VARIANTS):
            continue
        w = 2.0 if any(m.lower() in low for m in INTRODUCTION_MARKERS) else 1.0
        scores[seg["speaker"]] = scores.get(seg["speaker"], 0.0) + w
    if not scores:
        return None, scores
    top = max(scores.values())
    winners = [s for s, sc in scores.items() if sc == top]
    return (winners[0] if len(winners) == 1 else None), scores


def find_opening_speaker_anchor(transcript_segments):
    if not transcript_segments:
        return None
    return 0, transcript_segments[0]["speaker"]


def _build_role_prompt(full_text, ordered_labels, dealership_hint, opening_anchor):
    label_a, label_b = ordered_labels
    hint_parts = []
    if dealership_hint is not None:
        idx, speaker, variant = dealership_hint
        hint_parts.append(
            f'Hint: {speaker} said something close to the dealership\'s name ("{variant}") '
            f"at line {idx}. Whoever introduces the dealership is almost always Agent - "
            f"treat this as strong evidence that {speaker} = Agent.")
    if opening_anchor is not None:
        idx, speaker = opening_anchor
        hint_parts.append(
            f"Weaker hint: these are agent-initiated outbound calls, so {speaker} "
            f"(who speaks first, at line {idx}) is more likely Agent. Treat this as "
            f"soft evidence only - if the overall conversation content clearly says "
            f"otherwise, trust the content instead.")
    hint_text = ("\n" + "\n\n".join(hint_parts) + "\n") if hint_parts else ""
    return f"""This is a dealership service call transcript between exactly two speakers,
tagged {label_a} and {label_b} throughout (each tag always refers to the SAME
person for the entire call - a real person does not switch roles mid-call).
There are exactly two roles:

"Agent" - the dealership representative who placed the call. Typically:
introduces themselves and/or the dealership by name; asks about the
customer's vehicle, service history, or contact/appointment details; offers
or proposes appointment slots and timing; states information such as
service cost, duration, or the service center's address; speaks in a more
formal, scripted, business-like register; drives the conversation forward
with the next question or next step.

"Customer" - the person who received the call. Typically: confirms their own
identity or vehicle details when asked; answers scheduling questions with
their availability; accepts, declines, or asks to reschedule; may ask
clarifying questions about cost or how long the service will take; tends to
give shorter, more reactive answers that respond to what the Agent just said
rather than introduce new topics.

Determine the role of {label_a} and of {label_b} using the conversation as a
whole - not any single line in isolation. Important: the order {label_a} and
{label_b} happen to be listed in THIS prompt is arbitrary and carries no
information about who is Agent - do not let listing order influence your
answer, base it only on what each speaker actually says.
{hint_text}
Transcript:
{full_text}

Respond with ONLY 2 lines, in this exact format:
{label_a}=ROLE
{label_b}=ROLE
where ROLE is either "Agent" or "Customer", and the two tags must get
different roles. Do not add any other text.
"""


def _query_qwen_for_roles(prompt, valid_labels):
    options = {"temperature": 0, "num_predict": 32, "num_ctx": 4096}
    try:
        response = ollama.generate(
            model=QWEN_MODEL, prompt=prompt, think=False, options=options,
        )
    except TypeError:
        response = ollama.generate(model=QWEN_MODEL, prompt=prompt, options=options)
    result_text = response["response"].strip()
    result_text = re.sub(r"<think>.*?</think>", "", result_text, flags=re.DOTALL).strip()
    role_by_speaker = {}
    for line in result_text.splitlines():
        line = line.strip()
        if "=" in line:
            speaker_str, role = line.split("=", 1)
            speaker_str = speaker_str.strip().upper()
            if speaker_str in valid_labels:
                role_by_speaker[speaker_str] = role.strip()
    return role_by_speaker, result_text


def label_speaker_roles_global(transcript_segments, dealership_hint=None, opening_anchor=None):
    speaker_labels = sorted({seg["speaker"] for seg in transcript_segments})
    full_text = "\n".join(f"{seg['speaker']}: {seg['text']}" for seg in transcript_segments)
    if len(speaker_labels) < 2:
        prompt = _build_role_prompt(full_text, speaker_labels * 2, dealership_hint, opening_anchor)
        result, _ = _query_qwen_for_roles(prompt, speaker_labels)
        return result
    prompt = _build_role_prompt(full_text, speaker_labels, dealership_hint, opening_anchor)
    result, _ = _query_qwen_for_roles(prompt, speaker_labels)
    if len(result) == len(speaker_labels) and set(result.values()) == {"Agent", "Customer"}:
        print(f"Qwen3 role classification: {result}\n")
        return result
    print("Warning: Qwen3's role answer was incomplete - using heuristic anchors instead.\n"
          f"  Result: {result or '(incomplete)'}\n")
    return {}


def resolve_speaker_roles(speaker_labels, role_by_speaker, dealership_hint, opening_anchor=None):
    resolved = {}
    if dealership_hint is not None:
        _, agent_speaker, _ = dealership_hint
        for spk in speaker_labels:
            resolved[spk] = "Agent" if spk == agent_speaker else "Customer"
    for spk in speaker_labels:
        if spk not in resolved and spk in role_by_speaker:
            resolved[spk] = role_by_speaker[spk]
    if opening_anchor is not None:
        _, agent_speaker = opening_anchor
        for spk in speaker_labels:
            if spk not in resolved:
                resolved[spk] = "Agent" if spk == agent_speaker else "Customer"
    for spk in speaker_labels:
        resolved.setdefault(spk, spk)
    return resolved


def save_transcript(output_lines, audio_file, save_path=None, duration_sec=None):
    if not save_path:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(audio_file))[0]
        save_path = os.path.join(OUTPUT_DIR, base + ".transcript.json")
    elif str(save_path).lower().endswith(".txt"):
        save_path = os.path.splitext(str(save_path))[0] + ".transcript.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "format": "kataria-transcript-v2",
                "audio_duration_seconds": (
                    round(float(duration_sec), 3) if duration_sec is not None else None
                ),
                "turn_count": len(output_lines),
                # One entry per turn so the saved file reads like the conversation.
                "turns": list(output_lines),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nTranscript saved to: {save_path}")


AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac")


def _natural_key(name):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def default_transcript_path(audio_file):
    base = os.path.splitext(os.path.basename(audio_file))[0]
    return os.path.join(OUTPUT_DIR, base + ".transcript.json")


def process_file(
    pipeline, conformer_model, lex, audio_file, save_path=None,
    lm_decoder=None, progress=None,
):
    def _progress(percent, message):
        if progress:
            progress(f"[{percent}%] {message}")

    _progress(3, "Loading audio")
    print(f"\nProcessing: {audio_file}\n")
    tim = {}  # per-step wall-clock for this file (aggregated in the batch summary)

    t0 = time.time()
    wav_np, sr = librosa.load(audio_file, sr=16000, mono=True)
    wav_full = torch.from_numpy(wav_np).unsqueeze(0)
    duration_sec = len(wav_np) / sr
    tim["Audio load"] = time.time() - t0
    print(f"[timing] Audio load: {tim['Audio load']:.1f}s  (clip length: {duration_sec/60:.1f} min)")

    _progress(12, "Detecting speakers")
    t0 = time.time()
    diarized_segments = diarize_audio(pipeline, wav_full, sr)
    tim["Diarization"] = time.time() - t0
    print(f"[timing] Diarization: {tim['Diarization']:.1f}s")

    diarized_segments = merge_adjacent_segments(diarized_segments)
    diarized_segments = split_long_segments(diarized_segments)
    print(f"Found {len(diarized_segments)} speaker segments after merging/splitting.\n")

    _progress(32, f"Transcribing {len(diarized_segments)} speech segments")
    t0 = time.time()
    if lm_decoder is not None:
        # Beam + KenLM: needs per-segment log-probs, so no batched .transcribe().
        print("Transcribing segments (beam + LM)...\n")
        text_by_idx = transcribe_with_lm(conformer_model, lm_decoder, wav_full, sr, diarized_segments)
    else:
        # Greedy: transcribe ALL segments in one batched call (NeMo .transcribe()
        # has large per-call setup cost, so batching is ~5-8x faster than per-segment).
        print("Transcribing segments (single batched greedy call)...\n")
        idx_audio, audios = [], []
        for i, seg in enumerate(diarized_segments):
            w = _slice_wav(wav_full, sr, seg["start"], seg["end"])
            if w.shape[1] == 0:
                continue
            idx_audio.append(i)
            audios.append(w.squeeze(0).cpu().numpy().astype("float32"))
        with torch.inference_mode():
            if audios:
                try:
                    hyps = conformer_model.transcribe(audios, batch_size=ASR_BATCH_SIZE, verbose=False)
                except torch.cuda.OutOfMemoryError:
                    fallback = min(ASR_BATCH_SIZE // 2, 8)
                    print(f"[WARN] CUDA OOM at batch_size={ASR_BATCH_SIZE}, retrying with {fallback}", flush=True)
                    torch.cuda.empty_cache()
                    hyps = conformer_model.transcribe(audios, batch_size=fallback, verbose=False)
            else:
                hyps = []

        def _tt(h):
            if isinstance(h, (list, tuple)):
                h = h[0]
            return getattr(h, "text", h)

        text_by_idx = {i: _tt(h) for i, h in zip(idx_audio, hyps)}

    transcript_segments = []
    for i, seg in enumerate(diarized_segments):
        text = postprocess((text_by_idx.get(i) or "").strip(), lex)   # lexicon + number ITN
        if text:
            transcript_segments.append({"speaker": seg["speaker"], "start": seg["start"],
                                        "end": seg["end"], "text": text})
    tim["Transcription"] = time.time() - t0
    print(f"[timing] Transcription+postproc: {tim['Transcription']:.1f}s")

    print_raw_segments(transcript_segments)

    _progress(78, "Determining Agent and Customer roles")
    t0 = time.time()
    print("Determining speaker roles...\n")
    speaker_labels = sorted({seg["speaker"] for seg in transcript_segments})
    agent_speaker, vote_scores = vote_dealership_agent(transcript_segments)
    opening_anchor = find_opening_speaker_anchor(transcript_segments)
    if vote_scores:
        tally = ", ".join(f"{s}={sc:g}" for s, sc in sorted(vote_scores.items()))
        print(f"Dealership-name vote: {tally}")
    # A clear weighted winner is authoritative -> assign directly and SKIP Qwen
    # (removes the ~87s ollama bottleneck). Qwen only runs when there is NO mention
    # or the vote ties, i.e. exactly the cases the name signal cannot decide.
    if agent_speaker is not None:
        print(f"Weighted vote -> {agent_speaker} = Agent (skipping Qwen).")
        role_by_speaker = {spk: ("Agent" if spk == agent_speaker else "Customer")
                           for spk in speaker_labels}
    else:
        why = "no dealership name detected" if not vote_scores else "vote tied"
        print(f"{why} - falling back to Qwen + opening-speaker anchor.\n")
        role_by_speaker_raw = label_speaker_roles_global(transcript_segments, None, opening_anchor)
        role_by_speaker = resolve_speaker_roles(speaker_labels, role_by_speaker_raw, None, opening_anchor)
    print("Final speaker -> role mapping:")
    for spk in speaker_labels:
        print(f"  {spk} -> {role_by_speaker[spk]}")
    tim["Role labeling"] = time.time() - t0
    print(f"[timing] Role labeling: {tim['Role labeling']:.1f}s")

    print("\n" + "=" * 50 + "\nTRANSCRIPT\n" + "=" * 50 + "\n")
    output_lines = []
    for seg in transcript_segments:
        role = role_by_speaker[seg["speaker"]]
        line = f"{role} [{seg['start']:.1f}s-{seg['end']:.1f}s]: {seg['text']}"
        print(line)
        output_lines.append(line)

    _progress(94, "Saving structured transcript")
    save_transcript(output_lines, audio_file, save_path, duration_sec=duration_sec)
    _progress(100, "Transcription complete")
    return duration_sec, tim


def gather_audio_files(folder):
    files = [os.path.join(folder, name) for name in os.listdir(folder)
             if name.lower().endswith(AUDIO_EXTS) and os.path.isfile(os.path.join(folder, name))]
    return sorted(files, key=lambda p: _natural_key(os.path.basename(p)))


def main():
    if len(sys.argv) < 2:
        print('Usage:')
        print('  Folder (batch):  python "better diarized cli batch v2.py" /path/to/folder')
        print('  Single file:     python "better diarized cli batch v2.py" /path/to/call.mp3 [out.txt]')
        return
    target = sys.argv[1]
    if os.path.isdir(target):
        audio_files = gather_audio_files(target)
        single_save_path = None
        if not audio_files:
            print(f"No audio files found in folder: {target}")
            return
        print(f"Batch mode: {len(audio_files)} recording(s) found in {target}")
    elif os.path.isfile(target):
        audio_files = [target]
        single_save_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        print(f"Error: not a file or folder -> {target}")
        return

    model_load = {}  # one-time load costs, reported in the batch summary
    print(f"Loading diarization model ({DIARIZER})...")
    t0 = time.time()
    pipeline = load_diarizer()
    model_load["Diarization model"] = time.time() - t0
    print(f"[timing] Diarization model load: {model_load['Diarization model']:.1f}s")

    print("Loading fine-tuned code-switch ASR...")
    t0 = time.time()
    conformer_model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH, map_location=DEVICE)
    conformer_model.eval()
    model_load["ASR model"] = time.time() - t0
    print(f"[timing] ASR load: {model_load['ASR model']:.1f}s")

    lm_decoder = build_lm_decoder(conformer_model)  # None if USE_LM False / unavailable
    lex = load_lexicon(LEXICON_PATH)

    batch_start = time.time()
    total = len(audio_files)
    done, skipped, failed = [], [], []
    total_audio_sec = 0.0
    per_file_runtimes = []
    agg_tim = {}  # step timings summed across all transcribed recordings

    for i, audio_file in enumerate(audio_files, start=1):
        print("\n" + "#" * 60 + f"\n# [{i}/{total}] {os.path.basename(audio_file)}\n" + "#" * 60)
        save_path = single_save_path
        if save_path is None:
            expected = default_transcript_path(audio_file)
            if os.path.isfile(expected):
                print(f"Skipping - transcript already exists: {expected}")
                skipped.append(audio_file)
                continue
        file_t0 = time.time()
        try:
            duration_sec, tim = process_file(pipeline, conformer_model, lex, audio_file, save_path, lm_decoder)
            for k, v in tim.items():
                agg_tim[k] = agg_tim.get(k, 0.0) + v
            file_elapsed = time.time() - file_t0
            total_audio_sec += duration_sec
            per_file_runtimes.append((os.path.basename(audio_file), file_elapsed))
            print(f"\n[{i}/{total}] Done in {file_elapsed:.1f}s ({duration_sec/60:.1f} min of audio)")
            done.append(audio_file)
        except KeyboardInterrupt:
            print("\nInterrupted by user - stopping the batch.")
            raise
        except Exception as e:
            print(f"\n[{i}/{total}] FAILED: {os.path.basename(audio_file)} -> {type(e).__name__}: {e}")
            failed.append((audio_file, f"{type(e).__name__}: {e}"))
            continue

    elapsed = int(time.time() - batch_start)
    print("\n" + "=" * 60 + "\nBATCH COMPLETE\n" + "=" * 60)
    print(f"Recordings found:   {total}")
    print(f"Transcribed:        {len(done)}")
    print(f"Skipped (existing): {len(skipped)}")
    print(f"Failed:             {len(failed)}")
    if failed:
        print("\nFailures:")
        for path, err in failed:
            print(f"  - {os.path.basename(path)}: {err}")
    if per_file_runtimes:
        print("\nRuntime per recording:")
        for name, secs in per_file_runtimes:
            print(f"  - {name}: {secs:.1f}s")

    # Per-stage timing breakdown, summed across all transcribed recordings.
    proc_total = sum(agg_tim.values())
    print("\nTIMING BREAKDOWN (summed across transcribed recordings):")
    for k in ("Audio load", "Diarization", "Transcription", "Role labeling"):
        if k in agg_tim:
            share = (100 * agg_tim[k] / proc_total) if proc_total else 0
            print(f"  {k:16s}: {agg_tim[k]:7.1f}s  ({share:4.0f}%)")
    print(f"  {'PER-FILE TOTAL':16s}: {proc_total:7.1f}s")
    print("  one-time model loads:")
    for k, v in model_load.items():
        print(f"    {k:14s}: {v:7.1f}s")

    total_audio_min = total_audio_sec / 60
    print(f"\nTotal audio transcribed: {total_audio_min:.1f} min across {len(done)} recording(s)")
    print(f"Total runtime: {elapsed // 3600}h {(elapsed % 3600) // 60}m {elapsed % 60}s ({elapsed}s)")
    if total_audio_sec > 0:
        print(f"Overall speed: {elapsed / total_audio_sec:.2f}x real-time")


ASR_BATCH_SIZE = int(os.environ.get("ASR_BATCH_SIZE", "16"))


class TranscriberEngine:
    """Persistent engine: load models once, process many files without re-loading."""

    def __init__(self):
        self.pipeline = None
        self.asr = None
        self.lex = None
        self.lm_decoder = None
        self._loaded = False
        self.load_time = 0.0

    def load(self, progress=None):
        if self._loaded:
            return
        t0 = time.time()
        _log = progress or print

        _log(f"Loading diarization model ({DIARIZER})...")
        self.pipeline = load_diarizer()

        _log("Loading ASR model...")
        self.asr = nemo_asr.models.ASRModel.restore_from(MODEL_PATH, map_location=DEVICE)
        self.asr.eval()

        self.lm_decoder = build_lm_decoder(self.asr)
        self.lex = load_lexicon(LEXICON_PATH)

        self._loaded = True
        self.load_time = round(time.time() - t0, 1)
        _log(f"Engine ready — model load took {self.load_time}s")

    def process_file(self, audio_file, save_path=None, progress=None):
        if not self._loaded:
            self.load(progress=progress)
        return process_file(
            self.pipeline, self.asr, self.lex, audio_file,
            save_path, self.lm_decoder, progress,
        )

    @property
    def loaded(self) -> bool:
        return self._loaded


if __name__ == "__main__":
    main()
    elapsed = int(time.time() - start_time)
    print(f"Total Runtime: {elapsed//60}:{elapsed%60:02d}")
