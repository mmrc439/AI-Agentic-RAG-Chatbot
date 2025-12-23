import re
import torch
from pathlib import Path
from src.configs.config import DEBUG


# ───────────────────────── LLM + pipeline config ─────────────────────────

LLM_PROVIDER = "OPENAI"    # or literally anything else, that would force MistralAI

# Layout probe (page bands, headings, etc.)
RUN_GPT_LAYOUT_PROBE = True

# Image understanding (tables/code/images from figures)
# This flag controls whether gpt_image_describe() is used in the pipeline.
# Set to False to skip GPT calls for images and rely only on captions/heuristics.
RUN_GPT_IMAGE_DESCRIBE = True

# Document-level summaries (if you wire them in)
RUN_GPT_DOC_SUMMARY = False

# OpenAI model used for layout probe, image describe, and (optionally) summaries
GPT_MODEL = "gpt-4o-mini"

# Storage / data dirs
STORE_DIR = "chroma_store"
DATA_DIR  = "Docs"    # where your files live

# Local LLM (for retrieval / answer synthesis if you use Mistral)
LLM_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


# ───────────────────────── Retrieval / ranking knobs ────────────────────────

KEEP_TOP_K         = 12       # how many to rerank
SIM_THRESHOLD      = 0.45     # judge gate
CITE_SIM_THRESHOLD = 0.3      # citations gate
SCOPED_MAX         = 100      # per-doc cap
GLOBAL_K           = 10       # global candidate cap feeding broaden pool
REQ_BULLET_LIMIT   = 10

# Chunking window
MAX_TOKENS = 512        # max tokens per chunk
OVERLAP    = 50         # overlap between chunks


# ───────────────────────── Requirements / QA helpers ────────────────────────

MODAL_RE = re.compile(r"\b(shall|must|should|may)\b", re.I)
STRONG_VERB = {
    "shall": "MANDATORY",
    "must": "MANDATORY",
    "should": "RECOMMENDED",
    "may": "GUIDANCE",
}

# clause-splitter regex  ← needs to be above extract_requirements
_SPLIT = re.compile(r";\s+| and | or ", re.I)

# remove numbering from questions after decomposition
_LEAD_NUM_RE = re.compile(r"^\s*\d+\s*[\.\)]\s*")

# Does the user explicitly ask for a section / clause?
_CLAUSE_PAT = re.compile(
    r"""
    (                               # 1️⃣ explicit ask
        \b(what|which|where)\s+(section|clause|subclause|part)\b
      | \bwhere\s+in\s+regdoc-\d+(?:\.\d+)*\b
      | \bwhere\s+does\s+regdoc-\d+(?:\.\d+)*\b
    )
  | (                               # 2️⃣ numeric reference
        \bsection\s+\d+(?:\.\d+)*\b
      | §\s*\d+(?:\.\d+)*\b
    )
    """,
    flags=re.I | re.X,
)


def normalize_question(q: str) -> str:
    """
    Remove leading enumeration tokens like '1.' '4)' etc.
    Trim whitespace. Preserve the user's intent to ask a question:
    if a '?' appears anywhere in the original, ensure the normalized
    string ends with '?'.
    """
    orig = q
    q = _LEAD_NUM_RE.sub("", q).strip()
    if "?" in orig and not q.endswith("?"):
        q = q.rstrip("?.") + "?"
    return q


# ───────────────────────── Chunking heuristics ──────────────────────────────

DEFAULT_NUMERIC_HEADING = r"^\d+(?:\.\d+)*\s+"

# Caption patterns used to detect figure/table labels like "Figure 3-1: ..."
CAPTION_PATTERNS = [
    r'^\s*(Figure|Fig\.|Table|Image|Exhibit)\s+\d+(?:\.\d+)*\s*[:\-–]\s*.+$',
    r'^\s*(Fig|Fig\.)\s*\d+\s*[:\-–]?\s*.+$',
    r'^\s*(Listing)\s+\d+([.:]\s+|\s+-\s+|\s+)',
]

# Some callers still import this; leave it for compatibility
CAPTION_RE_LIST = [re.compile(p, re.I) for p in CAPTION_PATTERNS]


# ───────────────────────── Debug helper ─────────────────────────────────────

DBG_STEP = {"n": 0}


def dbg(msg: str):
    if DEBUG:
        DBG_STEP["n"] += 1
        print(f"[{DBG_STEP['n']:02}] {msg}", flush=True)
