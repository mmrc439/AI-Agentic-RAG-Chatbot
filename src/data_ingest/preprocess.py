import re
import hashlib
import os
import json
import datetime
import unicodedata
import textwrap
import copy
import math
import string
import time
import io
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import fitz
from PIL import Image
import tiktoken

from src.configs.config import DEBUG  # kept for compatibility
from src.configs.helper import *  # dbg, DEFAULT_NUMERIC_HEADING, CAPTION_RE_LIST, CAPTION_PATTERNS, GPT_MODEL, RUN_GPT_* flags, etc.

# global tokenizer used by chunker + helpers
enc = tiktoken.get_encoding("cl100k_base")

# ---------------- Configurable heuristics (env / config) ----------------
# Language & sentence splitter
INGEST_LANG = os.getenv("INGEST_LANG", "en").lower()                 # e.g., "en", "fr", "de", "zh", "ja", "ko"
USE_SPACY = os.getenv("INGEST_USE_SPACY", "0") == "1"                # set to "1" to try spaCy if installed
USE_NLTK = os.getenv("INGEST_USE_NLTK", "0") == "1"                  # set to "1" to try NLTK Punkt if installed

# Boilerplate hints & thresholds
_BOILERPLATE_HINTS = tuple(
    [h.strip().lower() for h in os.getenv(
        "INGEST_BOILERPLATE_HINTS",
        "all rights reserved,confidential,copyright ©,page "
    ).split(",") if h.strip()]
)
BOILERPLATE_MINLEN = int(os.getenv("INGEST_BOILERPLATE_MINLEN", "20"))

# Leaderless TOC gap threshold (px)
LEADERLESS_GAP_PX = float(os.getenv("INGEST_LEADERLESS_GAP_PX", "100"))

# Optional: sentence overlap tokens at joins (if your call-site passes overlap=0, this is moot)
DEFAULT_SENTENCE_JOIN_OVERLAP = int(os.getenv("INGEST_SENTENCE_JOIN_OVERLAP", "0"))

# ---------------- Optional language tokenizers (graceful fallback) ----------------
_spacy_nlp = None
if USE_SPACY:
    try:
        import spacy  # type: ignore
        if INGEST_LANG.startswith("en"):
            try:
                _spacy_nlp = spacy.load("en_core_web_sm")
            except Exception:
                _spacy_nlp = spacy.blank("en")
                _spacy_nlp.add_pipe("sentencizer")
        else:
            try:
                _spacy_nlp = spacy.blank(INGEST_LANG)
                _spacy_nlp.add_pipe("sentencizer")
            except Exception:
                _spacy_nlp = None
    except Exception:
        _spacy_nlp = None

_nltk_sentenize = None
if USE_NLTK and _spacy_nlp is None:
    try:
        from nltk.tokenize import sent_tokenize as _nltk_sentenize  # type: ignore
    except Exception:
        _nltk_sentenize = None

# ---------------- Chunking configuration ----------------
@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 512
    min_tokens: int = 64
    sentence_join_overlap: int = 32
    hard_max_tokens: int = 1024   # safety upper bound


DEFAULT_CHUNKING_CONFIG = ChunkingConfig(
    max_tokens=int(os.getenv("INGEST_MAX_TOKENS", "512")),
    min_tokens=int(os.getenv("INGEST_MIN_TOKENS", "64")),
    sentence_join_overlap=DEFAULT_SENTENCE_JOIN_OVERLAP or 32,
    hard_max_tokens=int(os.getenv("INGEST_HARD_MAX_TOKENS", "1024")),
)

# ---------------- NEW: safe encode/decode wrappers (keep using your enc) ----------------
def _encode(s: str):
    try:
        return enc.encode(s)
    except Exception:
        return s.split()


def _decode(toks):
    try:
        return enc.decode(toks)
    except Exception:
        return " ".join(toks)

# ───────── OpenAI client ─────────
def _init_openai_client():
    key = None
    try:
        from google.colab import userdata  # optional
        key = userdata.get("openai")
    except Exception:
        pass

    if not key:
        key = os.environ.get("OPENAI_API_KEY")

    if not key:
        dbg("[DBG] OPENAI_API_KEY not set; GPT features disabled")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        dbg("[DBG] OpenAI client initialized for GPT features")
        return client
    except Exception as e:
        print("[WARN] OpenAI client not initialized; GPT features disabled:", e, flush=True)
        return None


client = _init_openai_client()


def _openai_chat_completion(
    messages: List[dict],
    model: str,
    temperature: float = 0.0,
    max_retries: int = 2,
):
    """
    Centralized wrapper for OpenAI chat completions.
    Handles both `client.chat_completions` and `client.chat.completions`
    across SDK versions, with simple retry/backoff.
    """
    if not client:
        return None
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            if hasattr(client, "chat_completions"):
                return client.chat_completions.create(
                    model=model, messages=messages, temperature=temperature
                )
            else:
                return client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature
                )
        except Exception as e:
            last_err = e
            dbg(f"OpenAI chat completion failed (attempt {attempt+1}/{max_retries+1}): {e}")
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
    print("[ERROR] OpenAI chat completion failed after retries:", last_err, flush=True)
    return None

# ───────── Utils ─────────
def now_utc_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(*vals) -> str:
    h = hashlib.sha256()
    for v in vals:
        if isinstance(v, (dict, list, tuple)):
            h.update(json.dumps(v, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        else:
            h.update(str(v).encode("utf-8"))
    return h.hexdigest()

# ---------------- IMPROVED: richer text normalization ----------------
def clean_text(txt: str) -> str:
    """
    Normalize unicode, remove soft hyphens, repair hyphenated line-breaks,
    collapse excessive whitespace and newlines, strip zero-width spaces.
    """
    if not txt:
        return ""
    t = unicodedata.normalize("NFC", txt)
    t = t.replace("\u200b", "").replace("\u00AD", "")  # zero-width, soft hyphen
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)      # de-hyphenate across newlines
    t = re.sub(r"[ \t]+", " ", t)                     # collapse spaces
    t = re.sub(r"\n{3,}", "\n\n", t)                  # cap blank-line runs
    return t.strip()

# ---------------- Layout Awareness: reading-order extraction ----------------
def _blocks_text_from_dict_block(block: dict) -> str:
    lines = block.get("lines", []) or []
    parts: List[str] = []
    for ln in lines:
        spans = ln.get("spans", []) or []
        segs = [sp.get("text", "") for sp in spans if sp.get("text")]
        if segs:
            parts.append("".join(segs))
    return "\n".join([p for p in parts if p.strip()])


def _cluster_columns(xs: List[float], max_cols: int = 3) -> List[float]:
    if not xs:
        return []
    xs_sorted = sorted(xs)
    k = min(max_cols, max(1, len(xs_sorted)))
    centers = []
    for i in range(k):
        q_idx = int((i + 0.5) * len(xs_sorted) / max(k, 1))
        q_idx = min(len(xs_sorted) - 1, max(0, q_idx))
        centers.append(xs_sorted[q_idx])
    return sorted(set(centers))


def _page_text_reading_order(pg: fitz.Page) -> str:
    """
    Best-effort reading order using text blocks, with simple N-column heuristic.
    Handles 1–3 columns by clustering x-centers.
    """
    try:
        data = pg.get_text("dict")
        blocks = data.get("blocks", []) if isinstance(data, dict) else []
        text_blocks = []
        for b in blocks:
            if b.get("type", 0) != 0:
                continue
            bbox = b.get("bbox") or [0, 0, 0, 0]
            x0, y0, x1, y1 = bbox
            xc = (x0 + x1) / 2.0
            txt = _blocks_text_from_dict_block(b)
            if txt.strip():
                text_blocks.append((x0, y0, x1, y1, xc, txt))

        if not text_blocks:
            return pg.get_text("text") or ""

        xs = [b[4] for b in text_blocks]
        centers = _cluster_columns(xs, max_cols=3)

        if not centers or len(centers) == 1:
            ordered = sorted(text_blocks, key=lambda t: (t[1], t[0]))
        else:
            def col_idx(xc: float) -> int:
                return min(range(len(centers)), key=lambda i: abs(centers[i] - xc))
            ordered = sorted(text_blocks, key=lambda t: (col_idx(t[4]), t[1], t[0]))

        joined = []
        for _, _, _, _, _, t in ordered:
            jt = clean_text(t)
            if jt:
                joined.append(jt)
        return "\n\n".join(joined)
    except Exception as e:
        dbg(f"[DBG] reading-order failed; fallback to pg.get_text('text'): {e}")
        return pg.get_text("text") or ""


def _page_lines_reading_order(pg: fitz.Page) -> List[str]:
    txt = _page_text_reading_order(pg)
    return [ln for ln in (txt.splitlines() if txt else [])]

# ---------------- Language-aware sentence tokenization ----------------
_CJK_PUNCT = r"[。！？｡！？]"


def _sentence_tokenize(text: str) -> List[str]:
    """
    Language-aware sentence segmentation with graceful fallbacks.
    - spaCy (if enabled/available)
    - NLTK Punkt (if enabled/available)
    - CJK punctuation split
    - Regex fallback
    """
    t = text or ""
    if not t.strip():
        return []

    if INGEST_LANG in ("zh", "ja", "ko", "zh-cn", "zh-tw"):
        parts = re.split(f"({_CJK_PUNCT})", t)
        out: List[str] = []
        buf = ""
        for p in parts:
            if not p:
                continue
            if re.match(_CJK_PUNCT, p):
                buf += p
                if buf.strip():
                    out.append(buf.strip())
                buf = ""
            else:
                buf += p
        if buf.strip():
            out.append(buf.strip())
        return [s for s in out if s]

    if _spacy_nlp is not None:
        try:
            doc = _spacy_nlp(t)
            return [s.text.strip() for s in doc.sents if s.text.strip()]
        except Exception:
            pass

    if _nltk_sentenize is not None:
        try:
            return [s.strip() for s in _nltk_sentenize(t) if s.strip()]
        except Exception:
            pass

    SENT_RE = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z0-9])")
    parts = SENT_RE.split(t)
    return [p.strip() for p in parts if p and p.strip()]

# ---------------- NEW: helpers used by splitter quality gates ----------------
def _is_caption_like(s: str) -> bool:
    st = (s or "").strip()
    if not st:
        return False
    try:
        for cre in CAPTION_RE_LIST:
            if cre.match(st):
                return True
    except Exception:
        pass
    return False


def _is_boilerplate(s: str) -> bool:
    st = (s or "").strip()
    if not st:
        return True
    if _is_caption_like(st):
        return False
    low = st.lower()
    if len(st) < BOILERPLATE_MINLEN:
        return True
    return any(k in low for k in _BOILERPLATE_HINTS)


def _is_boilerplate_with_pos(
    line: str,
    y_center: Optional[float],
    header_band: Optional[Tuple[float, float]],
    footer_band: Optional[Tuple[float, float]],
) -> bool:
    st = (line or "").strip()
    if not st:
        return True
    if _is_caption_like(st) or structurally_heading_like(st):
        return False

    if y_center is not None and header_band and footer_band:
        if header_band[0] <= y_center <= header_band[1]:
            return True
        if footer_band[0] <= y_center <= footer_band[1]:
            return True

    return _is_boilerplate(st)


def _tok_len(s: str) -> int:
    return len(_encode(s))

# ───────── Semantic-first splitting (paragraph → sentence → tokens) ─────────
def split_text_tokens(
    s: str,
    max_tokens: int,
    overlap: int,
    cfg: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
) -> List[str]:
    """
    1) Prefer paragraph boundaries (blank lines)
    2) Then language-aware sentence boundaries
    3) Finally token windows with overlap
    Drops boilerplate; keeps short headings/captions; uses configurable token bounds.
    """
    if not (s or "").strip():
        return []

    s = clean_text(s)
    max_tokens = max(16, min(max_tokens, cfg.hard_max_tokens))
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
    pieces: List[str] = []

    for para in paragraphs:
        sents = _sentence_tokenize(para) or [para]
        buf: List[str] = []
        buf_toks = 0
        join_overlap = overlap if overlap is not None else cfg.sentence_join_overlap

        for sent in sents:
            st = _tok_len(sent)
            if not buf:
                buf, buf_toks = [sent], st
                continue
            if buf_toks + st <= max_tokens:
                buf.append(sent)
                buf_toks += st
            else:
                if buf:
                    chunk = " ".join(buf).strip()
                    if chunk and (not _is_boilerplate(chunk)):
                        pieces.append(chunk)
                if join_overlap and buf:
                    last = buf[-1]
                    if _tok_len(last) <= join_overlap and (not _is_boilerplate(last)):
                        pieces.append(last)
                buf, buf_toks = [sent], st

        if buf:
            chunk = " ".join(buf).strip()
            if chunk and (not _is_boilerplate(chunk)):
                pieces.append(chunk)

    out: List[str] = []
    for piece in pieces:
        toks = _encode(piece)
        if len(toks) <= max_tokens:
            if piece:
                out.append(piece)
            continue
        step = max(1, max_tokens - max(0, overlap or 0))
        i = 0
        while i < len(toks):
            j = min(len(toks), i + max_tokens)
            window = _decode(toks[i:j]).strip()
            if window and not _is_boilerplate(window):
                out.append(window)
            if j == len(toks):
                break
            i += step

    deduped: List[str] = []
    last = None
    for p in out:
        if p != last:
            deduped.append(p)
        last = p

    final_chunks: List[str] = []
    for chunk in deduped:
        if _tok_len(chunk) < cfg.min_tokens and not (
            _is_caption_like(chunk) or structurally_heading_like(chunk)
        ):
            continue
        final_chunks.append(chunk)

    return final_chunks

# ───────── Heading-ish checks & TOC detection helpers ─────────
def structurally_heading_like(s: str) -> bool:
    if not s:
        return False
    words = s.split()
    if len(words) > 40:
        return False
    if len(s) > 220:
        return False
    return True


def _trailing_page_token(s: str) -> Optional[str]:
    s = s.strip()
    m = re.search(r"\s(\d+|[ivxlcdm]+)$", s, re.I)
    return m.group(1) if m else None

# TOC detectors
_TOC_DOTTED_RE = re.compile(r"\.{3,}\s*(\d+|[ivxlcdm]+)\s*$", re.I)
_TOC_ALL_DOTS_RE = re.compile(r"^\s*\.{3,}\s*$", re.I)


def is_toc_like_line(s: Optional[str]) -> bool:
    if not s:
        return False
    s2 = s.strip()
    if not s2:
        return False
    if _TOC_ALL_DOTS_RE.match(s2):
        return True
    if _TOC_DOTTED_RE.search(s2):
        return True
    return False


def is_leaderless_toc_line(s: str, spans: Optional[List[dict]] = None) -> bool:
    """
    Detect 'title     7' (large internal gap + trailing page number/roman).
    Applied only to pre-body/TOC pages by caller.
    """
    st = s or ""
    if not st.strip():
        return False
    page_tok = _trailing_page_token(st)
    if not page_tok:
        return False
    if re.search(r"\s{12,}\S+$", st) or "\t" in st:
        return True
    if spans and len(spans) >= 2:
        try:
            xs = [sp.get("bbox", [0, 0, 0, 0])[0] for sp in spans if sp.get("bbox")]
            xs = [x for x in xs if isinstance(x, (int, float))]
            if len(xs) >= 2 and (xs[-1] - xs[-2]) > LEADERLESS_GAP_PX:
                return True
        except Exception:
            pass
    return False

# ───────── Fonts (optional gating used in Part 2) ─────────
def cluster_font_sizes(pdf: fitz.Document) -> Dict[str, float]:
    sizes = []
    for i in range(min(40, pdf.page_count)):
        for blk in pdf.load_page(i).get_text("dict").get("blocks", []):
            if blk.get("type") != 0:
                continue
            for ln in blk.get("lines", []):
                for sp in ln.get("spans", []):
                    if (sp.get("text") or "").strip():
                        sizes.append(float(sp.get("size", 12.0)))
    if not sizes:
        return {"median": 12.0, "p80": 14.0}
    sizes.sort()
    median = sizes[len(sizes) // 2]
    p80 = sizes[min(len(sizes) - 1, int(len(sizes) * 0.8))]
    return {"median": median, "p80": p80}


def heading_threshold(font_stats: Dict[str, float], hint_ratio: Optional[float]) -> float:
    ratio = max(1.3, float(hint_ratio) if hint_ratio else 1.4)
    return max(font_stats["median"] * ratio, font_stats["p80"])

# ───────── TOC discovery (anchor/probe only) ─────────
def find_toc_pages(pdf: fitz.Document) -> Tuple[List[int], dict]:
    toc_pages, meta = [], {"source": "none"}
    try:
        toc = pdf.get_toc()
        if toc:
            first_page = min(entry[2] - 1 for entry in toc if entry[2] > 0)
            toc_pages = list(range(first_page, min(first_page + 5, pdf.page_count)))
            meta["source"] = "outline"
            dbg(f"[DBG] TOC from outlines: pages {toc_pages}")
            return toc_pages, meta
    except Exception:
        pass

    for i in range(min(16, pdf.page_count)):
        pg = pdf.load_page(i)
        lines = _page_lines_reading_order(pg)
        if lines and re.match(r"^(table of contents|contents)\s*$", lines[0], re.I):
            toc_pages = list(range(i, min(i + 5, pdf.page_count)))
            meta["source"] = "heuristic:title"
            dbg(f"[DBG] TOC by title: pages {toc_pages}")
            return toc_pages, meta
        dotted = sum(
            1
            for ln in lines
            if re.search(r"\.{3,}\s*(\d+|[ivxlcdm]+)\s*$", (ln or "").strip(), re.I)
        )
        if dotted >= 4:
            toc_pages = list(range(i, min(i + 5, pdf.page_count)))
            meta["source"] = "heuristic:dots"
            dbg(f"[DBG] TOC from dotted leaders: pages {toc_pages}")
            return toc_pages, meta
    return toc_pages, meta

# ───────── TOC whitelist builders & pattern augmentation ─────────
def _normalize_heading_title(s: str) -> str:
    s = clean_text(s or "")
    s = re.sub(r"\s(\d+|[ivxlcdm]+)$", "", s, flags=re.I)
    s = re.sub(r"[\.…⋯·∙•]{3,}", " ", s)
    s = re.sub(r"^\d+(?:\.\d+)*\s+", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s.lower()


def _normalize_heading_number(s: str) -> Optional[str]:
    m = re.match(r"^(\d+(?:\.\d+)*)\s+", clean_text(s or ""))
    return m.group(1) if m else None


def build_toc_index_from_outline(pdf: fitz.Document) -> List[dict]:
    idx = []
    try:
        for lvl, title, page1 in (pdf.get_toc() or []):
            if page1 <= 0:
                continue
            num = _normalize_heading_number(title)
            title_norm = _normalize_heading_title(title)
            if not title_norm or len(title_norm) < 3:
                continue
            idx.append(
                {
                    "level": int(lvl),
                    "num": num,
                    "title_norm": title_norm,
                    "page_index0": int(page1) - 1,
                    "title": title,
                }
            )
    except Exception:
        pass

    toc_headers = {"index", "contents", "table of contents", "toc"}
    trim_idx = 0
    for i, e in enumerate(idx):
        if e["title_norm"] in toc_headers:
            trim_idx = i + 1
            break
    if trim_idx > 0:
        idx = idx[trim_idx:]

    seen, out = set(), []
    for e in idx:
        key = (e.get("num"), e.get("title_norm"))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def build_toc_index_from_pages(pdf: fitz.Document, toc_pages: List[int]) -> List[dict]:
    out = []
    for i in toc_pages:
        pg = pdf.load_page(i)
        for ln in _page_lines_reading_order(pg):
            t = clean_text(ln)
            title_norm = _normalize_heading_title(t)
            if not title_norm or len(title_norm) < 3:
                continue
            if is_toc_like_line(t) or is_leaderless_toc_line(t):
                out.append(
                    {
                        "level": None,
                        "num": _normalize_heading_number(t),
                        "title_norm": title_norm,
                        "page_index0": i,
                        "title": t,
                    }
                )
    seen, uniq = set(), []
    for e in out:
        key = (e.get("num"), e.get("title_norm"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return uniq


def make_toc_whitelist(pdf: fitz.Document, toc_pages: List[int]) -> dict:
    idx = build_toc_index_from_outline(pdf)
    if not idx:
        idx = build_toc_index_from_pages(pdf, toc_pages)
    nums = {e["num"] for e in idx if e.get("num")}
    titles = {e["title_norm"] for e in idx if e.get("title_norm")}
    max_depth = 1
    for n in nums:
        d = n.count(".") + 1
        if d > max_depth:
            max_depth = d
    unnumbered = bool(titles and not nums)
    return {
        "nums": nums,
        "titles": titles,
        "has_toc": bool(nums or titles),
        "max_depth": max_depth,
        "unnumbered": unnumbered,
        "idx": idx,
    }


def _augment_heading_patterns_to_toc_depth(
    compiled_by_level: Dict[int, List[re.Pattern]], toc_max_depth: int
) -> Dict[int, List[re.Pattern]]:
    out = {k: v[:] for k, v in compiled_by_level.items()}
    if 1 not in out or all(p.pattern != r"^\d+\.\s+" for p in out[1]):
        out.setdefault(1, []).append(re.compile(r"^\d+\.\s+"))
        out[1] = [p for p in out[1] if p.pattern != r"^\d+\s+"] if 1 in out else out[1]
    for lvl in range(2, max(2, toc_max_depth) + 1):
        pat = r"^\d+(?:\.\d+){" + str(lvl - 1) + r"}\s+"
        if lvl not in out or all(p.pattern != pat for p in out.get(lvl, [])):
            out.setdefault(lvl, []).append(re.compile(pat))
    return out

# ───────── Probe page selection (hierarchy-focused) ─────────
def first_body_index_from_labels(pdf: fitz.Document) -> Optional[int]:
    for i in range(pdf.page_count):
        try:
            lab = pdf.load_page(i).get_label()
            if lab and lab.strip() == "1":
                return i
        except Exception:
            pass
    return None


def find_page_with_multiheading_levels(
    pdf: fitz.Document, start: int, end: int, skip: set
) -> Optional[int]:
    end = min(end, pdf.page_count)
    for i in range(start, end):
        if i in skip:
            continue
        txt = "\n".join(_page_lines_reading_order(pdf.load_page(i)))
        levels = set()
        for line in txt.splitlines():
            m = re.match(r"^(\d+(?:\.\d+)*)\s+", clean_text(line))
            if m:
                levels.add(m.group(1).count(".") + 1)
        if len(levels) >= 2:
            return i
    return None


def densest_non_toc_body_page(pdf: fitz.Document, banned: set) -> int:
    best_idx, best = 0, -1
    for i in range(pdf.page_count):
        if i in banned:
            continue
        wc = len((_page_text_reading_order(pdf.load_page(i)) or "").split())
        if wc > best:
            best, best_idx = wc, i
    return best_idx


def select_optimal_probe_pages(
    pdf: fitz.Document, toc_pages: List[int], body_start: int
) -> List[int]:
    chosen: List[int] = []
    if toc_pages:
        chosen.append(toc_pages[0])
        dbg(f"[DBG] Probe: TOC page -> {toc_pages[0]}")
    early = find_page_with_multiheading_levels(
        pdf, body_start, body_start + 10, set(chosen)
    )
    if early is None:
        early = body_start
        dbg(f"[DBG] Probe: early fallback -> {early}")
    else:
        dbg(f"[DBG] Probe: early multi-level -> {early}")
    chosen.append(early)
    if len(chosen) < 3:
        extra = densest_non_toc_body_page(pdf, set(chosen))
        if extra not in chosen:
            chosen.append(extra)
            dbg(f"[DBG] Probe: densest -> {extra}")
    return chosen[:3]

# ───────── Bands & page numbering ─────────
def bands_for_page(pg: fitz.Page, rules: dict) -> tuple:
    W, H = pg.rect.width, pg.rect.height
    header_info = rules.get("header", {})
    footer_info = rules.get("footer", {})
    pad = 0.005 * H

    header_hint = header_info.get("y_band", [0.0, 0.06])
    footer_hint = footer_info.get("y_band", [0.94, 1.0])

    if not header_hint or len(header_hint) < 2:
        header_band = (0.0, 0.0)
    else:
        header_band = (
            max(0.0, header_hint[0] * H - pad),
            min(H, header_hint[1] * H + pad),
        )

    if not footer_hint or len(footer_hint) < 2:
        footer_band = (H, H)
    else:
        footer_band = (
            max(0.0, footer_hint[0] * H - pad),
            min(H, footer_hint[1] * H + pad),
        )

    cr = rules.get(
        "content_region", {"x0": 0.05, "y0": 0.08, "x1": 0.95, "y1": 0.95}
    )
    content_region = (cr["x0"] * W, cr["y0"] * H, cr["x1"] * W, cr["y1"] * H)
    return True, True, header_band, footer_band, content_region


def _horizontal_window_for_location(W: float, location: str) -> Tuple[float, float]:
    location = (location or "footer_center").lower()
    third = W / 3.0
    if "left" in location:
        return (0.0, third)
    if "right" in location:
        return (2 * third, W)
    return (third, 2 * third)


def extract_page_numbers_from_headers_footers(
    pg: fitz.Page, header_band: tuple, footer_band: tuple, page_numbering_info: dict
) -> Optional[str]:
    blocks = pg.get_text("dict").get("blocks", [])
    sys_ = (page_numbering_info or {}).get("system", "arabic").lower()
    location = (page_numbering_info or {}).get("location", "footer_center")

    if sys_ == "roman":
        pattern = r"^[ivxlcdm]+$"
    elif sys_ == "mixed":
        pattern = r"^(\d+|[ivxlcdm]+)$"
    else:
        pattern = r"^\d+$"

    W = pg.rect.width
    xw0, xw1 = _horizontal_window_for_location(W, location)

    def scan_yband(y0, y1):
        for blk in blocks:
            if blk.get("type") != 0:
                continue
            bbox = blk.get("bbox")
            if not bbox:
                continue
            ym = (bbox[1] + bbox[3]) / 2.0
            xm = (bbox[0] + bbox[2]) / 2.0
            if not (y0 <= ym <= y1):
                continue
            if not (xw0 <= xm <= xw1):
                continue
            txt = " ".join(
                sp.get("text", "")
                for ln in blk.get("lines", [])
                for sp in ln.get("spans", [])
            )
            candidate = clean_text(txt)
            if re.match(pattern, candidate, re.I):
                return candidate
            toks = candidate.split()
            if toks:
                last = toks[-1]
                if re.match(pattern, last, re.I) and len(toks) <= 3:
                    return last
        return None

    return scan_yband(*header_band) or scan_yband(*footer_band)


def autodetect_page_numbering(pdf: fitz.Document) -> Dict[str, Any]:
    try:
        W, H = pdf.load_page(0).rect.width, pdf.load_page(0).rect.height
    except Exception:
        return {"system": "arabic", "location": "unknown", "pattern_examples": []}

    header_band = (0.00 * H, 0.06 * H)
    footer_band = (0.94 * H, 1.00 * H)
    freq: Dict[str, int] = {}
    pages = min(pdf.page_count, 40)

    for i in range(pages):
        pg = pdf.load_page(i)
        for y0, y1 in (header_band, footer_band):
            blocks = pg.get_text("dict").get("blocks", [])
            for blk in blocks:
                if blk.get("type") != 0:
                    continue
                bbox = blk.get("bbox") or (0, 0, 0, 0)
                ym = (bbox[1] + bbox[3]) / 2.0
                if not (y0 <= ym <= y1):
                    continue
                txt = clean_text(
                    " ".join(
                        sp.get("text", "")
                        for ln in blk.get("lines", [])
                        for sp in ln.get("spans", [])
                    )
                )
                cand = txt.split()[-1] if txt else ""
                if re.match(r"^(\d+|[ivxlcdm]+)$", cand, re.I):
                    freq[cand.lower()] = freq.get(cand.lower(), 0) + 1

    total_hits = sum(freq.values())
    if total_hits == 0:
        dbg("[DBG] Page numbering autodetect: no consistent tokens in bands")
        return {"system": "arabic", "location": "unknown", "pattern_examples": []}

    romans = sum(v for k, v in freq.items() if re.match(r"^[ivxlcdm]+$", k))
    system = "roman" if romans > (total_hits - romans) else "arabic"
    top_samples = sorted(freq.items(), key=lambda kv: -kv[1])[:3]
    dbg(f"[DBG] Page numbering autodetect: system={system}, samples={top_samples}")
    return {
        "system": system,
        "location": "unknown",
        "pattern_examples": [k for k, _ in top_samples],
    }

# ───────── GPT layout probe ─────────
def render_pages_to_images(
    pdf: fitz.Document, idxs: List[int], dpi: int = 144
) -> List[Image.Image]:
    ims = []
    for i in idxs:
        if 0 <= i < pdf.page_count:
            try:
                pg = pdf.load_page(i)
                mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
                pix = pg.get_pixmap(matrix=mat, alpha=False)
                if pix.width > 0 and pix.height > 0:
                    ims.append(
                        Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    )
            except Exception as e:
                dbg(f"[DBG] Failed to render page {i}: {e}")
    if not ims:
        dbg("[DBG] No valid images rendered; returning empty list")
    return ims


def gpt_layout_probe(images: List[Image.Image]) -> dict:
    fallback = {
        "content_region": {"x0": 0.05, "y0": 0.08, "x1": 0.95, "y1": 0.95},
        "page_numbering": {
            "system": "arabic",
            "location": "footer_center",
            "pattern_examples": [],
        },
        "hierarchy_system": {"numbering_style": "numeric", "levels_found": []},
        "header": {"present": True, "y_band": [0.00, 0.06]},
        "footer": {"present": True, "y_band": [0.94, 1.00]},
        "heading_detection": {
            "patterns": [DEFAULT_NUMERIC_HEADING],
            "font_size_hint_ratio": 1.4,
        },
        "caption_patterns": CAPTION_PATTERNS,
    }
    if not client or not RUN_GPT_LAYOUT_PROBE or not images:
        dbg("[DBG] GPT probe: using fallback (no client/images/disabled)")
        return fallback
    try:
        data_urls = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_urls.append(f"data:image/png;base64,{b64}")
        if not data_urls:
            return fallback

        prompt = (
            "Return JSON describing page layout & heading regexes.\n"
            "Keys: content_region{x0,y0,x1,y1 in 0..1}, "
            "page_numbering{system:arabic|roman|mixed, location:header_*|footer_*}, "
            "hierarchy_system{numbering_style:numeric|mixed|none, levels_found:[{level,pattern}]}, "
            "header{present,y_band:[y0,y1]}, footer{present,y_band:[y0,y1]}, "
            "heading_detection{patterns:[...], font_size_hint_ratio}.\n"
            "Only JSON."
        )
        msg = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                + [
                    {"type": "image_url", "image_url": {"url": u}}
                    for u in data_urls
                ],
            }
        ]
        resp = _openai_chat_completion(
            messages=msg, model=GPT_MODEL, temperature=0.0
        )
        if not resp or not resp.choices:
            dbg("[DBG] GPT probe: empty response; using fallback")
            return fallback

        payload = json.loads(resp.choices[0].message.content)

        def fix(pattern: str) -> str:
            if not isinstance(pattern, str):
                return ""
            pattern = re.sub(r"\\+d", r"\\d", pattern)
            pattern = re.sub(r"\\+s", r"\\s", pattern)
            pattern = re.sub(r"\\+\.", r"\\.", pattern)
            pattern = re.sub(r"\\+\?", r"\\?", pattern)
            return pattern

        if "hierarchy_system" in payload:
            lvls = payload["hierarchy_system"].get("levels_found", [])
            for lvl in lvls:
                if "pattern" in lvl:
                    lvl["pattern"] = fix(lvl["pattern"])
        if "heading_detection" in payload:
            payload["heading_detection"]["patterns"] = [
                fix(p)
                for p in payload["heading_detection"].get("patterns", [])
            ]

        dbg(
            f"[DBG] GPT probe JSON (trunc): "
            f"{textwrap.shorten(json.dumps(payload), width=900, placeholder=' … ')}"
        )
        return {**fallback, **payload}
    except Exception as e:
        dbg(f"[DBG] GPT probe failed -> {e}; using fallback")
        return fallback

# ───────── Heading patterns & numeric vector helpers ─────────
def validate_heading_patterns(rules: dict) -> Dict[int, List[re.Pattern]]:
    hs = rules.get("hierarchy_system", {}) or {}
    numbering = (hs.get("numbering_style") or "numeric").lower()
    levels = hs.get("levels_found") or []
    compiled: Dict[int, List[re.Pattern]] = {}

    def accept(p: str) -> bool:
        if not p or not p.startswith("^"):
            return False
        if numbering == "numeric" and r"\d" not in p:
            return False
        return True

    by_level: Dict[int, List[str]] = {}
    for entry in levels:
        lvl = int(entry.get("level", 0))
        patt = str(entry.get("pattern", "")).strip()
        if accept(patt):
            by_level.setdefault(lvl, []).append(patt)

    if numbering == "numeric" and not by_level:
        by_level = {
            1: [DEFAULT_NUMERIC_HEADING],
            2: [r"^\d+\.\d+\s+"],
            3: [r"^\d+\.\d+\.\d+\s+"],
        }
        dbg("[DBG] Heading pattern validation: using numeric fallback set")

    for lvl in sorted(by_level.keys(), reverse=True):
        compiled[lvl] = [re.compile(p) for p in by_level[lvl]]

    if compiled:
        dbg("[DBG] Validated heading patterns:")
        for lvl in sorted(compiled.keys()):
            dbg(f"      L{lvl}: {[p.pattern for p in compiled[lvl]]}")
    else:
        dbg("[DBG] WARNING: No valid heading patterns; all text will be body")
    return compiled


def numeric_vector_from_heading(text: str) -> Optional[List[int]]:
    m = re.match(r"^(\d+(?:\.\d+)*)\s+", text)
    if not m:
        return None
    parts = [p for p in m.group(1).split(".") if p != ""]
    try:
        return [int(x) for x in parts]
    except Exception:
        return None


def effective_depth_from_vector(vec: List[int]) -> int:
    if not vec:
        return 1
    if len(vec) >= 1 and vec[-1] == 0:
        return max(1, len(vec) - 1)
    return len(vec)


PURE_NUM_LINE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*|[ivxlcdm]+)\s*([.):\-–—]*)\s*$", re.IGNORECASE
)


def is_pure_number_token(s: str) -> bool:
    return bool(PURE_NUM_LINE.match((s or "").strip()))

# ───────── Visual helpers: caption, crops, GPT visual extraction ─────────
def find_pdf_caption_near(
    pg: fitz.Page, bbox: Tuple[float, float, float, float], scan_px: float = 48.0
) -> Optional[str]:
    if not bbox:
        return None
    x0, y0, x1, y1 = bbox
    H = pg.rect.height
    regions = [
        fitz.Rect(x0, min(H, y1), x1, min(H, y1 + scan_px)),  # below
        fitz.Rect(x0, max(0.0, y0 - scan_px), x1, max(0.0, y0)),  # above
    ]
    for sub in regions:
        try:
            text = (pg.get_text("text", clip=sub) or "").splitlines()
            for ln in text:
                t = clean_text(ln)
                if not t:
                    continue
                for cre in CAPTION_RE_LIST:
                    if cre.match(t):
                        return t
        except Exception:
            continue
    return None


def render_bbox_image(
    pg: fitz.Page, bbox: Tuple[float, float, float, float], dpi: int = 192
) -> Optional[Image.Image]:
    try:
        r = fitz.Rect(*bbox)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = pg.get_pixmap(matrix=mat, alpha=False, clip=r)
        if pix.width <= 0 or pix.height <= 0:
            return None
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception:
        return None


def gpt_image_describe(img: Image.Image) -> dict:
    """
    Use GPT to analyze an extracted image (figure/table/chart/code).

    Returns a dict with backward-compatible keys:
      {
        "modality": "image" | "code" | "table" | "chart",
        "content": <string for code/image> OR {"columns":[...], "rows":[[...] ...]},
        "note": "<short description or key insight>",

        # NEW (for richer ingestion; safe to ignore downstream if unused):
        "title": "<optional title or caption-like text>",
        "facts": [ "<short fact 1>", "<short fact 2>", ... ],
        "table": { "columns": [...], "rows": [[...], ...] }
      }
    """
    fallback = {
        "modality": "image",
        "content": "",
        "note": "",
        "title": "",
        "facts": [],
        "table": {"columns": [], "rows": []},
    }

    # Early exit if GPT not available or disabled
    if not client or not RUN_GPT_IMAGE_DESCRIBE or img is None:
        dbg("[DBG] gpt_image_describe: using fallback (no client/disabled/img=None)")
        return fallback

    try:
        # Encode image into PNG base64 data URL
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        # Richer prompt with explicit schema, but still compatible with existing usage.
        prompt = (
            "You are a careful document analyst. You see an image extracted from a PDF.\n"
            "Identify what kind of image it is and extract as much structured, queryable information as possible.\n\n"
            "Return ONLY a single JSON object with this schema:\n"
            "{\n"
            '  \"modality\": \"chart\" | \"table\" | \"code\" | \"image\" | \"other\",\n'
            '  \"title\": string | null,\n'
            '  \"note\": string,\n'
            '  \"facts\": [string],\n'
            '  \"table\": {\n'
            '    \"columns\": [string],\n'
            '    \"rows\": [ [any, ...] ]\n'
            "  },\n"
            '  \"content\": string | object  // optional; for backward compatibility\n'
            "}\n\n"
            "Guidelines:\n"
            "- If there are axes, bars, lines, slices, or numeric scales → modality=\"chart\".\n"
            "- If it looks like a grid of rows/columns → modality=\"table\".\n"
            "- If it is mostly source code → modality=\"code\" and put the code (plain text, no backticks) in \"content\".\n"
            "- Otherwise use modality=\"image\" (or \"other\", which will be treated like \"image\").\n\n"
            "For charts and tables:\n"
            "- Extract all visible labels (e.g., country, region, month, category) and numeric values when readable.\n"
            "- Use the \"table\" object with \"columns\" and \"rows\" so that each bar/slice/row becomes one row in the table.\n"
            "- If a value is illegible, use null or the string \"unknown\" instead of skipping the row.\n"
            "- In \"note\", clearly state the key takeaway, including who/what is highest or lowest and its approximate value,\n"
            "  e.g. \"Chile has the highest booster dose rate at about 128 doses per 100 people\".\n"
            "- In \"facts\", add 3–10 short, self-contained sentences that mention both labels and numbers.\n"
            "  Example facts:\n"
            "  - \"Chile has the highest booster dose rate at about 128 doses per 100 people.\"\n"
            "  - \"China has administered about 3.5 billion total COVID-19 vaccine doses.\"\n"
            "  - \"North America accounts for about 23% of global cases.\"\n\n"
            "For non-numeric images (photos, conceptual diagrams, etc.):\n"
            "- You can leave \"table\" empty (columns=[] and rows=[]).\n"
            "- Still fill \"note\" with a 1–2 sentence description.\n"
            "- Use 2–5 short \"facts\" that describe the image in words.\n\n"
            "IMPORTANT:\n"
            "- Respond with JSON only. No markdown, no extra text, no explanations.\n"
        )

        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        resp = _openai_chat_completion(
            messages=msg,
            model=GPT_MODEL,
            temperature=0.0,
        )
        if not resp or not getattr(resp, "choices", None):
            dbg("[DBG] gpt_image_describe: empty response; using fallback")
            return fallback

        raw = resp.choices[0].message.content

        # Debug raw response (truncated)
        try:
            dbg(
                "[DBG] gpt_image_describe raw (trunc): "
                + textwrap.shorten(raw, width=400, placeholder=" … ")
            )
        except Exception:
            pass

        out = json.loads(raw)

        # Normalize modality
        mod = (out.get("modality") or "image").lower()
        if mod not in ("image", "code", "table", "chart"):
            # Treat unexpected labels (e.g., "other", "diagram") as plain image
            mod = "image"

        # Title
        title = (out.get("title") or "").strip()

        # Note
        note = (out.get("note") or "").strip()

        # Facts
        facts = out.get("facts") or []
        if not isinstance(facts, list):
            facts = [str(facts)] if facts else []
        facts = [str(f).strip() for f in facts if str(f).strip()]

        # Table
        table = out.get("table") or {}
        if not isinstance(table, dict):
            table = {"columns": [], "rows": []}
        cols = table.get("columns") or []
        rows = table.get("rows") or []
        if not isinstance(cols, list):
            cols = []
        if not isinstance(rows, list):
            rows = []
        table = {"columns": cols, "rows": rows}

        # Backward-compatible "content":
        # - For charts/tables: keep the table JSON in content.
        # - For code: prefer the code string in content.
        # - For images: use any string content if provided, else empty.
        content = out.get("content", "")

        if mod in ("table", "chart"):
            # If content isn't already a dict with columns/rows, use our normalized table.
            if not isinstance(content, dict) or "columns" not in content or "rows" not in content:
                content = table
        elif mod == "code":
            # Ensure string code
            if not isinstance(content, str):
                # Fallback: join facts/note as a single string if code wasn't in content
                maybe_code = "\n".join([note] + facts).strip()
                content = maybe_code
        else:  # image
            if not isinstance(content, str):
                content = ""

        return {
            "modality": mod,
            "content": content,
            "note": note,
            "title": title,
            "facts": facts,
            "table": table,
        }

    except Exception as e:
        dbg(f"[DBG] gpt_image_describe failed: {e}")
        return fallback

# ───────── GPT doc-level summary (map→reduce within token budget) ─────────
def _windowize_for_summary(pieces: List[str], char_window: int = 3200) -> List[str]:
    big = "\n\n".join(pieces)
    out = []
    i = 0
    while i < len(big):
        out.append(big[i : i + char_window])
        i += char_window
    return out


def _gpt_summarize(text: str, max_words: int = 220) -> str:
    if not client or not RUN_GPT_DOC_SUMMARY or not text.strip():
        return ""
    system = (
        "You are a technical summarizer. Write a precise, faithful, neutral summary "
        "of the document content for a retrieval system. Avoid marketing language and speculation."
    )
    user = (
        f"Summarize the following content in <= {max_words} words. "
        "Emphasize scope, key sections, definitions, requirements, tables/figures, and any caveats.\n\n"
        f"{text}"
    )
    resp = _openai_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=GPT_MODEL,
        temperature=0.0,
    )
    if not resp or not resp.choices:
        return ""
    return (resp.choices[0].message.content or "").strip()


def build_doc_summary_chunk_gpt(
    doc_name: str,
    file_hash: str,
    ingested_at: str,
    pieces: List[str],
) -> Optional[dict]:
    """
    Build a single doc-level summary chunk (modality='doc_summary'),
    putting the final summary text in the 'content' field.
    """
    if not pieces:
        return None

    windows = _windowize_for_summary(pieces, char_window=3200)
    window_summaries = []
    for w in windows:
        s = _gpt_summarize(w, max_words=220)
        if s:
            window_summaries.append(s)
    if not window_summaries:
        return None

    combined = "\n".join(window_summaries)
    final = _gpt_summarize(combined, max_words=260) or combined[:1600]

    cid = stable_hash("doc-summary", doc_name, final)
    return {
        "section": "Document Summary",
        "section_level": 0,
        "section_path": "Document / Summary",
        "section_type": "summary",
        "doc_name": doc_name,
        "doc_type": "pdf",
        "file_hash": file_hash,
        "page_index": "None",
        "page_label": "None",
        "page_type": "summary",
        "modality": "doc_summary",
        "content": final,
        "caption_text": "None",
        "bounding_box": "None",
        "ingested_at": ingested_at,
        "chunk_id": cid,
        "chunk_hash": cid,
    }
