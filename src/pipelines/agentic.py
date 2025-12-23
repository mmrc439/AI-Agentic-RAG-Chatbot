from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, TypedDict, Iterator
from collections import Counter, OrderedDict

import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import roman

from src.configs.helper import _SPLIT
from src.embeddings.index.chroma_db import setup_vectorstores
import torch
from sentence_transformers import SentenceTransformer, util

import hashlib, datetime, re, json, os, unicodedata, textwrap, copy, math, string, time, io, base64
from langchain_core.documents.base import Document
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import SearxSearchWrapper
from langgraph.graph import StateGraph
import torch, gc
from docx import Document as DocxDocument
from docx.oxml.ns import qn
import openai
from collections import defaultdict, OrderedDict
from rapidfuzz import process, fuzz

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
# from langchain import LLMChain, PromptTemplate
from rapidfuzz import process, fuzz

import numpy as np
from sentence_transformers.util import cos_sim
import tiktoken
from huggingface_hub import login
# from PIL import Image
from src.data_ingest.preprocess import *
from src.embeddings.index.chroma_db import setup_vectorstores
from src.configs.helper import *
from src.configs.helper import _CLAUSE_PAT, normalize_question
from src.configs.agentic_features import agentic_features_dict
from jinja2 import Environment, FileSystemLoader

import json

import requests
from functools import lru_cache

ALIAS2DOC: dict[str, list[str]] = defaultdict(list)

np.float_ = np.float64
# from chromadb import Settings

hf_token = os.getenv("HF_TOKEN")
searx_ip = os.getenv("SEARX_IP")
openai.api_key = os.getenv("OPENAI_API_KEY")

enc = tiktoken.get_encoding("cl100k_base")   # same as GPT-3.5/4

# os paths and json config for web search
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "prompts")

with open(os.path.join(BASE_DIR,"configs", "searx_config.json"), "r") as searx_config:
    searx_params = json.load(searx_config)



# ──────────── SETUP ─────────────────────────────────────────────────

nltk.download("punkt", quiet=True)
try:
    # Some environments expose this, some don't – fail silently if missing
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass

# ────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS + DEBUG UTIL
# ────────────────────────────────────────────────────────────────


# openai.api_key = userdata.get('openai')

if LLM_PROVIDER == "OPENAI":
    class OpenAIWrapper:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def invoke(self, prompt, **kwargs) -> str:
            resp = openai.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                temperature=kwargs.get("temperature", 0.2),
                max_tokens=kwargs.get("max_tokens", 512),
                top_p=kwargs.get("top_p", 0.9),
            )
            return resp.choices[0].message.content



vs, memory_vs, doc_retriever, all_docs, bm25_retriever, chunk_retriever = setup_vectorstores()





# ─── CHAT-MEMORY → PROMPT HELPER ────────────────────────────────────
def memory_context(query: str, k: int = 4, *, chat_id: str, user: str) -> str:
    """
    Return up to k semantically-closest past turns (for the same chat_id)
    as plain text blocks ready for prompt-injection.
    """
    if not on("memory"):
        return ""

    past_turns = recall_memory(query, k=k, chat_id=chat_id, user=user)   # ← pass chat_id
    dbg(f"[MEM-INJECT] {len(past_turns)} past turns into prompt")

    blocks = []
    for d in past_turns:                       # d is a langchain Document
        role = d.metadata.get("role", "UNK").upper()
        blocks.append(f"{role}: {d.page_content.strip()}")

    return "\n".join(blocks)

try:
    ALIAS2DOC
except NameError:
    ALIAS2DOC = defaultdict(list)

def _add_alias(key: str, doc_id: str):
    lst = ALIAS2DOC.setdefault(key, [])
    if doc_id not in lst:
        lst.append(doc_id)

def _aliases_from_meta(meta: dict) -> None:
    doc_id = meta["source"]
    base, _ = os.path.splitext(os.path.basename(doc_id))
    low = base.lower()

    _add_alias(low, doc_id)
    _add_alias(re.sub(r"[^\w]", "", low), doc_id)
    _add_alias(low.replace("-", "."), doc_id)
    _add_alias(low.replace(".", "-"), doc_id)

    num = ".".join(re.findall(r"\d+", low))
    if num:
        _add_alias(num, doc_id)
        _add_alias(num.replace(".", ""), doc_id)
        _add_alias(f"regdoc{num}", doc_id)
        _add_alias(f"regdoc{num.replace('.','')}", doc_id)

    _add_alias(" ".join(low.split("_")[:4]), doc_id)

    num = ".".join(re.findall(r"\d+", doc_id.lower()))   # e.g. "2.4.4"
    _add_alias(f"regdoc-{num}",   doc_id)                # "regdoc-2.4.4"
    _add_alias(f"regdoc.{num}",   doc_id)


metas = vs._collection.get(include=["metadatas"])["metadatas"]
for m in metas:
    _aliases_from_meta(m)

print(f"Alias table ready: {len(ALIAS2DOC)}")



BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "prompts")

env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

def render_prompt(template_name: str, **kwargs) -> str:
    """Render a Jinja template with variables."""
    template = env.get_template(template_name)
    return template.render(**kwargs)

# ─── CHAT-MEMORY HELPERS ────────────────────────────────────────────
def write_memory(
    role: str,
    text: str,
    doc_ids: list[str] | None = None,
    sources: list[str] | None = None,
    *,
    chat_id: str = "default_chat",
    user: str = "default",
    ts: float | None = None,
    extra_meta: dict = None
):
    meta = {
        "role":      role,
        "user":      user,
        "chat_id":   chat_id,
        "timestamp": ts or time.time(),
        "doc_ids":   json.dumps(doc_ids or []),
    }

    if sources is not None:
        meta["sources"] = json.dumps(sources)

    if extra_meta:
        for key, value in extra_meta.items():
            # JSON encode only if it's not a string
            if not isinstance(value, str):
                value = json.dumps(value)
            meta[key] = value

    memory_vs.add_texts([text], metadatas=[meta])

    dbg(
        f"[MEM-WRITE] role={role:<9} user={user:<10} "
        f"chat_id={chat_id:<12} → {memory_vs._collection.count()} total"
    )


def read_memory(
    *,
    user: str = "default",
    chat_id: str = "default_chat",
    limit: int = 100,
    with_text: bool = False,
):
    include = ["metadatas", "documents"] if with_text else ["metadatas"]

    # 🔎 DEBUG – show incoming filter
    dbg(f"[MEM-READ ] filter  user={user}  chat_id={chat_id}")

    data = memory_vs._collection.get(
        include=include,
        # ✅ IMPORTANT: filter by BOTH user and chat_id
        where={"$and": [{"user": user}, {"chat_id": chat_id}]},
    )

    rows = zip(data.get("documents", []), data["metadatas"]) \
           if with_text else [(None, m) for m in data["metadatas"]]

    rows = sorted(
        rows,
        key=lambda r: r[1].get("timestamp", 0),
        reverse=True,
    )[:limit]

    # 🔎 DEBUG – show how many you actually retrieved
    dbg(f"[MEM-READ ] returned {len(rows)} row(s)")

    out = []
    for txt, meta in rows:
        try:
            meta["doc_ids"] = json.loads(meta.get("doc_ids", "[]"))
        except (TypeError, json.JSONDecodeError):
            meta["doc_ids"] = []
        # new: decode citations if present
        try:
            meta["sources"] = json.loads(meta.get("sources", "[]"))
        except:
            meta["sources"] = []

        out.append({"text": txt, "meta": meta} if with_text else meta)

    return out

def delete_memory(chat_id: str, user: str | None = None):
    """Deletes memory entries for a given chat_id (and optionally user) from ChromaDB."""
    if not chat_id:
        return
    try:
        if user:
            where = {"$and": [{"chat_id": chat_id}, {"user": user}]}
        else:
            where = {"chat_id": chat_id}

        memory_vs._collection.delete(where=where)
        dbg(f"[MEM-DELETE] Purged history for chat_id={chat_id}, user={user or '*'}")
    except Exception as e:
        dbg(f"[MEM-DELETE] Error deleting history for chat_id={chat_id}: {e}")


# Optional: semantic recall of past turns ---------------------------
def recall_memory(
    query: str,
    k: int = 6,
    chat_id: str = "default_chat",
    user: str = "default"
):
    """
    Return the top-k most similar past turns for this (user, chat_id) pair.
    """
    retriever = memory_vs.as_retriever(
        search_kwargs={
            "k": k,
            "filter": {
                # ✅ IMPORTANT: same AND filter here too
                "$and": [
                    {"user": user},
                    {"chat_id": chat_id}
                ]
            }
        }
    )
    return retriever.invoke(query)


# 📦 Cell B — local LLM (Qwen-1.8 B) wrapped for LangChain
# --------------------------------------------------------
@lru_cache(maxsize=1)
def _load_llm() -> tuple[HuggingFacePipeline, AutoTokenizer]:
    if LLM_PROVIDER == "OPENAI":
        return OpenAIWrapper("gpt-4o-mini"), enc
    else:
        """Download & initialise ONCE, then cache."""
        tok = AutoTokenizer.from_pretrained(LLM_NAME, use_fast=False)
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

        mod = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            device_map="auto",
            torch_dtype="auto",          # add load_in_4bit=True for CPU basic
        )

        gen = pipeline(
            "text-generation",
            model=mod,
            tokenizer=tok,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
            repetition_penalty=1.1,
            return_full_text=False,
        )
        gc.collect()
        print("✓ Mistral-7B-Instruct initialised & cached")
        return HuggingFacePipeline(pipeline=gen), tok


def get_llm():
    llm, _ = _load_llm()
    return llm

def get_tokenizer():
    _, tok = _load_llm()
    return tok

# --- helper -----------------------------------------------------------
def chat_wrap(user_msg: str, system_msg: str = "You are a helpful response generation assistant.", assistant_tag: bool = True):
    llm = get_llm()
    if LLM_PROVIDER == "OPENAI":
        # return list of message dicts for ChatCompletion
        return [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]
    else:
        # old HF string‑prompt style
        tok = get_tokenizer()
        return tok.apply_chat_template(
            [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            tokenize=False,
            add_generation_prompt=assistant_tag
        )

def decompose(user_q: str, chat_ctx: str = "") -> List[str]:
    """
    Split user_q into the *minimum* list of stand‑alone questions.

    Enhancements:
      • Pronoun expansion: rewrite follow‑up segments that contain
        'it', 'they', 'those', 'them', 'that', 'the above', etc.,
        by substituting the most relevant referent drawn from the
        chat context (recent turns) OR from earlier lines in the
        current input if multi‑question.
      • Returns ≥1 question; each ends with '?'.
      • Never fabricates new sub‑questions.

    NOTE: Expansion happens in‑prompt; no separate rewrite step needed.
    """
    # guard strings for display
    chat_disp = chat_ctx if chat_ctx.strip() else "—"
    prompt = render_prompt("query_decompose.jinja", user_q=user_q, chat_disp=chat_disp)
    # print("Prompt being sent:\n", prompt)

    raw = _txt(
        get_llm().invoke(
            chat_wrap(prompt, assistant_tag=False),
            temperature=0
        )
    ).strip()

    # collect nonempty, stop at sentinel
    parts = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln == "===END===":
            break
        if not ln.endswith("?"):
            ln += "?"
        parts.append(ln)

    # safety: if over‑split short input, collapse
    if len(parts) > 4 and len(user_q.split()) < 12:
        q = user_q.strip()
        if not q.endswith("?"):
            q += "?"
        parts = [q]

    # ensure at least 1
    if not parts:
        q = user_q.strip()
        if not q.endswith("?"):
            q += "?"
        parts = [q]

    return parts

gc.collect(); torch.cuda.empty_cache()
_load_llm()


FEATURES = agentic_features_dict
print({k:v["desc"] for k,v in FEATURES.items()})
FEATURES_DEFAULT = copy.deepcopy(FEATURES)







ALL_DOCS = all_docs

minilm     = SentenceTransformer("BAAI/bge-base-en-v1.5")

def on(flag: str) -> bool:
    return FEATURES.get(flag, {}).get("enabled", 0) == 1

def add_step(state, step):
    log = state.setdefault("plan", [])
    if not log or log[-1] != step:      # de-dupe
        log.append(step)

def add_trace(state, step, ok, info=""):
    state.setdefault("trace", []).append(
        f"{step:<18} | {'OK' if ok else 'FAIL'} | {info}")

def _txt(resp):                       # works for str OR ChatMessage
    return resp if isinstance(resp, str) else getattr(resp, "content", str(resp))

def wants_clause_location(question: str) -> bool:
    """
    Return True when the user explicitly asks for the *location* of a section/
    clause/subclause *OR* when the query itself cites a numeric § reference
    ("section 3.5", "§ 2.1.4").
    """
    return bool(_CLAUSE_PAT.search(question))

REQ_INTENT_RE = re.compile(
    r"\b(requirements?|obligations?|shall|must|should|list(?:\s+the)?\s+requirements?)\b",
    re.I
)
def wants_requirements(q: str) -> bool:
    return bool(REQ_INTENT_RE.search(q))



# ── keep this so you still match RegDoc identifiers ───────────────
# keep this for RegDoc matching
# ──────────── PATTERN FOR LITERAL “REGDOC-X.Y.Z” ───────────────
DOC_PHRASE_RE = re.compile(r"regdoc[-_\s]*\d+(?:\.\d+)*", re.I)

@lru_cache(maxsize=1)
def _get_doc_keywords() -> dict[str, set[str]]:
    """
    Map each ingested doc_id → set of alpha‑words ≥4 letters.
    """
    metas = vs._collection.get(include=["metadatas"])["metadatas"]
    mapping: dict[str, set[str]] = {}
    for m in metas:
        doc_id = m["source"]
        kws = set(re.findall(r"[a-z]{4,}", doc_id.lower()))
        mapping[doc_id] = kws
    return mapping

def extract_doc_phrase(q: str) -> Optional[str]:
    """
    1) If the question literally names REGDOC‑X.Y.Z, grab that first.
    2) Otherwise try keyword‑overlap matching on filenames.
    3) Then fallback to quoted substrings.
    4) Then simple suffix patterns (v1, -2024, etc.).
    """
    low = q.lower()

    # 1️⃣ Explicit RegDoc reference wins outright
    m = DOC_PHRASE_RE.search(q)
    if m:
        val = m.group(0).lower()
        if DEBUG:
            print(f"[DEBUG] extract_doc_phrase → explicit RegDoc fallback: {val}")
        return val

    # 2️⃣ Keyword‑overlap matching on doc_id words
    tokens = set(re.findall(r"[a-z]{4,}", low))
    best_doc, best_score = None, 0.0
    for doc_id, kws in _get_doc_keywords().items():
        if not kws:
            continue
        common = tokens & kws
        score = len(common) / len(kws)
        if len(common) >= 2 and score > best_score:
            best_doc, best_score = doc_id, score

    if best_doc and best_score >= 0.3:
        if DEBUG:
            print(f"[DEBUG] extract_doc_phrase → keyword match: {best_doc} (score {best_score:.2f})")
        return best_doc

    # 3️⃣ Quoted string fallback
    m = re.search(r'"([^"]{5,50})"', q)
    if m:
        val = m.group(1).lower()
        if DEBUG:
            print(f"[DEBUG] extract_doc_phrase → quoted fallback: {val}")
        return val

    # 4️⃣ Simple suffix fallback (e.g., mydoc_v2, file-2024)
    m = re.search(r'\b\w+(?:_v?\d+|-\d{4})\b', q, flags=re.I)
    if m:
        val = m.group(0).lower()
        if DEBUG:
            print(f"[DEBUG] extract_doc_phrase → suffix fallback: {val}")
        return val

    if DEBUG:
        print("[DEBUG] extract_doc_phrase → no match")
    return None






# 1) Keep your original resolve_doc unchanged:
def resolve_doc(phrase: Optional[str]) -> Optional[str]:
    if not phrase: return None
    key = re.sub(r"[^\w]", "", phrase.lower())
    if key in ALIAS2DOC and len(ALIAS2DOC[key]) == 1:
        return ALIAS2DOC[key][0]
    best = process.extractOne(key, ALIAS2DOC.keys(), scorer=fuzz.ratio)
    if best:
        cand, score, _ = best
        uniq = list(dict.fromkeys(ALIAS2DOC[cand]))
        if score >= 80 and len(uniq) == 1:
            return uniq[0]
    return None



def _canon(text: str) -> str:
    return re.sub(r"[^\w]", "", text.lower())


# ── Clause-extraction helpers ─────────────────────────────────────
nltk.download("stopwords", quiet=True)
STOP = set(stopwords.words("english"))

HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+[A-Za-z].+", re.M)
DOT_NUM_RE = re.compile(r"\b[1-9]\d*(?:\.[0-9]+)+\b")

def _key_terms(text: str) -> set[str]:
    return {t for t in re.findall(r"\b[a-z]{4,}\b", text.lower()) if t not in STOP}

def _strip_headers(txt: str) -> str:
    return re.sub(r"^\s*Page \d+.*?$", "", txt, flags=re.M)

def _extract_candidate_clauses(
    chunk: Document,
    query: str
) -> List[Tuple[float, str, str]]:
    txt    = _strip_headers(chunk.page_content)
    dbg(f"_extract_candidate_clauses: scanning heading §{chunk.metadata['section']} “{chunk.metadata['title'][:40]}…” for query terms")
    doc_id = chunk.metadata["source"].lower()
    q_set  = _key_terms(query)
    hits   = []
    doc_num    = ".".join(re.findall(r"\d+", doc_id))
    is_doc_num = lambda sec: sec == doc_num
    has_near   = lambda lo, p, r=40: any(k in lo[max(0,p-r):p+r] for k in q_set)

    # numbered headings
    for m in HEADING_RE.finditer(txt):
        sec, heading = m.group(1), m.group(0).lower()
        if is_doc_num(sec) and not (q_set & _key_terms(heading)): continue
        overlap = len(q_set & _key_terms(heading))
        score   = 3.0 + 1.5 * overlap - (0.05 if is_doc_num(sec) else 0)
        hits.append((score, doc_id, sec))

    # dotted numbers in body
    if q_set:
        lo = txt.lower()
        for m in DOT_NUM_RE.finditer(lo):
            sec, pos = m.group(0), m.start()
            if is_doc_num(sec) and not has_near(lo, pos): continue
            dists = [abs(pos - p.start()) for k in q_set for p in re.finditer(re.escape(k), lo)]
            if not dists: continue
            score = 1.0 / (min(dists) + 20) - (0.05 if is_doc_num(sec) else 0)
            hits.append((score, doc_id, sec))
    dbg(f"   → returning {len(hits)} clause candidates for this heading")
    return hits


def flex_std_regex(raw: str) -> re.Pattern:
    tokens = re.split(r'[\s\-–—:\u2010-\u2014]+', raw.strip(), maxsplit=2)
    if len(tokens) < 2:
        return re.compile(re.escape(raw), re.I)
    std, num = map(re.escape, tokens[:2])
    sep = r'[\s\-\u2010-\u2014:]*'
    return re.compile(rf'\b{std}{sep}{num}\b', flags=re.I)


def detect_enumeration_question(q: str) -> Optional[Tuple[int, str]]:
    m = re.search(r"(?i)\b(?:what|list|name|enumerate)\s+(\d+)\s+(\w+)", q)
    if not m: return None
    return int(m.group(1)), m.group(2).lower()

def extract_enumeration_items(text: str, count: int) -> List[str]:
    lines, items = text.splitlines(), []
    for ln in lines:
        b = re.match(r"\s*[•\-]\s*(.+)", ln)
        n = re.match(r"\s*\d+\s*[\.\)]\s*(.+)", ln)
        if b: items.append(b.group(1).strip())
        elif n: items.append(n.group(1).strip())
        if len(items) == count: break
    return items

def load_section_text(doc_id: str, sec: str) -> str:
    return "\n".join(
        d.page_content
        for d in ALL_DOCS
        if d.metadata.get("source") == doc_id
        and d.metadata.get("section") == sec
    )

def is_enumeration_query(q: str) -> Optional[Tuple[int, str]]:
    enum = detect_enumeration_question(q)
    if enum: return enum
    m = re.search(r"\b(?:which|list)\s+(\d+)\s+(\w+)", q, flags=re.I)
    return (int(m.group(1)), m.group(2).lower()) if m else None

def is_section_query(q: str) -> bool:
    return bool(re.search(
        r"\bwhere\b.*\bsection\b|\bwhich section\b|\bwhat section\b|\bin section\b",
        q, flags=re.I
    ))

def heading_similarity(question: str, heading: str) -> float:
    if not hasattr(heading_similarity, "q_emb"):
        heading_similarity.q_emb = minilm.encode(
            question, normalize_embeddings=True, show_progress_bar=False
        )
    h_emb = minilm.encode([heading], normalize_embeddings=True, show_progress_bar=False)
    return float(util.cos_sim(heading_similarity.q_emb, h_emb)[0][0])

# ——— Requirement & limit sentence detector ————————————————

def _sim(a: str, b: str) -> float:
    """
    BGE‐base cosine similarity between two short texts.
    """
    return util.cos_sim(
        minilm.encode(a, normalize_embeddings=True, show_progress_bar=False),
        minilm.encode(b, normalize_embeddings=True, show_progress_bar=False)
    ).item()

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" .,:;")

def extract_requirements(
    docs,
    question: str,
    max_out: int = 8,
    sim_thresh: float = 0.15
):
    """
    Scan retrieved *chunk*‐documents on *body* pages and return requirements:
      { verb, strength, text, doc, sec }
    """
    out, seen = [], set()
    q_emb = minilm.encode(question, normalize_embeddings=True,
                          show_progress_bar=False)

    for d in docs:
        # —— metadata filters ——
        if d.metadata.get("chunk_type") != "chunk":
            continue
        if d.metadata.get("page_type") != "body":
            continue
        if d.metadata.get("word_count", 0) < 15:
            continue

        for sent in re.split(r"(?<=[.?!])\s+", d.page_content):
            words = sent.split()
            if not 7 <= len(words) <= 60:
                continue

            m = MODAL_RE.search(sent)
            if not m:
                continue
            verb = m.group(1).lower()

            # safe early‐position check
            try:
                if sent.lower().split().index(verb) > 11:
                    continue
            except ValueError:
                continue

            # relevance gate
            if _sim(question, sent) < sim_thresh:
                continue

            # split into concise fragments
            for frag in _SPLIT.split(sent):
                mod = MODAL_RE.search(frag)
                if not mod:
                    continue
                bullet = _clean(frag)
                low_bullet = bullet.lower()
                if low_bullet in seen:
                    continue
                seen.add(low_bullet)

                clause_match = re.search(r"\b\d+\.\d+(?:\.\d+)*\b", d.page_content)
                out.append({
                    "verb":     mod.group(1).lower(),
                    "strength": STRONG_VERB[mod.group(1).lower()],
                    "text":     bullet,
                    "doc":      d.metadata["source"],
                    "sec":      clause_match.group(0) if clause_match else "—"
                })
                if len(out) >= max_out:
                    return out
    return out


# ——— External standard citation detector ————————————————
STD_RE = re.compile(
    r"\b(?:IAEA|CSA|ISO|IEC|ANSI)\s+[A-Z]?[0-9][A-Za-z0-9\.\-]*(?:[:\-–]\d{2,4})?\b",
    re.I
)

def find_ext_refs(text: str) -> List[str]:
    return list(dict.fromkeys(STD_RE.findall(text)))[:10]

def _depth(sec: str) -> int:
    return sec.count(".")

print("n_synth reloaded", time.time())

# ────────────────────────────────────────────────────────────────
# NODE: n_retrieve — hybrid retrieval + metadata filters
# ────────────────────────────────────────────────────────────────

def n_retrieve(s):
    """
    Unified retrieval with optional broaden pass.
    Always retrieves on `question_for_retrieval` if present,
    but detects explicit mentions on the original question.
    """
    add_step(s, "retrieve_regdocs")

    orig_q      = s["question"]
    retrieval_q = s.get("question_for_retrieval", orig_q)
    q           = retrieval_q

    if DEBUG:
        print(f"[DEBUG] n_retrieve ENTER orig_q={orig_q!r} retrieval_q={retrieval_q!r} "
              f"broaden={s.get('broaden_needed')} last_target={s.get('last_target')}")

    # ── 1️⃣ BROADEN OVERRIDE (only if no inherited scope) ─────
    if s.get("broaden_needed") and not s.get("last_target"):
        s.pop("broaden_needed", None)
        glob_vec  = chunk_retriever.invoke(q) or []
        glob_bm25 = bm25_retriever.invoke(q) or []
        docs = [d for d in (glob_vec + glob_bm25)]
        print(f"Docs retrieved: {docs}")
        # dedupe + cap
        seen = set(); out = []
        for d in docs:
            key = (d.metadata["doc_name"], d.metadata["section"], d.metadata.get("chunk_id") or id(d))
            if key in seen: continue
            seen.add(key); out.append(d)
            if len(out) >= 100: break
        s["docs"] = out
        # used_sources
        used = []
        for d in out:
            src = d.metadata["doc_name"] or d.metadata.get("doc_name", "")
            if src not in used: used.append(src)
        s["used_sources"] = used
        # drop old retrieve trace
        if "trace" in s:
            s["trace"] = [ln for ln in s["trace"] if not ln.startswith("retrieve_regdocs")]
        add_trace(s, "retrieve_regdocs", bool(out), f"broaden-global -> {len(out)} docs")
        return s

    # ── 2️⃣ STANDARD PATH ─────────────────────────────────────

    # 0) External-standard keyword pass
    std_match = STD_RE.search(q)
    kw_docs = []
    if std_match:
        raw_std = std_match.group(0)
        canon   = lambda t: re.sub(r"[^A-Za-z0-9]", "", t).lower()
        full    = canon(raw_std)
        num_m   = re.search(r"\d+", full)
        needle  = num_m.group(0) if num_m else ""
        def _std_hit(text: str) -> bool:
            c = canon(text)
            return (full and full in c) or (needle and needle in c)
        kw_docs = [d for d in all_docs if _std_hit(d.page_content)][:50]

    # 1) Scope resolution (explicit mentions only on the original Q)
    explicit_phrase  = extract_doc_phrase(orig_q)
    inherited_target = s.get("last_target")
    target           = resolve_doc(explicit_phrase) if explicit_phrase else inherited_target
    if DEBUG:
        print(f"[DEBUG] n_retrieve scope  explicit={explicit_phrase!r} "
              f"inherited={inherited_target!r} target={target!r}")

    # 2) Section‑override (strict doc scope for section lookup)
    if explicit_phrase and target and wants_clause_location(orig_q):
        body_chunks = [
            d for d in all_docs
            if d.metadata["source"] == target][:100]
        s["docs"]         = body_chunks
        s["used_sources"] = [target]
        add_trace(s, "retrieve_regdocs", True, f"section-override {target}")
        return s

    # 3) Doc‑scope override (strict scope for any explicit doc mention)
    if target and not wants_clause_location(orig_q):
        scoped = [
            d for d in all_docs
            if d.metadata["source"] == target]
        s["docs"]         = scoped
        s["used_sources"] = [target]
        add_trace(s, "retrieve_regdocs", True, f"doc-scope {target}")
        return s

    # 4) Build mixed pool (seeded by scoped + global + keyword hits)
    docs = []
    if target:
        docs.extend(d for d in all_docs
                    if d.metadata["source"] == target)[:SCOPED_MAX]

    glob_vec  = (chunk_retriever.invoke(q) or [])[:GLOBAL_K]
    glob_bm25 = (bm25_retriever.invoke(q) or [])[:GLOBAL_K]
    glob = [d for d in (glob_vec + glob_bm25)]
    if target:
        glob = [d for d in glob if d.metadata["doc_name"] != target]
    docs.extend(glob)

    # prepend standard‑keyword hits
    docs = kw_docs + docs

    # dedupe + cap
    seen = set(); deduped = []
    for d in docs:
        key = (d.metadata["doc_name"], d.metadata["section"], d.metadata.get("chunk_id") or id(d))
        if key in seen: continue
        seen.add(key); deduped.append(d)
        if len(deduped) >= 100: break
    s["docs"] = deduped

    # used_sources
    used = []
    if target: used.append(target)
    for d in deduped:
        src = d.metadata["doc_name"]
        if src not in used: used.append(src)
    s["used_sources"] = used

    # trace
    mode = (f"scoped(seed) {target} -> {len(deduped)} docs"
            if target else f"global -> {len(deduped)} docs")
    add_trace(s, "retrieve_regdocs", bool(deduped), mode)

    return s







# ────────────────────────────────────────────────────────────────
# NODE: n_rerank — smarter hybrid scoring with metadata boosts
# ────────────────────────────────────────────────────────────────
def n_rerank(s):
    if not on("reranker") or not s.get("docs"):
        add_trace(s, "rerank_chunks", False, "disabled or no docs")
        return s

    add_step(s, "rerank_chunks")
    docs_to_score = s["docs"][:KEEP_TOP_K]

    # 1️⃣ embedding sim
    q_emb  = minilm.encode(
        s["question"], normalize_embeddings=True, show_progress_bar=False
    )
    d_embs = minilm.encode(
        [d.page_content for d in docs_to_score],
        normalize_embeddings=True, show_progress_bar=False
    )
    raw_sims = cos_sim(q_emb, d_embs)[0].tolist()

    # 2️⃣ metadata-aware + heading bonus
    scored = []
    for sim, doc in zip(raw_sims, docs_to_score):
        meta   = doc.metadata
        score  = 0.8 * sim                       # base weight

        # small boost if heading title semantically matches question
        score += 0.2 * heading_similarity(
            s["question"], meta.get("section", "")
        )

        # demotions / promotions (unchanged)
        if meta.get("page_type") != "body":
            score -= 0.25
        wc = meta.get("word_count", 0)
        if wc < 20:  score -= 0.10
        if wc > 200: score -= 0.05
        score += 0.01 * meta.get("section_level", 0)
        # if meta.get("bib") or meta.get("junk"):
        #     score -= 0.20
        if "table of contents" in doc.page_content.lower():
            score -= 0.35

        scored.append((score, doc))

    scored.sort(key=lambda x: -x[0])
    sims_sorted, docs_sorted = zip(*scored)

    s["sims"]    = list(sims_sorted)
    s["docs"]    = list(docs_sorted)
    s["top_sim"] = sims_sorted[0] if sims_sorted else 0.0
    dbg(f"🪄 RERANK | top_sim={s['top_sim']:.3f}")
    add_trace(s, "rerank_chunks", True, f"top_sim={s['top_sim']:.2f}")
    return s



# ────────────────────────────────────────────────────────────────
# NODE: n_judge
# ────────────────────────────────────────────────────────────────
def n_judge(s):
    """
    Decide whether retrieval is 'enough' or we should broaden.
    — New guard: if we already inherited a scope and user didn't
      explicitly name a different doc, we auto‑pass.
    """
    add_step(s, "judge_relevance")
    q    = s["question"]
    docs = s.get("docs", [])
    # reset
    s["broaden_needed"] = False

    # detect explicit doc mention & inherited scope
    phrase    = extract_doc_phrase(q)
    target    = resolve_doc(phrase) if phrase else None
    inherited = s.get("last_target")

    # ── NEW GUARD ────────────────────────────────────────────
    # if we have inherited scope and user didn't name another,
    # we treat as 'enough' and never broaden again
    if inherited and not phrase:
        s["enough"] = True
        add_trace(s, "judge_relevance", True, "inherited-scope → enough")
        return s

    # 0️⃣ Explicit non‑section doc query
    if target and not wants_clause_location(q):
        s["enough"] = True
        add_trace(s, "judge_relevance", True, f"explicit-target: {target}")
        return s

    # 1️⃣ Clause‑location override
    if target and wants_clause_location(q):
        s["enough"] = True
        add_trace(s, "judge_relevance", True, f"clause-override for {target}")
        return s

    # 2️⃣ No docs
    if not docs:
        add_trace(s, "judge_relevance", False, "no docs")
        s["enough"] = False
        return s

    # 3️⃣ External‑standard keywords
    if STD_RE.search(q):
        s["enough"] = True
        add_trace(s, "judge_relevance", True, "std_keyword")
        return s

    # 4️⃣ Reranker sim logic
    if on("reranker"):
        top_sim = s.get("top_sim", 0.0)
        ok      = top_sim >= SIM_THRESHOLD
        s["enough"] = ok
        # only broaden if similarity low *and* we explicitly named a doc
        if not ok and inherited:
            s["broaden_needed"] = True
            add_trace(s, "judge_relevance", False,
                      f"sim {top_sim:.2f} → broaden")
        else:
            add_trace(s, "judge_relevance", ok,
                      f"sim {top_sim:.2f}")
        return s

    # 5️⃣ Manual sim fallback
    q_emb  = minilm.encode(q, normalize_embeddings=True,
                          show_progress_bar=False)
    d_embs = minilm.encode(
        [d.page_content for d in docs],
        normalize_embeddings=True,
        show_progress_bar=False
    )
    sims   = util.cos_sim(q_emb, d_embs)[0].tolist()
    top_sim = max(sims) if sims else 0.0
    s["top_sim"] = top_sim
    ok           = top_sim >= SIM_THRESHOLD
    s["enough"]  = ok
    if not ok and inherited:
        s["broaden_needed"] = True
        add_trace(s, "judge_relevance", False,
                  f"manual sim {top_sim:.2f} → broaden")
    else:
        add_trace(s, "judge_relevance", ok,
                  f"manual sim {top_sim:.2f}")
    return s








# ────────────────────────────────────────────────────────────────
# NODE: n_follow
# ────────────────────────────────────────────────────────────────
def n_follow(s):
    add_step(s, "follow_up")
    have_docs = bool(s.get("docs"))
    have_web  = bool((s.get("web_results") or "").strip())
    if DEBUG:
        print(f"[DEBUG] follow_up: have_docs={have_docs} have_web={have_web} "
          f"enough={s.get('enough')} docs_len={len(s.get('docs') or [])}")
    if not s.get("enough", False) and not have_web:
        s["follow_up_needed"] = True
        add_trace(s, "follow_up", True, "asked (not enough context)")
        return s
    elif have_docs or have_web:
        s["follow_up_needed"] = False
        add_trace(s, "follow_up", False, "skipped")
        return s
    s["follow_up_needed"] = True
    add_trace(s, "follow_up", True, "asked")
    return s


# ────────────────────────────────────────────────────────────────
# NODE: n_web
# ────────────────────────────────────────────────────────────────
def n_web(s):
    """
    Web‑fallback via SearXNG API.
    Now runs whenever web_search is on, regardless of s['enough'].
    Reads searx_config.json in configs folder to get search parameters
    """
    # only gate by feature flag:
    if not on("web_search"):
        add_trace(s, "web_search", False, "skipped")
        return s

    add_step(s, "web_search") #why added after being refd?
    host_ip = searx_ip
    if not host_ip:
        add_trace(s, "web_search", False, "no-ip")
        return s

    question = s["question"]
    language = searx_params["language"]
    engines = searx_params["engines"]
    num_results = searx_params["num_results"]

    try:
        search = SearxSearchWrapper(searx_host=host_ip, k=10)
        data = search.results(question, format="json", language=language, engines=engines, num_results=num_results)
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
        print(f"Failure - Unable to establish connection: {e}.")
        add_trace(s, "web_search", False, f"error {e}")
        return s
    except Exception as e:
        add_trace(s, "web_search", False, f"error {e}")
        return s
    
    # if search does not find result
    if not data or "snippet" not in data[0]:
        add_trace(s, "web_search", False, "No results found")
        return s

    snippets = [entry["snippet"] for entry in data]
    urls     = [entry["link"] for entry in data]

    s["web_results"] = "\n\n".join(snippets)
    s["web_urls"]    = urls

    # leave s['docs'] intact so you get both sources
    add_trace(s, "web_search", True, f"got {len(urls)} urls")
    return s




# ────────────────────────────────────────────────────────────────
# NODE: n_clause
# ────────────────────────────────────────────────────────────────

def n_clause(s):
    add_step(s, "clause_locator")

    # reset any previous lock
    s.pop("target_doc_id", None)       # ← NEW
    s.pop("target_section", None)      # ← NEW

    # ── 0️⃣ Skip if judge said “not enough” or feature-flag off ────
    if not s.get("enough", False) or not on("clause_locator"):
        add_trace(s, "clause_locator", False, "skipped (insufficient context)")
        s["clauses"] = []
        return s

    # ── 1️⃣ If explicit section lookup, do override ───────────────
    q      = s["question"]
    phrase = extract_doc_phrase(q)
    target = resolve_doc(phrase) if phrase else None

    if target and wants_clause_location(q):
        candidate_chunks = [
            d for d in s.get("docs", [])
            if d.metadata["source"] == target
               and d.metadata.get("page_type") == "body"
        ]
        q_terms = _key_terms(q)
        best_chunk = None
        best_score = 0
        for d in candidate_chunks:
            title_terms = set(re.findall(r"\b[a-z]{4,}\b",
                                         d.metadata.get("title", "").lower()))
            overlap = len(q_terms & title_terms)
            if overlap > best_score:
                best_score = overlap
                best_chunk = d

        if best_chunk and best_score > 0:
            sec = best_chunk.metadata["section"]
            s["clauses"] = [(1.0, target, sec)]
            s["target_doc_id"] = target          # ← NEW: lock the doc
            s["target_section"] = sec            # ← NEW: (optional) lock section
            add_trace(s, "clause_locator", True, f"override §{sec}")
            return s

    # ── 2️⃣ Fallback to embedding-based clause locator ────────────
    body_docs = [
        d for d in s.get("docs", [])
           if d.metadata.get("page_type") == "body"
    ]
    if not body_docs:
        add_trace(s, "clause_locator", False, "no body chunks")
        s["clauses"] = []
        # ensure no stale lock:
        s.pop("target_doc_id", None)            # ← NEW
        s.pop("target_section", None)           # ← NEW
        return s

    q_emb  = minilm.encode(q, normalize_embeddings=True, show_progress_bar=False)
    d_embs = minilm.encode([d.page_content for d in body_docs],
                            normalize_embeddings=True, show_progress_bar=False)
    sims   = cos_sim(q_emb, d_embs)[0].tolist()

    # pick best per section, then overall
    best_per_sec = {}
    for sim, d in zip(sims, body_docs):
        sec = d.metadata["section"]
        if sim > best_per_sec.get(sec, (0, None))[0]:
            best_per_sec[sec] = (sim, d)
    best_sec, (best_sim, best_doc) = max(best_per_sec.items(),
                                         key=lambda kv: kv[1][0])
    
    if best_sim >= 0.25:
        # Always use canonical doc_id = metadata["source"] when available
        doc_id = best_doc.metadata.get("source") or best_doc.metadata.get("doc_name")

        s["clauses"] = [(best_sim, doc_id, best_sec)]
        s["target_doc_id"] = doc_id          # lock canonical doc id
        s["target_section"] = best_sec

        add_trace(s, "clause_locator", True, f"{doc_id} §{best_sec}")
    else:
        s["clauses"] = []
        # ensure no stale lock if below threshold:
        s.pop("target_doc_id", None)
        s.pop("target_section", None)
        add_trace(s, "clause_locator", False, "below threshold")

    return s










# ────────────────────────────────────────────────────────────────
# NODE: clean_answer
# ────────────────────────────────────────────────────────────────
def clean_answer(text: str) -> str:
    m = re.search(r"ANSWER:\s*(.*)", text, flags=re.I)
    if not m:
        return "NOT FOUND"
    first = re.split(r"\n|  ", m.group(1).strip(), maxsplit=1)[0].strip()
    if (first.upper().startswith("NOT FOUND")
        or len(first.split()) < 3
        or re.search(r"\b(context|question|web)\b", first, flags=re.I)):
        return "NOT FOUND"
    return first



# ── helper: detect no-data answers ────────────────────────────────
def _no_data(text: str) -> bool:
    low = text.lower()
    return low.startswith(("not found", "i'm sorry", "could not find"))

def _top_chunks_for_answer(answer: str,
                           docs: list,
                           k: int = 2) -> list[str]:
    """
    Return up to k `source` ids whose chunk-text is most similar to *answer*.
    Uses MiniLM cosine similarity on embeddings already in memory.
    """
    if not docs or not answer.strip():
        return []
    ans_emb = minilm.encode(answer, normalize_embeddings=True,
                            show_progress_bar=False)
    cand_embs = minilm.encode(
        [d.page_content for d in docs], normalize_embeddings=True,
        show_progress_bar=False
    )
    sims = util.cos_sim(ans_emb, cand_embs)[0].tolist()
    scored = sorted(zip(sims, docs), key=lambda x: -x[0])
    return [d.metadata["doc_name"] for sim, d in scored[:k] if sim > 0.25]


# ────────────────────────────────────────────────────────────────
#  n_synth — synthesise answer & attach valid sources
# ────────────────────────────────────────────────────────────────
def n_synth(s):
    add_step(s, "synthesize")
    dbg("🖋 SYNTH enter")

    s.setdefault("used_sources", [])
    response_sections: dict[str, list[str]] = {}

    # ---------- 1️⃣ Enumeration fast-path (unchanged) -------------
    enum = detect_enumeration_question(s["question"])
    if enum and s.get("clauses"):
        count, term = enum
        # doc_id here is now canonical: metadata["source"]
        _, doc_id, sec = max(s["clauses"], key=lambda p: p[0])

        text = load_section_text(doc_id, sec)
        items = extract_enumeration_items(text, count) or [i.strip() for i in text.split(";")][:count]

        response_sections["Answer"] = [f"The {count} {term} are (from §{sec}):"] + [
            f"- {itm}" for itm in items
        ]

        # look up metadata by source + section
        md = next(
            (
                d.metadata
                for d in s["docs"]
                if d.metadata.get("source") == doc_id
                and str(d.metadata.get("section")) == str(sec)
            ),
            None,
        )

        if md:
            doc_name   = md.get("doc_name", doc_id)
            sect_path  = md.get("section_path", sec)
            pl         = md.get("page_label")
            pi         = md.get("page_index")

            if pl is not None and pi is not None and pl != pi:
                src_line = f"**{doc_name}**: §{sec} “{sect_path}” (page printed {pl}) (page {pi})"
            elif pl is not None:
                src_line = f"**{doc_name}**: §{sec} “{sect_path}” (page {pl})"
            else:
                src_line = f"**{doc_name}**: §{sec} “{sect_path}”"

            response_sections["Sources"] = [src_line]
            s["used_sources"] = [doc_id]

        s["response_sections"] = response_sections
        add_trace(s, "synthesize", True, "enum-fastpath")
        return s


        # ---------- 2️⃣ Clause-location fast-path ---------------------
    if s.get("clauses") and wants_clause_location(s["question"]):
        # doc_id is canonical: metadata["source"]
        _, doc_id, sec = s["clauses"][0]

        md = next(
            (
                d.metadata
                for d in s["docs"]
                if d.metadata.get("source") == doc_id
                and str(d.metadata.get("section")) == str(sec)
            ),
            None,
        )

        if md:
            doc_name = md.get("doc_name", doc_id)
            title    = md.get("title") or md.get("section_path")
            pl       = md.get("page_label") or md.get("page")

            head = f"{doc_name} §{sec}"
            if title:
                head += f' “{title}”'
            if pl is not None:
                head += f" (page {pl})"
        else:
            # Fallback if no metadata match (should be rare)
            head = f"{doc_id} §{sec}"

        response_sections["Answer"] = [head]
        response_sections["Sources"] = [head]
        s["response_sections"] = response_sections
        s["used_sources"] = [doc_id]
        add_trace(s, "synthesize", True, "clause-fastpath")
        return s


    # ---------- 3️⃣ Follow-up needed --------------------------------
    if s.get("follow_up_needed"):
        response_sections["Answer"] = ["Could not find any data for this query. Could you clarify?"]
        s["response_sections"] = response_sections

        # carry forward the same doc we were scoped to
        if s.get("docs"):
            src = s["docs"][0].metadata["doc_name"]
            s["used_sources"] = [src]
        else:
            s["used_sources"] = []

        add_trace(s, "synthesize", True, "follow_up_shortcircuit")
        return s

    # ---------- 4️⃣ Web-only shortcut ------------------------------
    web = s.get("web_results", "").strip()
    if on("web_search") and web and not s.get("enough"):
        response_sections["Answer"] = [web]
        if s.get("web_urls"):
            response_sections["Sources"] = [f"- {u}" for u in s["web_urls"]]
        s["response_sections"] = response_sections
        # keep only Sources for web-only answers
        s['response_sections'].update({"external_refs": [], "requirements": [], "subtopics": [], "next_questions": []})
        add_trace(s, "synthesize", True, "web-pass")
        return s

    # ---------- 5️⃣ No vector context ------------------------------
    has_ctx = bool(s.get("docs"))
    if not has_ctx and not web:
        response_sections["Answer"] = ["NOT FOUND"]
        s["response_sections"] = response_sections
        add_trace(s, "synthesize", False, "no_data")
        return s

    # ---------- 6️⃣ Chat-memory for pronouns etc. ------------------
    mem_ctx = memory_context(
        s["question"], k=4,
        user=s.get("user", "default"),
        chat_id=s.get("chat_id", "default_chat")
    )

    # detect explicit target doc
    phrase = extract_doc_phrase(s["question"])

    # ---------- 6a ▸ Top-doc context pass only if no explicit doc ----
    if has_ctx and not phrase:
        top_doc = s["docs"][0]
        ctx = top_doc.page_content
        q_terms = set(re.findall(r"\b[a-z]{4,}\b", s["question"].lower()))
        ctx_terms = set(re.findall(r"\b[a-z]{4,}\b", ctx.lower()))
        if len(q_terms & ctx_terms) / (len(q_terms) or 1) >= 0.25:
            raw = _txt(get_llm().invoke(chat_wrap(render_prompt("ctx_pass.jinja", mem_ctx=mem_ctx, ctx=ctx, question=s['question'])))).strip()
            if not _no_data(raw):
                clean = re.sub(r"^ANSWER:\s*", "", raw, flags=re.I).strip()
                response_sections["Answer"] = [clean]
                md = top_doc.metadata
                if md['page_label'] == md['page_index']:
                    response_sections["Sources"] = [f"**{md['doc_name']}**: §{md['section_path']} (page {md['page_label']})"]
                else:
                    response_sections["Sources"] = [f"**{md['doc_name']}**: §{md['section_path']} (page printed {md['page_label']}) (page {md['page_index']})"]
                s["response_sections"] = response_sections
                s["used_sources"] = [md["doc_name"]]
                add_trace(s, "synthesize", True, "ctx-pass")
                return s

    # ========== 7️⃣ Full-context fallback → EVIDENCE-FIRST ==========
    # Helper to pick evidence chunks (MiniLM cosine)
    def _top_chunks_for_answer_chunks(text: str, docs: list, k: int = 3, min_sim: float = 0.25):
        if not text or not docs:
            return []
        try:
            q_emb  = minilm.encode(text, normalize_embeddings=True, show_progress_bar=False)
            d_embs = minilm.encode([d.page_content for d in docs], normalize_embeddings=True, show_progress_bar=False)
        except Exception:
            return []
        sims   = cos_sim(q_emb, d_embs)[0].tolist()
        scored = sorted(zip(sims, docs), key=lambda x: -x[0])
        out = []
        for sc, d in scored:
            if len(out) >= k: break
            if sc >= min_sim: out.append(d)
        return out

    docs_list = s.get("docs", []) or []

    # lock candidate set if clause locator or explicit phrase found a doc
    locked_doc = (
        s.get("target_doc_id") or
        (s["clauses"][0][1] if s.get("clauses") else None) or
        (resolve_doc(phrase) if phrase else None)
    )
    # locked_doc is now always a canonical doc_id = metadata["source"]
    cand_docs = [
        d for d in docs_list
        if (not locked_doc or d.metadata.get("source") == locked_doc)
    ]


    # 7a) Choose evidence BEFORE answering (use QUESTION for evidence selection)
    evidence = _top_chunks_for_answer_chunks(s["question"], cand_docs[:KEEP_TOP_K], k=3, min_sim=0.25)
    if not evidence and cand_docs:
        evidence = cand_docs[:1]  # safe fallback so we don't answer from "everything"

    # 7b) Build the evidence-only context (+WEB if available)
    ctx = "\n\n".join(d.page_content for d in evidence)
    web_txt = web
    if web_txt:
        ctx += "\n\nWEB:\n" + web_txt

    web_suffix = " and (if present) WEB." if web_txt else "."

    prompt = prompt = render_prompt(
    "evidence_first.jinja",
    ctx=ctx,
    web_suffix=" and (if present) WEB." if web_txt else ".",
    question=s['question']
)

    raw2 = _txt(get_llm().invoke(chat_wrap(prompt))).strip()
    final = re.sub(r"^ANSWER:\s*", "", raw2, flags=re.I).strip()

    # if still no useful data
    if _no_data(final):
        response_sections["Answer"] = [final]
        s["used_sources"] = []
        s["response_sections"] = response_sections
        add_trace(s, "synthesize", True, "no-data-fallback")
        return s

    # ---------- assemble answer + citations from EVIDENCE ----------
    # ---------- assemble answer + citations from EVIDENCE ----------
    response_sections["Answer"] = [final]

    src_lines: list[str] = []
    used_docs: list[str] = []
    seen_docs: set[str] = set()

    for d in evidence:
        md = d.metadata or {}

        # Canonical ID used for scoping/memory (prefer "source"; fallback to "doc_name")
        src_id = md.get("source") or md.get("doc_name", "")

        # Pretty name for display
        doc_pretty = md.get("doc_name", "")

        # Skip if we don't have a displayable doc name or we've already cited this pretty doc
        if not doc_pretty or doc_pretty in seen_docs:
            continue
        seen_docs.add(doc_pretty)

        # Track canonical IDs for memory/scoping
        if src_id:
            used_docs.append(src_id)

        sec   = md.get("section_path") or md.get("section") or "Document"
        title = md.get("title", "")
        pl    = md.get("page_label")
        pi    = md.get("page_index")

        if pl is not None and pi is not None and pl != pi:
            line = f"**{doc_pretty}**: §{sec} “{title}” (page printed {pl}) (page {pi})"
        elif pl is not None:
            line = f"**{doc_pretty}**: §{sec} “{title}” (page {pl})"
        else:
            line = f"**{doc_pretty}**: §{sec} “{title}”"
        src_lines.append(line)

    if src_lines:
        response_sections["Sources"] = src_lines
        # Deduplicate while preserving order
        s["used_sources"] = list(dict.fromkeys(used_docs))

    s["response_sections"] = response_sections
    add_trace(s, "synthesize", True, "fallback-evidence-first")
    return s











# ╭──────────────── n_req  — normative requirement list ──────────────╮
# ─────────────────────────────────────────────────────────────
#  3) n_req  — unblock requirements on clause-queries
# ─────────────────────────────────────────────────────────────
def n_req(s):
    # dbg(f"n_req running on these docs: {[d.metadata['source'] + ' §' + d.metadata['section'] for d in s['docs'][:KEEP_TOP_K]]}")
    if (
        not on("req_extractor")
        or not s.get("response_sections")
        or not wants_requirements(s["question"])
    ):
        add_trace(s, "req_extract", False, "skipped")
        return s

    reqs = extract_requirements(
        s["docs"][:KEEP_TOP_K],
        s["question"],
        max_out=REQ_BULLET_LIMIT
    )
    if reqs:
        # build a list of formatted bullets
        bullets = [
            f"- **[{r['strength']}]** {r['text']} *( {r['doc']} §{r['sec']} )*"
            for r in reqs
        ]
        # attach as its own section
        s["response_sections"]["Requirements"] = bullets
        add_trace(s, "req_extract", True, f"{len(bullets)} reqs")
    else:
        add_trace(s, "req_extract", False, "none")
    return s






# ╭──────────────── n_ext  — external reference list ────────────────╮
def n_ext(s):
    """
    Attach external-standard citations (IAEA, CSA, ISO, etc.) to response_sections
    after the answer is generated. Requires:
      - `external_refs` toggle on
      - a completed response in response_sections["Answer"]
    """
    # 1️⃣ If we already need a follow-up, skip external refs
    if s.get("follow_up_needed"):
        add_trace(s, "ext_refs", False, "skipped (follow-up)")
        return s

    # 2️⃣ Feature-flag off or no answer yet → skip
    #    We now store answers in response_sections["Answer"], not s["answer"]
    if (
        not on("external_refs")
        or not s.get("response_sections")
        or not s["response_sections"].get("Answer")
    ):
        add_trace(s, "ext_refs", False, "skipped (no answer yet)")
        return s

    # 3️⃣ Scan the top-K retrieved chunks for external standard citations
    texts = " ".join(d.page_content for d in s["docs"][:KEEP_TOP_K])
    refs = find_ext_refs(texts)

    if refs:
        # Attach as its own section under "External references"
        s["response_sections"]["External references"] = refs
        add_trace(s, "ext_refs", True, f"{len(refs)} refs")
    else:
        add_trace(s, "ext_refs", False, "none")

    return s


# ╭────────────────────────────────────────────────────────────╮
# │  NODE: n_cite                                              │
# ╰────────────────────────────────────────────────────────────╯
# ─── and in your n_cite ─────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
#  n_cite — attach sources only when they are meaningful
# ─────────────────────────────────────────────────────────────
def n_cite(s):
    # 0) feature flag
    if not on("citations"):
        add_trace(s, "citations", False, "disabled")
        return s

    # 1) If answer already stored in structured sections, skip
    if s.get("response_sections"):
        add_trace(s, "citations", False, "sections-present")
        return s

    raw_answer = s.get("answer", "").strip()
    if not raw_answer:
        add_trace(s, "citations", False, "no answer str")
        return s

    # helper to detect no-data replies
    if _no_data(raw_answer):
        add_trace(s, "citations", False, "no-data answer")
        return s

    # 2) collect used_sources
    cites = [src for src in s.get("used_sources", []) if src]
    if not cites:
        add_trace(s, "citations", False, "empty used_sources")
        return s

    add_step(s, "citations")

    lines = []
    for src in cites:
        md = next((d.metadata for d in s["docs"] if d.metadata["doc_name"] == src), {})
        doc_name = md.get("doc_name", "")
        sec      = md.get("section_path", "")
        page     = md.get("page", "")

        part = f"- **{doc_name}**"
        if sec:
            part += f": §{sec}"
        if page:
            part += f" (page {page})"
        lines.append(part)

    if lines:
        s["answer"] = raw_answer + "\n\nSources:\n" + "\n".join(lines)

    add_trace(s, "citations", bool(lines), f"{len(lines)} sources")
    return s

import re

def n_suggest(s):
    # Only run if feature enabled and we have retrieved docs
    if not on("topic_suggestion") or not s.get("docs"):
        add_trace(s, "suggest", False, "disabled or no docs")
        return s
    if on("web_search") and s.get("web_results", "").strip():
        add_trace(s, "suggest", False, "skipped (web search)")
        return s

    add_step(s, "suggest")

    # Build a single context string from top-K chunks
    context = "\n\n".join(d.page_content for d in s["docs"][:KEEP_TOP_K])

    # Dynamic prompt: let the model decide how many items to return
    prompt = f"""You are a research assistant.
Using *only* the CONTEXT below, output exactly two markdown sections:

### Subtopics
- List the key subtopics the user could explore next.

### Next Questions
- List the most useful follow‑up questions the user might ask.

Do NOT include any extra commentary.

CONTEXT:
{context}
"""

    # Invoke the LLM with temperature=0 for consistency
    raw = _txt(get_llm().invoke(chat_wrap(prompt), temperature=0))
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    subtopics, next_qs = [], []
    section = None

    # Parse bullets under each heading
    for ln in lines:
        lower = ln.lower()
        if lower.startswith("### subtopics"):
            section = "sub"
            continue
        if lower.startswith("### next questions"):
            section = "next"
            continue
        if ln.startswith("-") and section:
            item = ln.lstrip("- ").strip()
            if section == "sub":
                subtopics.append(item)
            elif section == "next":
                next_qs.append(item)

    # Attach to response_sections if found
    if subtopics:
        s["response_sections"]["Subtopics"] = [f"- {t}" for t in subtopics]
    if next_qs:
        s["response_sections"]["Next questions"] = [f"- {q}" for q in next_qs]

    add_trace(s, "suggest", True,
              f"{len(subtopics)} topics, {len(next_qs)} questions")
    return s



# ╭────────────────────────────────────────────────────────────╮
# │  NODE: n_gap                                               │
# ╰────────────────────────────────────────────────────────────╯
def n_gap(s):
    if on("query_gap_logger") and s.get("enough") is False:
        add_step(s, "gap_log")
        with open("/kaggle/working/unanswered.csv", "a") as f:
            f.write(f"{time.time()},{s['question']}\n")
    dbg(f"🚨 GAP | logged={s.get('enough') is False}")
    add_trace(s, "gap_log", True,
              "logged" if s.get("enough") is False else "none")
    return s

# 📦 Cell F — state schema & LangGraph build
# -----------------------------------------

class ChatState(TypedDict, total=False):
    question: str
    question_for_retrieval: str       # ← what we actually send to n_retrieve
    retrieval_seed: str               # ← the prior user query that seeded last_target
    docs: List[Any]
    web_results: str
    web_urls: List[str]
    clauses: List[str]
    enough: bool
    follow_up_needed: bool
    answer: str
    plan: List[str]
    trace: List[str]
    top_sim: float
    used_sources: List[str]
    response_sections: Dict[str, List[str]]
    chat_id: str
    user: str
    last_target: str
    last_answer: str
    broaden_needed: bool  # <-- added: judge can set this to trigger a global re-retrieval


# ---- build the graph ----
g = StateGraph(ChatState)

g.add_node("retrieve", n_retrieve)
g.add_node("rerank",   n_rerank)
g.add_node("judge",    n_judge)
g.add_node("web",      n_web)
g.add_node("clause",   n_clause)
g.add_node("follow",   n_follow)
g.add_node("synth",    n_synth)
g.add_node("req",      n_req)
g.add_node("ext",      n_ext)
g.add_node("cite",     n_cite)
g.add_node("suggest",  n_suggest)
g.add_node("gap",      n_gap)

# linear base edges
g.add_edge("retrieve", "rerank")
g.add_edge("rerank",   "judge")

# 3-way router after judge: broaden? else enough? else web
def _route_after_judge(s: ChatState) -> str:
    """
    Routing logic after n_judge:
      - If judge flagged broaden_needed → loop back to 'retrieve' (global pass).
      - Else if enough=True → go on to clause processing.
      - Else → try web search.
    Returns one of: 'broaden', 'clause', 'web'.
    """
    if s.get("broaden_needed"):
        return "broaden"
    return "clause" if s.get("enough") else "web"

g.add_conditional_edges(
    "judge",
    _route_after_judge,
    {
        "broaden": "retrieve",  # loop back for global retrieval
        "web":     "web",
        "clause":  "clause",
    },
)

# downstream path unchanged
g.add_edge("web",      "clause")
g.add_edge("clause",   "follow")
g.add_edge("follow",   "synth")
g.add_edge("synth",    "req")
g.add_edge("req",      "ext")
g.add_edge("ext",      "cite")
g.add_edge("cite",     "suggest")
g.add_edge("suggest",  "gap")

g.set_entry_point("retrieve")
g.set_finish_point("gap")
graph = g.compile()

print("✓ Graph compiled")

# ────────────────────────────────────────────────────────────────────
#  Helper: invoke the graph for ONE question + handle chat memory
# ────────────────────────────────────────────────────────────────────
def run_and_report(
    question: str,
    *,
    chat_id: str = "default_chat",
    user: str = "default",
) -> dict:
    """
    Execute the LangGraph pipeline for one question.
    Prints step reports & partial answers only if DEBUG is truthy.
    """
    if DEBUG:
        print(f"\nRunning for query: {question}")

    res = graph.invoke({
        "question": question,
        "user":     user,
        "chat_id":  chat_id
    })

    if DEBUG:
        print("\nSTEP REPORT\n" + "─" * 60)
        for line in res.get("trace", []):
            print(line)
        print("─" * 60)
        for header, lines in (res.get("response_sections") or {}).items():
            print(f"\n{header}:")
            for l in lines:
                print(l)

    # persist memory if enabled (unchanged)
    if on("memory"):
        write_memory("user", question, user=user, chat_id=chat_id)
        answer_txt = " ".join(res["response_sections"].get("Answer", []))
        sources    = res.get("used_sources", [])
        write_memory(
            "assistant",
            answer_txt,
            doc_ids=sources,
            user=user,
            chat_id=chat_id
        )

    return res



# ────────────────────────────────────────────────────────────────────
#  Main entrypoint: handle multi-part questions + chat session
# ────────────────────────────────────────────────────────────────────


def ask_entrypoint(
    user_q: str,
    *,
    chat_id: str = "default_chat",
    user:    str = "default",
    **run_kwargs
):
    """
    Top-level orchestrator, now with retrieval_seed support:
      • retrieval_seed is the prior user Q that first selected last_target.
      • question_for_retrieval = retrieval_seed + current question, if seed exists.
    """
    # 1️⃣ Pull memory (for last_target & retrieval_seed)
    if on("memory"):
        rows = read_memory(chat_id=chat_id, user=user, with_text=True, limit=8)
        if DEBUG:
            print("[DEBUG] MEMORY ROWS:", rows)

        last_target    = None
        retrieval_seed = None
        # rows are newest→oldest
        for idx, entry in enumerate(rows):
            meta = entry["meta"]
            if meta.get("role") == "assistant":
                doc_ids = meta.get("doc_ids") or []
                if isinstance(doc_ids, str):
                    try:    doc_ids = json.loads(doc_ids)
                    except: doc_ids = []
                if doc_ids:
                    last_target = doc_ids[0]
                    # the preceding row (idx+1) is the user Q that triggered it
                    if idx+1 < len(rows) and rows[idx+1]["meta"].get("role") == "user":
                        retrieval_seed = rows[idx+1]["text"]
                    break

        if DEBUG:
            print(f"[DEBUG] seeded last_target={last_target!r} retrieval_seed={retrieval_seed!r}")
        # build a short chat context if you like (for pronoun resolution, etc.)
        u_txt = [r["text"] for r in rows if r["meta"]["role"]=="user"][:5]
        a_txt = [r["text"] for r in rows if r["meta"]["role"]=="assistant"][:3]
        chat_ctx = "\n".join(u_txt + a_txt)
    else:
        last_target    = None
        retrieval_seed = None
        chat_ctx       = ""

    # 2️⃣ Decompose into standalone parts
    raw_parts = decompose(user_q, chat_ctx)
    if DEBUG:
        print("[DEBUG] decomposed parts:", raw_parts)

    merged_sections = OrderedDict()
    combined_trace  = []
    parts_output    = []
    last_answer     = None

    # 3️⃣ Process each sub-question
    for raw_part in raw_parts:
        norm_part = normalize_question(raw_part)

        # build the retrieval query
        if retrieval_seed:
            retrieval_q = retrieval_seed.strip() + " " + norm_part
        else:
            retrieval_q = norm_part

        phrase  = extract_doc_phrase(norm_part)
        has_ref = bool(re.search(r"\b(it|they|those|that|this|these|the above)\b", norm_part, re.I))
        if not phrase and not has_ref:
            last_target = None

        state = {
            "question":                 norm_part,
            "question_for_retrieval":   retrieval_q,
            "orig_question":            raw_part,
            "user":                     user,
            "chat_id":                  chat_id,
            "last_target":              last_target,
            "last_answer":              last_answer,
        }
        if DEBUG:
            print(f"[DEBUG] running part={raw_part!r} last_target={last_target!r} retrieval_q={retrieval_q!r}")

        res = graph.invoke(state, **run_kwargs)
        combined_trace.extend(res.get("trace", []))

        rsp = res.get("response_sections", {})
        parts_output.append({
            "question":      raw_part,
            "answer":        rsp.get("Answer", []),
            "sources":       rsp.get("Sources", []),
            "requirements":  rsp.get("Requirements", []),
            "external_refs": rsp.get("External references", []),
            "subtopics":      rsp.get("Subtopics", []),
            "next_questions": rsp.get("Next questions", []),
            "trace":         res.get("trace", []),
        })
        for sect, lines in rsp.items():
            merged_sections.setdefault(sect, []).extend(lines)

        # update last_answer
        last_answer = " ".join(rsp.get("Answer", []))

        # update last_target
        resolved = resolve_doc(phrase) if phrase else None
        if resolved:
            last_target = resolved
        elif has_ref:
            used = res.get("used_sources") or []
            if used:
                last_target = used[0]

        if DEBUG:
            print(f"[DEBUG] updated last_target={last_target!r}")

    # 4️⃣ Follow-up short-circuit (if any)
    if any("follow_up_shortcircuit" in step for step in combined_trace):
        followup_text = merged_sections.get("Answer", [""])[0]
        # if on("memory"):
        #     write_memory("user",      user_q,        doc_ids=[last_target], chat_id=chat_id, user=user)
        #     write_memory("assistant", followup_text, doc_ids=[last_target], chat_id=chat_id, user=user)
        return {
            "answer":        followup_text,
            "sources":       [],
            "requirements":  [],
            "external_refs": [],
            "subtopics":      [],
            "next_questions": [],
            "trace":         combined_trace,
        }

    # 5️⃣ Fuse partial answers via LLM
    bullets = merged_sections.get("Answer", [])
    block   = "\n".join(f"- {b.strip()}" for b in bullets if b.strip())
    fuse_prompt = f"""You are a helpful research assistant.
TASK: Combine the following partial answers into one concise answer.
PARTIAL ANSWERS:
{block}

FINAL ANSWER:"""
    raw_final = _txt(get_llm().invoke(chat_wrap(fuse_prompt))).strip()
    final_ans = re.sub(r"^FINAL ANSWER:\s*", "", raw_final, flags=re.I).strip()

    # 6️⃣ Aggregate and return
    output = {
        "answer":        final_ans,
        "sources":       merged_sections.get("Sources", []),
        "requirements":  merged_sections.get("Requirements", []),
        "external_refs": merged_sections.get("External references", []),
        "subtopics":      merged_sections.get("Subtopics", []),
        "next_questions": merged_sections.get("Next questions", []),
        "trace":         combined_trace,
    }

    # if on("memory"):
    #     write_memory("user",      user_q,    doc_ids=[last_target], chat_id=chat_id, user=user)
    #     write_memory("assistant", final_ans, doc_ids=[last_target], chat_id=chat_id, user=user)

    if DEBUG:
        full = {"parts": parts_output, "final": output}
        print(json.dumps(full, indent=2, ensure_ascii=False))
        return full
    else:
        return output

def switch(_silent_reset=False, **overrides):
    FEATURES.clear()
    FEATURES.update(copy.deepcopy(FEATURES_DEFAULT))
    for k,v in overrides.items():
        FEATURES[k]["enabled"] = v
    if not _silent_reset:
        print("Active toggles:", {k: f["enabled"] for k,f in FEATURES.items()})

def print_memory_history(
    user: str = "default",
    chat_id: str = "default_chat",
    limit: int = 100,
):
    """
    Debug helper: pretty-print recent memory entries for a (user, chat_id) pair.
    """
    hist = read_memory(user=user, chat_id=chat_id, limit=limit, with_text=True)

    print("=== Chat History ===")
    for i, row in enumerate(hist, 1):
        # when with_text=True, each row is {"text": ..., "meta": ...}
        text = row.get("text") if isinstance(row, dict) else str(row)
        print(f"{i:02d}. {text}")
    print("====================")
