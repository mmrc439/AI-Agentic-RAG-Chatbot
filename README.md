
# Agentic RAG Chatbot — Updated README

This repository is a document‑answering **Agentic RAG** chatbot. It ingests PDFs/DOCX, builds a **ChromaDB** index with **BGE-base** embeddings, and runs an agentic answerer with features like reranking, clause extraction, citations, memory, and optional web search. A minimal **Flask** UI is included.

> New here? Start with **Quickstart** → **Run the app** → **Upload a doc** → **Ask a question**.

---

## Contents

- [Quickstart](#quickstart)
- [Project Structure](#project-structure)
- [Configuration & Flags](#configuration--flags)
  - [Core flags (`src/configs/helper.py`)](#core-flags-srcconfigshelperpy)
  - [Agentic feature toggles (`src/configs/agentic_features.py`)](#agentic-feature-toggles-srcconfigsagentic_featurespy)
  - [Environment variables (`.env`)](#environment-variables-env)
- [How It Works](#how-it-works)
  - [Ingestion & Chunking](#ingestion--chunking)
  - [Embeddings & Vector Stores](#embeddings--vector-stores)
  - [Retrieval & Ranking](#retrieval--ranking)
  - [Reasoning, Memory & Output](#reasoning-memory--output)
- [Web App Endpoints](#web-app-endpoints)
- [Common Tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

**Requirements**
- Python 3.10+ recommended
- `pip install -r requirements.txt`
- Fill `.env` (see below) with your keys (OpenAI for default LLM; HF token optional).

**Steps**
1. Create folders if they don’t exist: `Docs/` (source docs) and `data/` (web uploads).
2. Start a virtual environment `py -3.10 -m venv .venv`, then `.\.venv\Scripts\activate`
3. Install deps: `pip install -r requirements.txt`.
4. Configure flags (see tables below).
5. Run the web app: `python -m src.web.app`  (note: Flask runs with `debug=True` in app.py — set to False for prod.)
6. Open `http://localhost:5000` → upload files → chat.

> First run will build a Chroma store in `STORE_DIR` (default: "chroma_store").

---

## Project Structure

```
rag/
├─ .env
├─ requirements.txt
├─ README.md
├─ Docs/                      # served at /Docs/... (static pass-through)
├─ data/                 # upload target for /upload (UI)
├─ src/
│  ├─ configs/
│  │  ├─ config.py            # loads DEBUG from .env
│  │  ├─ helper.py            # core flags (LLM, chunking, GPT probes, paths)
│  │  └─ agentic_features.py  # feature toggles (reranker, memory, web_search, ...)
│  ├─ data_ingest/preprocess.py  # header/footer cleanup, TOC helpers, GPT probes
│  ├─ chunking/chunker.py        # parse_pdf/docx_to_chunks, chunk schema
│  ├─ embeddings/index/chroma_db.py  # Chroma setup, initial indexing, retrievers
│  ├─ pipelines/agentic.py      # main agent: retrieval→reasoning→answer (+memory)
│  ├─ prompts/*.jinja           # prompt templates
│  └─ web/app.py               # Flask app + endpoints
└─ scripts/
   ├─ ingest.py                # programmatic ingestion helper
   └─ delete.py                # delete files from Chroma + filesystem
```

---

## Configuration & Flags

### Core flags (`src/configs/helper.py`)

| Flag                   | Default                                                              | Where                 |
| ---------------------- | -------------------------------------------------------------------- | --------------------- |
| LLM_PROVIDER           | "OPENAI"    # or literally anything else, that would force MistralAI | src/configs/helper.py |
| GPT_MODEL              | "gpt-4o-mini"                                                        | src/configs/helper.py |
| LLM_NAME               | "mistralai/Mistral-7B-Instruct-v0.3"                                 | src/configs/helper.py |
| DEVICE                 | "cuda" if torch.cuda.is_available() else "cpu"                       | src/configs/helper.py |
| RUN_GPT_LAYOUT_PROBE   | True                                                                 | src/configs/helper.py |
| RUN_GPT_IMAGE_DESCRIBE | False                                                                | src/configs/helper.py |
| RUN_GPT_DOC_SUMMARY    | False                                                                | src/configs/helper.py |
| STORE_DIR              | "chroma_store"                                                       | src/configs/helper.py |
| DATA_DIR               | "Docs"    # where your files live                                    | src/configs/helper.py |
| MAX_TOKENS             | 512        # max tokens per chunk                                    | src/configs/helper.py |
| OVERLAP                | 50            # overlap between chunks                               | src/configs/helper.py |
| CITE_SIM_THRESHOLD     | 0.3      # citations gate                                            | src/configs/helper.py |
| SCOPED_MAX             | 100        # per-doc cap                                             | src/configs/helper.py |
| GLOBAL_K               | 40         # global candidate cap feeding broaden pool               | src/configs/helper.py |
| REQ_BULLET_LIMIT       | 10                                                                   | src/configs/helper.py |

**Notes**

- **LLM selection**: If `LLM_PROVIDER == "OPENAI"`, the app uses OpenAI Chat Completions with `GPT_MODEL` (default: `gpt-4o-mini`). Otherwise it uses the local/HF chat template with `LLM_NAME` (default: `mistralai/Mistral-7B-Instruct-v0.3`).  
- **Chunking**: `MAX_TOKENS` & `OVERLAP` control chunk sizes.  
- **GPT probes**: 
  - `RUN_GPT_LAYOUT_PROBE`: infer header/footer regexes & TOC hints from a few sample pages.
  - `RUN_GPT_IMAGE_DESCRIBE`: describe images/charts into text chunks.
  - `RUN_GPT_DOC_SUMMARY`: make a 1‑chunk doc summary (boosts doc‑level recall).
- **Storage**: `STORE_DIR` is the Chroma persist folder. `DATA_DIR` is the initial doc scan root (served at `/Docs/...`).

### Agentic feature toggles (`src/configs/agentic_features.py`)

| Name             | What it does                                    | Default | Edit here                       | Toggle at runtime                                                                              |
| ---------------- | ----------------------------------------------- | ------- | ------------------------------- | ---------------------------------------------------------------------------------------------- |
| reranker         | MiniLM re-rank retrieved chunks                 | on      | src/configs/agentic_features.py | Use Python: `from src.pipelines.agentic import switch; switch(**{'reranker': 0 or 1})`         |
| web_search       | SERPAPI fallback search                      | off     | src/configs/agentic_features.py | Use Python: `from src.pipelines.agentic import switch; switch(**{'web_search': 0 or 1})`       |
| follow_up        | Clarifying-question agent                       | on      | src/configs/agentic_features.py | Use Python: `from src.pipelines.agentic import switch; switch(**{'follow_up': 0 or 1})`        |
| clause_locator   | Locate exact RegDoc sections                    | on      | src/configs/agentic_features.py | Use Python: `from src.pipelines.agentic import switch; switch(**{'clause_locator': 0 or 1})`   |
| req_extractor    | Pull SHALL / MUST requirement lines             | on      | src/configs/agentic_features.py | Use Python: `from src.pipelines.agentic import switch; switch(**{'req_extractor': 0 or 1})`    |
| external_refs    | List IAEA / CSA / ISO refs in answer            | on      | src/configs/agentic_features.py | Use Python: `from src.pipelines.agentic import switch; switch(**{'external_refs': 0 or 1})`    |
| citations        | Attach paragraph-level citations                | on      | src/configs/agentic_features.py | Use Python: `from src.pipelines.agentic import switch; switch(**{'citations': 0 or 1})`        |
| topic_suggestion | Suggest related subtopics & follow‑up questions | on      | src/configs/agentic_features.py | Use Python: `from src.pipelines.agentic import switch; switch(**{'topic_suggestion': 0 or 1})` |
| query_gap_logger | Log low-confidence queries                      | off     | src/configs/agentic_features.py | Use Python: `from src.pipelines.agentic import switch; switch(**{'query_gap_logger': 0 or 1})` |
| memory           | Memory for chats                                | on      | src/configs/agentic_features.py | Use Python: `from src.pipelines.agentic import switch; switch(**{'memory': 0 or 1})`           |

**Runtime toggling example (Python REPL):**
```python
from src.pipelines.agentic import switch
# turn OFF web search, turn ON query gap logger:
switch(web_search=0, query_gap_logger=1)
```

### Environment variables (`.env`)

```ini
OPENAI_API_KEY=
HF_TOKEN=
SERPAPI_API_KEY=
DEBUG=1
```

- `DEBUG` (0/1) controls verbose logging across modules.  
- `OPENAI_API_KEY` required if using the OpenAI provider.  
- `HF_TOKEN` only needed for gated HF models.  
- `SERPAPI_API_KEY` only if you wire up web search via SerpAPI (optional — feature flag in `agentic_features.py`).

---

## How It Works

### Ingestion & Chunking

- **Entry points**: 
  - Web uploads: POST `/upload` → files saved under `data/`.
  - Programmatic: `scripts/ingest.py: ingest_documents(paths)`.
- **Parsers** (`src/chunking/chunker.py`, `src/data_ingest/preprocess.py`):
  - `parse_pdf_to_chunks`, `parse_docx_to_chunks` create structured chunks with fields like `doc_name`, `page_index`, `page_label`, `modality (text/code/table/image)`, `caption_text`, `bounding_box`, `ingested_at`, `chunk_id`.
  - Optional GPT probes (layout, image captions, doc summary) are controlled via flags in `helper.py`.
  - TOC & page‑label helpers live in `preprocess.py` (e.g., `find_toc_pages`, `autodetect_page_numbering`, `extract_page_numbers_from_headers_footers`).

### Embeddings & Vector Stores

- **Embeddings**: `BAAI/bge-base-en-v1.5` via `langchain_community.embeddings.HuggingFaceEmbeddings` (normalized).  
- **Stores** (`src/embeddings/index/chroma_db.py`):
  - `docs_and_headings` — main chunk store
  - `doc_index` — doc‑level summaries
  - `chat_memory` — short past turns for personalization/context
- Persists in `"chroma_store"`; auto‑created on first run.

### Retrieval & Ranking

- **Doc retriever**: top‑3 from `doc_index`.  
- **Chunk retrievers**: 
  - **BM25**: `BM25Retriever.from_documents(all_docs, k=40)` (classical keyword scoring).
  - **Vector**: `Chroma(...).as_retriever(search_kwargs={'k': 20})`.  
- **Reranking**: SentenceTransformer (`BAAI/bge-base-en-v1.5`) cosine similarity re‑scores blended candidates (toggle via `reranker`).

### Reasoning, Memory & Output

- **Agent loop** (`src/pipelines/agentic.py`):
  - Decompose complex questions → target doc(s) → retrieve evidence → structure sections:
    **Answer**, **Sources**, **Requirements**, **External references**, **Subtopics**, **Next questions**.
  - **Memory** (`on("memory")`): `write_memory`, `read_memory`, `delete_memory` target the `chat_memory` collection; used to bias follow‑ups toward prior doc IDs.
  - Feature flags from `agentic_features.py` gate behaviors (citations, clause extraction, web search fallback, etc.).

---

## Web App Endpoints

| Route                 | Methods |
| --------------------- | ------- |
| /Docs/<path:filename> | GET     |
| /                     | GET     |
| /chatbot.html         | GET     |
| /files.html           | GET     |
| /about                | GET     |
| /upload               | POST    |
| /delete               | POST    |
| /files                | GET     |
| /chat                 | POST    |
| /chat/start           | POST    |
| /chat/sessions        | POST    |
| /chat/history         | POST    |
| /chat/rename          | POST    |
| /chat/delete          | POST    |

**Defaults**
- Upload folder: `data` (`src/web/app.py` → `UPLOAD_FOLDER`).
- Allowed extensions: `{".pdf", ".docx"}`.
- Serves `Docs/` at `/Docs/<filename>` with content‑type for PDF/DOCX.

---

## Common Tasks

- **Run the app**  
  ```bash
  python -m src.web.app
  ```

- **Ingest a folder programmatically**  
  ```python
  from pathlib import Path
  from scripts.ingest import ingest_documents
  paths = list(Path("Docs").glob("**/*.[pP][dD][fF]")) + list(Path("Docs").glob("**/*.docx"))
  ingest_documents(paths)
  ```

- **Delete files from index & disk**  
  ```python
  from pathlib import Path
  from scripts.delete import delete_docs_from_chroma
  delete_docs_from_chroma([Path("Docs/example.pdf")])
  ```

- **Toggle features at runtime**  
  ```python
  from src.pipelines.agentic import switch
  switch(memory=1, web_search=0, reranker=1)
  ```

---

## Troubleshooting

- **No answers / empty retrieval**  
  - Check that `Docs/` or `data/` contains supported files and ingestion logs show chunks added.  
  - Ensure `STORE_DIR` is writable and not corrupted; delete and rebuild if needed.

- **Citations look wrong**  
  - Verify `RUN_GPT_LAYOUT_PROBE` (header/footer/page‑label regex extraction) and that TOC/page‑label helpers can see consistent numbering.  
  - Adjust `MAX_TOKENS`/`OVERLAP` to avoid splitting paragraphs mid‑sentence.

- **Memory not influencing follow‑ups**  
  - Confirm `memory` is **on** in `agentic_features.py` and that `chat_id` is consistent between calls.  
  - Use `delete_memory(chat_id)` to reset; then ask a seed question and a targeted follow‑up.

- **Switch to local model**  
  - Set `LLM_PROVIDER = "anything-but-OPENAI"` in `src/configs/helper.py`, and ensure `LLM_NAME` is available (e.g., via HF / Ollama).

---

**Diagrams**  
See `architecture/` for the big‑picture and agentic flow diagrams.

---

**Maintainers’ checklist (when changing defaults)**

- Update tables in this README if you add/remove flags.  
- Keep `.env` keys minimal and optional where possible.  
- Prefer feature toggles in `agentic_features.py` over ad‑hoc if/else blocks.
