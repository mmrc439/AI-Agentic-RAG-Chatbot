from __future__ import annotations

from typing import List, Dict, Tuple, Any
from pathlib import Path
import json
import hashlib
import time

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.chunking.chunker import (
    _collect_doc_text_pieces,
    parse_docx_to_chunks,
    parse_pdf_to_chunks,
    to_documents_and_ids,
)
from src.data_ingest.preprocess import (
    build_doc_summary_chunk_gpt,
    now_utc_iso,
    sha256_file,  # ← early duplicate skip
)
from src.embeddings.index.chroma_db import vs
from src.configs.helper import dbg, STORE_DIR


# ----------------------- Manifest schema versioning ---------------------------
# Bump this when your chunking / ingestion schema meaningfully changes.
# Files ingested under older manifest_version will be *re-ingested once*,
# then skipped on subsequent runs.
MANIFEST_SCHEMA_VERSION = 2

# ----------------------- Globals (kept for parity) ---------------------------
all_docs: List[Document] = []
bm25_retriever = None
chunk_retriever = None

# ----------------------- Manifest paths & helpers -----------------------------
_MANIFEST_DIR = Path(STORE_DIR)
_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
_MANIFEST_PATH = _MANIFEST_DIR / "manifest.jsonl"


def _atomic_append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON line. (Simple append; good enough for single-process.)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_manifest_hashes(path: Path) -> Dict[str, int]:
    """
    Return mapping: file_hash -> max(manifest_version) seen in manifest.

    This lets us:
      - Re-ingest docs when MANIFEST_SCHEMA_VERSION increases.
      - Skip docs only if manifest_version >= MANIFEST_SCHEMA_VERSION.
    """
    seen: Dict[str, int] = {}
    if not path.exists():
        return seen
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = (ln or "").strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                fh = (obj.get("file_hash") or "").strip()
                if not fh:
                    continue
                ver_raw = obj.get("manifest_version", 1)
                try:
                    ver = int(ver_raw)
                except Exception:
                    ver = 1
                prev = seen.get(fh, 0)
                if ver > prev:
                    seen[fh] = ver
    except Exception as e:
        dbg(f"[INGEST] manifest read error: {e}")
    return seen


def _manifest_counts_for_doc(chunks: List[dict]) -> Dict[str, Any]:
    """Compute simple counts by modality for a single doc’s chunks."""
    out: Dict[str, int] = {}
    for ch in chunks:
        mod = (ch.get("modality") or "text").lower()
        out[mod] = out.get(mod, 0) + 1
    total = sum(out.values())
    out2 = {"total_chunks": total, "by_modality": out}
    return out2


# ----------------------- Table → text surrogate (A4) --------------------------
def _make_table_text_surrogates(chunks: List[dict], *, max_rows: int = 8) -> List[dict]:
    """
    Build plain-text surrogates for table chunks to improve keyword & embedding recall.
    - Each surrogate is a new 'text' chunk with `surrogate_kind="table_text"` and
      `parent_chunk_id` pointing to the source table.
    - Added conservatively: up to `max_rows` examples per table.
    """
    surrogates: List[dict] = []
    for ch in chunks:
        if (ch.get("modality") or "").lower() != "table":
            continue
        content = ch.get("content", "")
        try:
            payload = json.loads(content) if isinstance(content, str) else (content or {})
        except Exception:
            payload = {}

        cols = [str(c) for c in (payload.get("columns") or [])]
        rows = payload.get("rows") or []
        rows = rows[:max_rows]  # cap examples

        # Build a compact, readable text block
        header_line = " | ".join(cols) if cols else ""
        body_lines = []
        for r in rows:
            body_lines.append(" | ".join(str(c) for c in r))

        # Include some context from the chunk’s section/heading if present
        heading_path = (
            ch.get("heading_path")
            or ch.get("section_path")
            or ch.get("section")
            or "Document"
        )
        table_caption = ch.get("caption_text") or ""
        page_label = ch.get("page_label") or ch.get("page_index") or "?"

        text_parts = [f"TABLE (p. {page_label}) — {heading_path}"]
        if table_caption:
            text_parts.append(f"Caption: {table_caption}")
        if header_line:
            text_parts.append(f"Columns: {header_line}")
        if body_lines:
            text_parts.append("Rows:\n" + "\n".join(body_lines))

        text_block = "\n".join(text_parts).strip()
        if not text_block:
            continue

        # Derive a stable deterministic surrogate ID based on parent chunk
        parent_cid = (
            ch.get("chunk_id")
            or ch.get("chunk_hash")
            or hashlib.sha1(text_block.encode("utf-8")).hexdigest()
        )
        doc_id = ch.get("doc_id") or ""
        page_label_str = str(page_label)
        surrogate_id = (
            f"{doc_id}::{page_label_str}::tbltxt::"
            f"{hashlib.sha1(parent_cid.encode('utf-8')).hexdigest()[:8]}"
        )

        # Clone key metadata; mark as text surrogate
        md = dict(ch)
        md.update({
            "modality": "text",
            "kind": "table_text_surrogate",
            "content": text_block,
            "caption_text": None,
            "bounding_box": None,
            "parent_chunk_id": parent_cid,
            "surrogate_kind": "table_text",
            "chunk_id": surrogate_id,
            "chunk_hash": surrogate_id,
        })
        surrogates.append(md)

    return surrogates


# ----------------------- Core ingest ------------------------------------------------
def ingest_documents(paths: List[Path]) -> int:
    """
    Ingest a batch of documents.
    - Skips duplicates by file_hash (manifest-driven + schema version).
    - Re-ingests docs if their manifest_version < MANIFEST_SCHEMA_VERSION.
    - Emits doc summary chunks via GPT (if configured).
    - Adds table→text surrogates to both BM25 and vector store.
    - Writes ingestion manifest entries per document (B1).
    - Returns total number of chunks ingested into the vector store.
    """
    global all_docs, bm25_retriever, chunk_retriever

    if not paths:
        return 0

    # Load previously seen file hashes and their manifest_versions
    seen_hashes: Dict[str, int] = _load_manifest_hashes(_MANIFEST_PATH)

    base_chunks: List[dict] = []
    files_processed = 0
    files_skipped_duplicate = 0

    # ── 1) Parse documents into base chunks (+ early duplicate skipping) ──────
    for p in paths:
        # 0) extension gate (keep behavior explicit)
        ext = p.suffix.lower()
        if ext not in (".pdf", ".docx"):
            # NOTE: .doc (legacy) is not supported here; convert to .docx before ingest.
            dbg(f"[INGEST] skipping unsupported {p.name} (ext={ext})")
            continue

        # 1) EARLY duplicate skip via file hash (cheap I/O vs. full parse)
        try:
            fhash = sha256_file(p)
        except Exception as e:
            dbg(f"[INGEST] failed to hash {p.name}: {e}")
            fhash = ""

        # Skip only if this file has already been ingested with
        # manifest_version >= current MANIFEST_SCHEMA_VERSION
        if fhash and seen_hashes.get(fhash, 0) >= MANIFEST_SCHEMA_VERSION:
            files_skipped_duplicate += 1
            dbg(
                f"[INGEST] duplicate (schema >= v{MANIFEST_SCHEMA_VERSION}) "
                f"pre-parse (hash={fhash[:8]}…) -> skipping {p.name}"
            )
            continue

        # 2) Parse document into chunks
        try:
            if ext == ".pdf":
                parsed = parse_pdf_to_chunks(str(p))
            else:
                parsed = parse_docx_to_chunks(str(p))
        except Exception as e:
            dbg(f"[INGEST] failed on {p.name}: {e}")
            continue

        if not parsed:
            dbg(f"[INGEST] no chunks parsed from {p.name}")
            continue

        # Trust parser’s file_hash (should match early hash; if not, we still use parser one)
        file_hash = parsed[0].get("file_hash", fhash)

        # Duplicate protection (manifest-based; covers case when early hash was unavailable)
        if file_hash and seen_hashes.get(file_hash, 0) >= MANIFEST_SCHEMA_VERSION:
            files_skipped_duplicate += 1
            dbg(
                f"[INGEST] duplicate (schema >= v{MANIFEST_SCHEMA_VERSION}) "
                f"post-parse (hash={file_hash[:8]}…) -> skipping {p.name}"
            )
            continue

        base_chunks.extend(parsed)
        files_processed += 1

    if not base_chunks:
        if files_skipped_duplicate > 0:
            dbg(
                f"[INGEST] nothing new (all inputs already ingested "
                f"at schema >= v{MANIFEST_SCHEMA_VERSION}). "
                f"Skipped {files_skipped_duplicate} file(s)."
            )
        else:
            dbg("[INGEST] no new chunks to add")
        return 0

    # ── 2) Group base chunks by (doc_name, file_hash) ─────────────────────────
    by_doc_base: Dict[Tuple[str, str], List[dict]] = {}
    for ch in base_chunks:
        key = (ch.get("doc_name", ""), ch.get("file_hash", ""))
        by_doc_base.setdefault(key, []).append(ch)

    # ── 3) Build table→text surrogates per doc (A4) ───────────────────────────
    surrogates_by_doc: Dict[Tuple[str, str], List[dict]] = {}
    for key, items in by_doc_base.items():
        surr = _make_table_text_surrogates(items, max_rows=8)
        if surr:
            surrogates_by_doc[key] = surr

    # ── 4) Build doc summary chunks per doc ───────────────────────────────────
    summaries_by_doc: Dict[Tuple[str, str], dict] = {}
    for key, items_base in by_doc_base.items():
        doc_name, file_hash = key
        # Combine base + surrogates for this doc only
        doc_surrogates = surrogates_by_doc.get(key, [])
        items_all_for_summary = items_base + doc_surrogates

        ingested_at = items_all_for_summary[0].get("ingested_at") or now_utc_iso()
        pieces = _collect_doc_text_pieces(items_all_for_summary, doc_name)
        sum_chunk = build_doc_summary_chunk_gpt(doc_name, file_hash, ingested_at, pieces)
        if sum_chunk:
            summaries_by_doc[key] = sum_chunk

    # ── 5) Assemble final chunk set per doc: base + surrogates + summary ──────
    all_chunks: List[dict] = []
    for key, items_base in by_doc_base.items():
        doc_surrogates = surrogates_by_doc.get(key, [])
        sum_chunk = summaries_by_doc.get(key)
        doc_chunks: List[dict] = []
        doc_chunks.extend(items_base)
        doc_chunks.extend(doc_surrogates)
        if sum_chunk:
            doc_chunks.append(sum_chunk)
        all_chunks.extend(doc_chunks)

    total_surrogates = sum(len(v) for v in surrogates_by_doc.values())
    if total_surrogates:
        dbg(f"[INGEST] added {total_surrogates} table-text surrogate chunk(s)")
    if summaries_by_doc:
        dbg(f"[INGEST] added {len(summaries_by_doc)} summary chunk(s)")

    # ── 6) Build LangChain Documents + ids from all chunks ────────────────────
    docs, ids = to_documents_and_ids(all_chunks)
    if not docs:
        dbg("[INGEST] nothing convertible to Documents")
        return 0

    # ── 7) Add to vector store & persist (batched, with simple retries) ───────
    try:
        BATCH = 100
        for i in range(0, len(docs), BATCH):
            _docs = docs[i:i + BATCH]
            _ids = ids[i:i + BATCH]
            if not _docs:
                continue

            attempts = 0
            while True:
                try:
                    vs.add_documents(_docs, ids=_ids)
                    break
                except Exception as e:
                    attempts += 1
                    if attempts >= 3:
                        raise
                    dbg(
                        f"[INGEST] vector store add failed "
                        f"(batch {i // BATCH + 1}), retry {attempts}: {e}"
                    )
                    time.sleep(0.5 * attempts)  # simple backoff
        vs.persist()
    except Exception as e:
        dbg(f"[INGEST] vector store add failed: {e}")
        # Bubble up: caller may want to surface failure for CI/ops.
        raise

    # ── 8) Update in-memory lists & retrievers ────────────────────────────────
    all_docs.extend(docs)
    try:
        # BM25 on *all* docs (including surrogates & summaries)
        bm25_retriever = BM25Retriever.from_documents(all_docs, k=40)
    except Exception as e:
        dbg(f"[INGEST] BM25 retriever build failed: {e}")

    try:
        # Vector retriever on chunks (the embeddings store already has all docs)
        chunk_retriever = vs.as_retriever(search_kwargs={"k": 20})
    except Exception as e:
        dbg(f"[INGEST] Vector retriever build failed: {e}")

    # ── 9) Write manifest entries (B1) per document just ingested ─────────────
    new_file_hashes_written = 0
    for key, items_base in by_doc_base.items():
        doc_name, file_hash = key
        if not file_hash:
            continue

        # Avoid duplicate manifest lines at the *current* schema version
        if seen_hashes.get(file_hash, 0) >= MANIFEST_SCHEMA_VERSION:
            continue

        # Build the full set of chunks for this doc to compute accurate counts
        doc_surrogates = surrogates_by_doc.get(key, [])
        sum_chunk = summaries_by_doc.get(key)
        doc_chunks_for_manifest: List[dict] = []
        doc_chunks_for_manifest.extend(items_base)
        doc_chunks_for_manifest.extend(doc_surrogates)
        if sum_chunk:
            doc_chunks_for_manifest.append(sum_chunk)

        # doc_id is uniform across chunks of a doc (set by chunker)
        doc_id = doc_chunks_for_manifest[0].get("doc_id") or ""
        counts = _manifest_counts_for_doc(doc_chunks_for_manifest)
        rec = {
            "doc_name": doc_name,
            "file_hash": file_hash,
            "doc_id": doc_id,
            "total_chunks": counts.get("total_chunks", len(doc_chunks_for_manifest)),
            "by_modality": counts.get("by_modality", {}),
            "ingested_at": (
                doc_chunks_for_manifest[0].get("ingested_at") or now_utc_iso()
            ),
            "manifest_version": MANIFEST_SCHEMA_VERSION,
        }
        try:
            _atomic_append_jsonl(_MANIFEST_PATH, rec)
            new_file_hashes_written += 1
            # Keep in-memory map in sync so we don't write twice in this run
            seen_hashes[file_hash] = MANIFEST_SCHEMA_VERSION
        except Exception as e:
            dbg(f"[INGEST] manifest write error for {doc_name}: {e}")

    dbg(
        f"[INGEST] +{len(docs)} chunks from {files_processed} file(s) "
        f"(skipped duplicates at schema >= v{MANIFEST_SCHEMA_VERSION}: {files_skipped_duplicate}; "
        f"manifest wrote {new_file_hashes_written})"
    )

    # === VERIFY VECTOR STORE === (best-effort; won’t crash ingest) ============
    try:
        # Prefer asking the live 'vs' object if it exposes a count
        count = None
        if hasattr(vs, "_collection") and hasattr(vs._collection, "count"):
            count = vs._collection.count()  # type: ignore[attr-defined]
        elif hasattr(vs, "collection") and hasattr(vs.collection, "count"):
            count = vs.collection.count()   # type: ignore[attr-defined]
        if count is not None:
            dbg(f"[DEBUG] ✅ Stored chunks in Chroma: {count}")
    except Exception as e:
        dbg(f"[DEBUG] ℹ️ Could not verify Chroma store count: {e}")

    # Optional BM25 retriever check (best-effort)
    try:
        dbg(f"[DEBUG] ✅ BM25 docs in memory: {len(all_docs)}")
    except Exception as e:
        dbg(f"[DEBUG] ℹ️ BM25 retriever not available: {e}")

    return len(docs)
