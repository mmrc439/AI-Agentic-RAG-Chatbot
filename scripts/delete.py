from __future__ import annotations

from pathlib import Path
from typing import List, Set
import json

from chromadb import Settings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.embeddings.index.chroma_db import vs
from src.configs.helper import dbg, STORE_DIR, DEVICE
import scripts.ingest as ingest  # <- ingest.py lives in scripts/


def _rewrite_manifest_without_doc_names(doc_names: List[str]) -> None:
    """
    Remove manifest entries for the given doc_names so that:
      - They don't show up as 'already ingested' on future runs.
      - Re-uploads of the same file will be ingested again.
    """
    manifest_path = ingest._MANIFEST_PATH  # defined in scripts/ingest.py
    if not manifest_path.exists():
        return

    keep_lines: List[str] = []
    removed_hashes: Set[str] = set()

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for ln in f:
                line = (ln or "").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    # If it's not valid JSON, keep it as-is
                    keep_lines.append(ln)
                    continue

                if obj.get("doc_name") in doc_names:
                    fh = obj.get("file_hash")
                    if fh:
                        removed_hashes.add(fh)
                    # skip this line → effectively delete it
                    continue

                keep_lines.append(ln)

        tmp = manifest_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(keep_lines)
        tmp.replace(manifest_path)

        if removed_hashes:
            dbg(f"[DELETE] cleaned manifest for {len(removed_hashes)} file_hash(es)")
    except Exception as e:
        dbg(f"[DELETE] manifest rewrite failed: {e}")


def _delete_from_doc_index(doc_names: List[str]) -> None:
    """
    Also remove matching entries from the doc-level index collection ("doc_index"),
    which is used for doc selection. This prevents deleted docs from being chosen
    as candidates.
    """
    if not doc_names:
        return

    sources = [Path(n).stem for n in doc_names]  # doc_index uses 'source' = stem
    settings = Settings(
        is_persistent=True,
        persist_directory=str(STORE_DIR),
        anonymized_telemetry=False,
    )

    # Recreate embeddings (deletion is infrequent, so perf cost is fine)
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    try:
        doc_vs = Chroma(
            collection_name="doc_index",
            embedding_function=emb,
            client_settings=settings,
        )
        doc_vs._collection.delete(where={"source": {"$in": sources}})
        dbg(f"[DELETE] removed doc_index entries for sources={sources}")
    except Exception as e:
        dbg(f"[DELETE] doc_index delete failed: {e}")


def delete_docs_from_chroma(paths: List[Path]) -> None:
    """
    Delete documents from:
      • main chunk collection ("docs_and_headings") via doc_name
      • doc-level index ("doc_index") via source (stem)
      • ingestion manifest (so they can be re-ingested later)
      • local filesystem (the actual file)
    """
    if not paths:
        return

    doc_names = [p.name for p in paths]

    # 1) Delete from main chunk collection
    try:
        vs._collection.delete(where={"doc_name": {"$in": doc_names}})
        dbg(f"[DELETE] removed chunks for doc_name in {doc_names}")
    except Exception as e:
        dbg(f"[DELETE] vs delete failed: {e}")

    # 2) Delete from doc_index collection (doc-level index)
    _delete_from_doc_index(doc_names)

    # 3) Rewrite manifest to drop these docs
    _rewrite_manifest_without_doc_names(doc_names)

    # 4) Delete files from disk
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except Exception as e:
            dbg(f"[DELETE] failed to remove file {p}: {e}")
