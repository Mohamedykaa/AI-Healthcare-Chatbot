import logging
import os
import re
import shutil
import time

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    LOG_LEVEL,
    MEDICAL_KNOWLEDGE_FILES,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def get_embedding_function() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def validate_chroma_db(vectorstore: Chroma) -> bool:
    try:
        collection = vectorstore._collection
        count = collection.count()
        logger.info("ChromaDB validation passed (%d documents)", count)
        return count > 0
    except Exception as exc:
        logger.warning("ChromaDB validation failed: %s", exc)
        return False


def normalize_text(text: str) -> str:
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def _is_valid_chroma_sqlite(db_path: str) -> bool:
    """Check if a SQLite file at *db_path* is a valid ChromaDB database.

    Validation rules (designed for chromadb 0.4.x):
    1. File must exist and be non-empty.
    2. File must be a valid SQLite database (header check).
    3. If the database contains a ``collections`` table, it is considered
       valid — the exact column set is left for ChromaDB itself to handle
       at load time, which prevents false-positive "incompatible" verdicts
       that previously triggered unnecessary re-ingestion.
    4. An empty SQLite file (no tables) is valid — it means a fresh DB.

    This function NEVER opens the file via the chromadb library, so it
    cannot acquire a file lock.
    """
    if not os.path.exists(db_path):
        return False

    # Empty file = not a valid SQLite database
    if os.path.getsize(db_path) == 0:
        return False

    import sqlite3
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        if not tables:
            # Empty database — valid (fresh init)
            return True

        # If collections table exists, the DB was created by ChromaDB
        if "collections" in tables:
            return True

        # Tables exist but no 'collections' → not a ChromaDB database
        logger.warning("Database has tables %s but no 'collections' table", tables)
        return False

    except sqlite3.DatabaseError as err:
        logger.warning("SQLite integrity check failed for %s: %s", db_path, err)
        return False
    finally:
        if conn:
            conn.close()


def _safe_remove_directory(dir_path: str, max_attempts: int = 3) -> bool:
    """Attempt to remove a directory, retrying on Windows file-lock errors.

    Returns True if the directory was successfully removed, False otherwise.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            shutil.rmtree(dir_path)
            return True
        except PermissionError as exc:
            logger.warning(
                "Attempt %d/%d to remove %s failed (file lock): %s",
                attempt, max_attempts, dir_path, exc,
            )
            if attempt < max_attempts:
                time.sleep(1)  # Brief pause for locks to release
    return False


def load_or_create_vectorstore(embedding_function: HuggingFaceEmbeddings) -> Chroma:
    """Load an existing ChromaDB or create a new one from medical knowledge files.

    Startup rules:
    1. If the persist directory exists and contains a valid SQLite DB,
       attempt to load it directly with ChromaDB.
    2. If ChromaDB loads and validates (has documents), return it.
    3. If the DB is corrupt or invalid, remove it and recreate.
    4. If no persist directory exists, create from scratch.
    """
    db_file = os.path.join(CHROMA_PERSIST_DIR, "chroma.sqlite3")

    if os.path.exists(CHROMA_PERSIST_DIR):
        logger.info("Found existing ChromaDB at %s", CHROMA_PERSIST_DIR)

        # Pre-flight SQLite check (read-only, no file locks)
        if os.path.exists(db_file) and not _is_valid_chroma_sqlite(db_file):
            logger.warning("ChromaDB SQLite file is invalid, removing directory...")
            if not _safe_remove_directory(CHROMA_PERSIST_DIR):
                logger.error(
                    "Cannot remove invalid database at %s — continuing with fresh creation",
                    CHROMA_PERSIST_DIR,
                )

    # Attempt to load existing DB
    if os.path.exists(CHROMA_PERSIST_DIR):
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embedding_function,
            )
            if validate_chroma_db(vectorstore):
                logger.info("ChromaDB loaded and validated successfully")
                return vectorstore
            else:
                logger.warning("ChromaDB is empty, will recreate from knowledge files")
                # Release resources before deleting
                _release_chroma_resources(vectorstore)
                _safe_remove_directory(CHROMA_PERSIST_DIR)
        except Exception as exc:
            logger.error("Error loading ChromaDB: %s", exc)
            if os.path.exists(CHROMA_PERSIST_DIR):
                _safe_remove_directory(CHROMA_PERSIST_DIR)

    # --- Create from scratch ---
    return _create_vectorstore_from_files(embedding_function)


def _release_chroma_resources(vectorstore: Chroma) -> None:
    """Best-effort release of ChromaDB file handles before directory deletion."""
    try:
        if hasattr(vectorstore, "_client"):
            client = vectorstore._client
            if hasattr(client, "close"):
                client.close()
            if hasattr(client, "_system") and hasattr(client._system, "stop"):
                client._system.stop()
    except Exception:
        pass

    import gc
    del vectorstore
    gc.collect()


def _create_vectorstore_from_files(embedding_function: HuggingFaceEmbeddings) -> Chroma:
    """Load medical knowledge files, split into chunks, and create a new ChromaDB."""
    logger.info("Loading medical knowledge from multiple sources...")
    all_documents = []
    for filepath in MEDICAL_KNOWLEDGE_FILES:
        if os.path.exists(filepath):
            logger.info("   Loading: %s", filepath)
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.page_content = normalize_text(doc.page_content)
                doc.metadata["source"] = filepath
            all_documents.extend(docs)
        else:
            logger.warning("   Not found (skip): %s", filepath)

    if not all_documents:
        raise FileNotFoundError("No medical knowledge files found.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n---\n", "\n\n", "\n", ". ", " ", ""],
    )
    splits = text_splitter.split_documents(all_documents)
    logger.info("Created %d text chunks", len(splits))

    BATCH_SIZE = 5000
    if len(splits) <= BATCH_SIZE:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            persist_directory=CHROMA_PERSIST_DIR,
        )
    else:
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embedding_function,
        )
        for i in range(0, len(splits), BATCH_SIZE):
            batch = splits[i : i + BATCH_SIZE]
            vectorstore.add_documents(batch)

    logger.info("ChromaDB created and persisted at %s", CHROMA_PERSIST_DIR)
    return vectorstore
