import logging
import os
import re
import shutil

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
        vectorstore.similarity_search("health", k=1)
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


def load_or_create_vectorstore(embedding_function: HuggingFaceEmbeddings) -> Chroma:
    if os.path.exists(CHROMA_PERSIST_DIR):
        logger.info("Found existing ChromaDB at %s", CHROMA_PERSIST_DIR)
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embedding_function,
            )
            if validate_chroma_db(vectorstore):
                logger.info("ChromaDB loaded and validated successfully")
                return vectorstore
            else:
                logger.warning("ChromaDB validation failed, recreating...")
                shutil.rmtree(CHROMA_PERSIST_DIR)
        except Exception as exc:
            logger.error("Error loading ChromaDB: %s", exc)
            if os.path.exists(CHROMA_PERSIST_DIR):
                shutil.rmtree(CHROMA_PERSIST_DIR)

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
