import os
import re
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.core.config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    MEDICAL_KNOWLEDGE_FILES,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

def get_embedding_function() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def validate_chroma_db(vectorstore: Chroma) -> bool:
    try:
        results = vectorstore.similarity_search("health", k=1)
        collection = vectorstore._collection
        return collection.count() > 0
    except Exception as e:
        print(f"ChromaDB validation failed: {e}")
        return False

def normalize_text(text: str) -> str:
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def load_or_create_vectorstore(embedding_function: HuggingFaceEmbeddings) -> Chroma:
    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Found existing ChromaDB at {CHROMA_PERSIST_DIR}")
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embedding_function
            )
            if validate_chroma_db(vectorstore):
                print("ChromaDB loaded and validated successfully")
                return vectorstore
            else:
                print("ChromaDB validation failed, recreating...")
                shutil.rmtree(CHROMA_PERSIST_DIR)
        except Exception as e:
            print(f"Error loading ChromaDB: {e}")
            if os.path.exists(CHROMA_PERSIST_DIR):
                shutil.rmtree(CHROMA_PERSIST_DIR)
    
    print("Loading medical knowledge from multiple sources...")
    all_documents = []
    for filepath in MEDICAL_KNOWLEDGE_FILES:
        if os.path.exists(filepath):
            print(f"   Loading: {filepath}")
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.page_content = normalize_text(doc.page_content)
                doc.metadata["source"] = filepath
            all_documents.extend(docs)
        else:
            print(f"   Not found (skip): {filepath}")
    
    if not all_documents:
        raise FileNotFoundError("No medical knowledge files found.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n---\n", "\n\n", "\n", ". ", " ", ""]
    )
    splits = text_splitter.split_documents(all_documents)
    
    BATCH_SIZE = 5000
    if len(splits) <= BATCH_SIZE:
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function, persist_directory=CHROMA_PERSIST_DIR)
    else:
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_function)
        for i in range(0, len(splits), BATCH_SIZE):
            batch = splits[i:i + BATCH_SIZE]
            vectorstore.add_documents(batch)
    
    vectorstore.persist()
    print(f"ChromaDB created and persisted at {CHROMA_PERSIST_DIR}")
    return vectorstore
