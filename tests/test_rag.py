import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Force console encoding to utf-8 for Windows
sys.stdout.reconfigure(encoding='utf-8')

from src.services.vectorstore import get_embedding_function, load_or_create_vectorstore
from src.core.config import RETRIEVER_SCORE_THRESHOLD, RETRIEVER_K

def main():
    print("="*60)
    print("RAG QUALITY DIAGNOSTICS")
    print("="*60)
    
    # Initialize Vectorstore
    print("Loading vectorstore...")
    embedding_func = get_embedding_function()
    vectorstore = load_or_create_vectorstore(embedding_func)
    print("Vectorstore loaded successfully.\n")

    queries = [
        "The headache is about 7/10, I haven't been sleeping well and I'm very stressed at work. No fainting or vision problems. It's usually worse in the evening.",
        "I have a headache",
        "fatigue and dizziness"
    ]
    
    print(f"Current RETRIEVER_SCORE_THRESHOLD: {RETRIEVER_SCORE_THRESHOLD}")
    print(f"Current RETRIEVER_K: {RETRIEVER_K}\n")

    for query in queries:
        print(f"\n--- QUERY: '{query}' ---")
        
        # We use similarity_search_with_relevance_scores to see the raw scores
        # before the threshold filter applies.
        try:
            results_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=10)
            
            print("Retrieved Documents (Top 10):")
            for i, (doc, score) in enumerate(results_with_scores):
                status = "ACCEPTED" if score >= RETRIEVER_SCORE_THRESHOLD else "DISCARDED (Below Threshold)"
                print(f"\n[{i+1}] Score: {score:.4f} -> {status}")
                print(f"    Source: {doc.metadata.get('source', 'Unknown')}")
                snippet = doc.page_content[:150].replace('\n', ' ')
                print(f"    Content: {snippet}...")
                
            print(f"\nNumber of docs that PASS threshold: {sum(1 for doc, score in results_with_scores if score >= RETRIEVER_SCORE_THRESHOLD)}")
            
        except Exception as e:
            print(f"Error during retrieval: {e}")

if __name__ == "__main__":
    main()
