import ollama
import faiss
import pickle
import numpy as np
import re
from langchain_huggingface import HuggingFaceEmbeddings
from config import MODEL_NAME, TOP_K, EMBEDDING_MODEL, FAISS_INDEX_PATH, CHUNKS_PATH, TEMPERATURE

# Load FAISS index and chunks (do this once at startup)
print("🔄 Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

with open(CHUNKS_PATH, 'rb') as f:
    chunks = pickle.load(f)

embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)
print("✅ Ready to answer questions!\n")

def boost_filename_matches(question: str, relevant_chunks: list, distances: list) -> tuple:
    """
    Boost chunks whose filename matches keywords in question.
    Returns re-sorted chunks and distances.
    """
    # Extract keywords from question (words longer than 3 chars)
    keywords = re.findall(r'\b\w{4,}\b', question.lower())
    
    # Create list of (chunk, distance, boost_score)
    scored_chunks = []
    
    for chunk, dist in zip(relevant_chunks, distances):
        filename = chunk['source'].lower()
        boost = 0
        
        # Check if any keyword appears in filename
        for keyword in keywords:
            if keyword in filename:
                boost = -0.3  # Reduce distance (better ranking)
                break
        
        scored_chunks.append((chunk, dist + boost, dist))
    
    # Sort by boosted distance (lower is better)
    scored_chunks.sort(key=lambda x: x[1])
    
    # Unpack sorted results
    sorted_chunks = [item[0] for item in scored_chunks]
    sorted_distances = [item[2] for item in scored_chunks]  # Return original distances for display
    
    return sorted_chunks, sorted_distances

def answer_question(question: str, show_chunks: bool = False, use_boost: bool = True) -> None:
    """Answer a question using RAG with streaming response"""
    
    # Step 1: Convert question to vector
    question_vector = embeddings_model.embed_query(question)
    question_vector = np.array([question_vector])
    
    # Step 2: Search FAISS for similar chunks
    distances, indices = index.search(question_vector, TOP_K)
    
    # Step 3: Get relevant chunks
    relevant_chunks = [chunks[i] for i in indices[0]]
    chunk_distances = distances[0].tolist()
    
    # Step 4: Apply filename boosting (optional)
    if use_boost:
        relevant_chunks, chunk_distances = boost_filename_matches(
            question, relevant_chunks, chunk_distances
        )
    
    # Step 5: Build context with source information
    context_parts = []
    sources = []
    
    for i, (chunk, dist) in enumerate(zip(relevant_chunks, chunk_distances)):
        context_parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
        sources.append(f"{chunk['source']} (distance: {dist:.2f})")
    
    # Step 6: Show chunks if debug mode
    if show_chunks:
        print("\n" + "="*60)
        print("🔍 RETRIEVED CHUNKS (for debugging):")
        if use_boost:
            print("✨ Filename boosting: ENABLED")
        print("="*60)
        for i, (chunk, dist) in enumerate(zip(relevant_chunks, chunk_distances), 1):
            print(f"\n📄 Chunk {i} - {chunk['source']} (distance: {dist:.4f})")
            print(f"   {chunk['text'][:200]}...")
        print("="*60 + "\n")
    
    context = "\n\n".join(context_parts)
    
    # Step 7: Format prompt
    prompt = f"""You are a helpful assistant that answers questions based on provided documents.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the context above
- If the answer is not in the context, say "I don't have information about that in the available documents."
- Cite which document(s) you're referring to
- Be specific and accurate

Answer:"""
    
    # Step 8: Send to Ollama with streaming
    print(f"📚 Sources retrieved: {', '.join(sources)}\n")
    print("💬 Answer: ", end='', flush=True)
    
    try:
        stream = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": TEMPERATURE},
            stream=True
        )
        
        # Step 9: Stream the response
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        
        print("\n")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tip: Make sure Ollama is running! Run 'ollama serve' in another terminal.\n")

def search_all_chunks(question: str):
    """Show ALL chunks ranked by similarity"""
    question_vector = embeddings_model.embed_query(question)
    question_vector = np.array([question_vector])
    
    # Search for ALL chunks
    distances, indices = index.search(question_vector, len(chunks))
    
    print("\n" + "="*60)
    print("🔍 ALL CHUNKS RANKED:")
    print("="*60)
    
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        chunk = chunks[idx]
        print(f"{i:2d}. {chunk['source']:40} (distance: {dist:.4f})")
        if i == TOP_K:
            print("   " + "-"*56 + f" TOP {TOP_K} CUTOFF")
    
    print("="*60 + "\n")

def main():
    """Interactive Q&A loop"""
    print("=" * 60)
    print("🤖 AI Assistant - Ask questions about your documents")
    print("=" * 60)
    print("Commands:")
    print("  - Type your question normally")
    print("  - Type 'debug' before your question to see retrieved chunks")
    print("  - Type 'rank' before your question to see ALL chunks ranked")
    print("  - Type 'noboost' before question to disable filename boosting")
    print("  - Type 'quit' or 'exit' to stop\n")
    
    while True:
        user_input = input("❓ Your question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not user_input:
            print("⚠️  Please enter a question.\n")
            continue
        
        # Check for rank mode
        if user_input.lower().startswith('rank '):
            question = user_input[5:].strip()
            search_all_chunks(question)
            continue
        
        # Check for noboost mode
        use_boost = True
        if user_input.lower().startswith('noboost '):
            use_boost = False
            user_input = user_input[8:].strip()
        
        # Check for debug mode
        show_debug = False
        if user_input.lower().startswith('debug '):
            show_debug = True
            user_input = user_input[6:].strip()
        
        try:
            answer_question(user_input, show_chunks=show_debug, use_boost=use_boost)
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()