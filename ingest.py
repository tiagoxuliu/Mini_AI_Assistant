from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import pickle
import os
import numpy as np
from PyPDF2 import PdfReader
from config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR, FAISS_INDEX_PATH, CHUNKS_PATH, EMBEDDING_MODEL

def load_documents():
    """Load all .txt and .pdf files from data directory"""
    documents = []
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR} folder. Please add your documents there.")
        return documents
    
    files = os.listdir(DATA_DIR)
    
    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        
        try:
            if filename.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({"content": content, "source": filename})
                    print(f"✅ Loaded: {filename}")
            
            elif filename.endswith('.pdf'):
                reader = PdfReader(filepath)
                content = ""
                for page in reader.pages:
                    content += page.extract_text()
                documents.append({"content": content, "source": filename})
                print(f"✅ Loaded: {filename}")
        
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
    
    return documents

def create_chunks(documents):
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks_with_metadata = []
    
    for doc in documents:
        chunks = splitter.split_text(doc["content"])
        for chunk in chunks:
            chunks_with_metadata.append({
                "text": chunk,
                "source": doc["source"]
            })
    
    return chunks_with_metadata

def create_embeddings(chunks):
    """Create embeddings for all chunks"""
    print("\n🔄 Creating embeddings... (this may take a minute)")
    
    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
        # ✅ REMOVE trust_remote_code line
    )
    
    embeddings_list = []
    chunk_texts = []
    
    for i, chunk in enumerate(chunks):
        vector = embeddings_model.embed_query(chunk["text"])
        embeddings_list.append(vector)
        chunk_texts.append(chunk)
        
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(chunks)} chunks...")
    
    return np.array(embeddings_list), chunk_texts


def save_to_faiss(embeddings, chunks):
    """Save embeddings to FAISS index"""
    # Create faiss_index directory if it doesn't exist
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    # Save chunks metadata
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"\n✅ Successfully saved {len(chunks)} chunks to FAISS index")

def main():
    print("=" * 50)
    print("📚 Document Ingestion Pipeline")
    print("=" * 50)
    
    # Step 1: Load documents
    print("\n📖 Loading documents...")
    documents = load_documents()
    
    if not documents:
        print("\n⚠️  No documents found. Please add .txt or .pdf files to the data/ folder.")
        return
    
    print(f"\n✅ Loaded {len(documents)} documents")
    
    # Step 2: Create chunks
    print("\n✂️  Splitting into chunks...")
    chunks = create_chunks(documents)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Step 3: Create embeddings
    embeddings, chunk_metadata = create_embeddings(chunks)
    
    # Step 4: Save to FAISS
    print("\n💾 Saving to FAISS index...")
    save_to_faiss(embeddings, chunk_metadata)
    
    print("\n" + "=" * 50)
    print("✅ Ingestion complete! You can now run qa.py")
    print("=" * 50)

if __name__ == "__main__":
    main()