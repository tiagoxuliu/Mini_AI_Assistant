import os
from qa import main as qa_main

if __name__ == "__main__":
    # Check if FAISS index exists
    if not os.path.exists("faiss_index/faiss_index.bin"):
        print("⚠️  FAISS index not found!")
        print("Please run 'python ingest.py' first to process your documents.\n")
    else:
        qa_main()