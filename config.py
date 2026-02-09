# Model settings
MODEL_NAME = "mistral"
TEMPERATURE = 0.5

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K = 10

# Paths
DATA_DIR = "data/"
FAISS_INDEX_PATH = "faiss_index/faiss_index.bin"
CHUNKS_PATH = "faiss_index/chunks.pkl"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"