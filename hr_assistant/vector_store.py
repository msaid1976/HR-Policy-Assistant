"""Step 4: store chunk embeddings in Qdrant Cloud so we can search them later."""


from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from hr_assistant import config
from hr_assistant.embeddings import get_embeddings_model
from hr_assistant.logger import get_logger

logger = get_logger(__name__)


# build_vector_store

def build_vector_store(chunks):
    """Embed every chunk and upload it into a Qdrant Cloud collection."""
    logger.info(
        "Embedding %d chunk(s) and uploading to Qdrant collection '%s'...",
        len(chunks),
        config.QDRANT_COLLECTION_NAME,
    )
    embeddings_model = get_embeddings_model()
    vector_store = QdrantVectorStore.from_documents(
        chunks,
        embedding=embeddings_model,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )
    logger.info("Uploaded to Qdrant collection '%s'", config.QDRANT_COLLECTION_NAME)
    return vector_store



# def build_vector_store_old(chunks) :
#     """Embed every chunk and build
#     a searchable FAISS index in memory."""
#     embeddings_model = get_embeddings_model()
#     return FAISS. from_documents(chunks, embeddings_model)



# def save_vector_store_old(vector_store, path: str = config.VECTOR_STORE_PATH):
#     """Save the FAISS index to disk The
#     so we don't have to rebuild it every time."""
#     vector_store.save_local(path)


# def load_vector_store_old(path: str = config.VECTOR_STORE_PATH) :
#     """Load a previously saved FAISS index from disk."""
#     embeddings_model = get_embeddings_model()
#     # allow_dangerous_deserialization is safe here because we only ever load
#     # an index that this same app created and saved.
#     return FAISS.load_local(path, embeddings_model, allow_dangerous_deserialization=True)

# def vector_store_exists(path: str = config.VECTOR_STORE_PATH) -> bool:
#     """Check if a saved FAISS index already exists on disk."""
#     return os.path.exists(os.path.join(path, "index.faiss"))
    


def load_vector_store():
    """Connect to a Qdrant Cloud collection that was already built before."""
    logger.info("Connecting to existing Qdrant collection '%s'", config.QDRANT_COLLECTION_NAME)
    embeddings_model = get_embeddings_model()
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )


def vector_store_exists() -> bool:
    """Check if the Qdrant Cloud collection already exists."""
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    return client.collection_exists(config.QDRANT_COLLECTION_NAME)


def get_retriever(vector_store, k: int = config.TOP_K_RESULTS):
    """Turn a vector store into a retriever
    that returns the top-k matching chunks."""
    logger.info("Creating retriever with top_k=%d", k)
    return vector_store.as_retriever(search_kwargs={"k": k})
