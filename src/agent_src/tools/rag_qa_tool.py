import logging

from crewai.tools import tool
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings
import chromadb

from src.agent_src.config.agent_settins import AgentSettings

logger = logging.getLogger(__name__)

logger.info("Loading HuggingFace embedding model...")
embed_model = HuggingFaceEmbedding()


@tool
def rag_query_tool(query: str) -> dict:
    """
    Query the RAG vector database and return an answer
    along with source file names.
    """

    settings = AgentSettings()

    vector_store_path = settings.VECTOR_STORE_DIR
    collection_name = settings.COLLECTION_NAME

    Settings.llm = Groq(
        model=settings.MODEL_NAME,
        temperature=settings.MODEL_TEMPERATURE,
        api_key=settings.GROQ_API_KEY,
    )

    db = chromadb.PersistentClient(path=vector_store_path)

    chroma_collection = db.get_or_create_collection(
        collection_name
    )

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )

    query_engine = index.as_query_engine(
        similarity_top_k=3
    )

    response = query_engine.query(query)

    metadata = getattr(response, "metadata", None) or {}

    source_file_names = {
        m.get("file_name")
        for m in metadata.values()
        if isinstance(m, dict)
    }

    return {
        "answer": response.response,
        "source_files": list(source_file_names)
    }

