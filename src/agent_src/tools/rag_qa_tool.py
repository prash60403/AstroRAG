import logging

from crewai.tools import tool
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings
import chromadb

from src.agent_src.config.agent_settins import AgentSettings

# Get a logger for this module
logger = logging.getLogger(__name__)

# download & load embedding model
logger.info("Loading HuggingFace embedding model...")
embed_model = HuggingFaceEmbedding()


@tool
def rag_query_tool(query: str) -> dict:
    """
    Answers a query by retrieving relevant documents and generating a response.
    Returns both the generated answer and the source file names from which the information was retrieved.

    Args:
        query (str): The input query string to be processed.

    Returns:
        dict: A dictionary with the following keys:
            - 'answer': The generated answer string.
            - 'source_files': List of source file names used for retrieval.

    Notes:
        - Requires properly configured AgentSettings and access to the vector store.
        - The function loads the embedding model and LLM each time it is called.
    """
    settins= AgentSettings()
    vector_store_path= settins.VECTOR_STORE_DIR
    collection_name= settins.COLLECTION_NAME

    #configure LLM settings
    Settings.llm = Groq(
        model=settins.MODEL_NAME,
        temperature=settins.MODEL_TEMPERATURE,
        api_key=settins.GROQ_API_KEY,
    )

    #load Chroma vector store
    db= chromadb.PersistentClient(path= vector_store_path)
    chroma_collection= db.get_or_create_collection(collection_name)

    #connect to the vectore store 
    vector_store= ChromaVectorStore(chroma_collection= chroma_collection)
    storage_context= StorageContext.from_defaults(vector_store= vector_store)

    # Load index from Chroma
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )
    # Create the query engine
    query_engine = index.as_query_engine(similarity_top_k=3)
    # Pass the query to the query engine
    response = query_engine.query(query)
    source_file_names = {m.get("file_name") for m in getattr(response, "metadata", {}).values()}

    return {"answer": response.response,
            "source_files": list(source_file_names)}

output = rag_query_tool(query="Explain Ecosystem and Evolution.")
print(output)
print(output["answer"])
print(output["source_files"])