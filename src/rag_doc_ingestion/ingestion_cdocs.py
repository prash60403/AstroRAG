import logging

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.rag_doc_ingestion.config.doc_ingestion_settings import DocIngestionSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger= logging.getLogger(__name__)

settings= DocIngestionSettings()

logger.info("Starting document ingestion process...")

embed_model= HuggingFaceEmbedding()

def build_vector_store_from_documents():
    logger.info("starting vector store ingestion process...")

    try:
        docs_dir_path= settings.DOCUMENTS_DIR
        vector_store_path= settings.VECTOR_STORE_DIR
        collection_name= settings.COLLECTION_NAME
        logger.info("Loading documents from directory: %s", docs_dir_path)
        loader= SimpleDirectoryReader(docs_dir_path)
        documents= loader.load_data()
        #create Parser with chunking strategy
        parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=50)
        logger.info("Parsing Documents into Nodes")
        nodes= parser.get_nodes_from_documents(documents)
        logger.info("Parsed (len(nodes)) nodes")
        logger.info(f"Initializing ChromaDB persistent client at: {vector_store_path}")
        db= chromadb.PersistentClient(path= vector_store_path)
        # Create or retrieve the vector collection
        chroma_collection= db.get_or_create_collection(name= collection_name)
        logger.info(f"Creating Chroma vector store with collection name: {collection_name}")
        vector_store= ChromaVectorStore(chroma_collection=chroma_collection)
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.info("Building vector store index.")
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            vector_store=vector_store,
            embed_model=embed_model
        )
        logger.info("Vector store build successfully.")
        return 0
    except Exception as e:
        logger.error(f"Error during vector store build: {e}")
        return 1
if __name__ == "__main__":
    build_vector_store_from_documents()

