from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore

def create_vector_store(chunks, embeddings) -> VectorStore:
    """
    Creates and populates a Chroma vector store with document chunks and embeddings.
    """
    if not chunks or not embeddings:
        print("Chunks or embeddings model missing for vector store creation.")
        return None
    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db" # Directory to persist the vector store
        )
        print("Vector store created and populated.")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

# # Example usage (assuming 'chunks' from Step 2 and 'embeddings_model' from Step 3):
# if chunks and embeddings_model:
#     vector_store = create_vector_store(chunks, embeddings_model)
    if vector_store:
        print("Vector store is ready.")
