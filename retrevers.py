# from langchain_core.retrievers import Retriever
from langchain_core.retrievers import BaseRetriever

def create_retriever(vector_store) :
    """
    Creates a retriever from the given vector store.
    """
    if not vector_store:
        print("Vector store missing for retriever creation.")
        return None
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 similar documents
    print("Retriever created.")
    return retriever

# Example usage (assuming 'vector_store' from Step 4):
if vector_store:
    retriever = create_retriever(vector_store)
    if retriever:
        # You can test the retriever
        # retrieved_docs = retriever.invoke("What is the main topic of the document?")
        print(f"Retrieved {len(retrieved_docs)} documents.")
        pass
