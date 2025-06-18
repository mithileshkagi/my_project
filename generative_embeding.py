from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings

def create_embeddings() -> Embeddings:
    """
    Initializes and returns the Google Generative AI Embeddings model.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        print("Google Generative AI Embeddings model initialized.")
        return embeddings
    except Exception as e:
        print(f"Error initializing embeddings model: {e}")
        return None

# Example usage:
embeddings_model = create_embeddings()
if embeddings_model:
    # You can test it by embedding a simple text
    # test_embedding = embeddings_model.embed_query("Hello world")
    # print(f"Test embedding length: {len(test_embedding)}")
    pass
