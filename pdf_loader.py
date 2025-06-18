# Import necessary libraries
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Define the path to your PDF document
# For demonstration, let's assume you have a PDF file named 'example.pdf'
# in the same directory as your script.
pdf_path = "data/budget_speech.pdf" # Replace with the actual path to your PDF

def load_documents(file_path):
    """
    Loads a PDF document from the given file path.
    """
    try:
        loader = PyPDFLoader("C:\INTERN\data\budget_speech.pdf")
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from {file_path}")
        return documents
    except Exception as e:
        print(f"Error loading document: {e}")
        return None

# Example usage:
documents = load_documents(pdf_path)
if documents:
    print(f"First 100 characters of the first page: {documents[0].page_content[:100]}")
