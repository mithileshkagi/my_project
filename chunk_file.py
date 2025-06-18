from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Initialize variables first
chunks = []  # ✅ Define chunks before use
embeddings_model = None  # You must initialize this properly later

# Load documents
try:
    with open('sample.txt', 'r') as file:
        text = file.read()
        documents = [Document(page_content=doc) for doc in text.split('\n\n')]
except FileNotFoundError:
    print("Error: sample.txt not found.")
    documents = []

# Split documents
if documents:
    def split_documents(documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        return splitter.split_documents(documents)

    chunks = split_documents(documents)

# Dummy example (replace with your actual embedding model setup)
embeddings_model = "some_model_instance"

# ✅ Safe to use this check now
if chunks and embeddings_model:
    print("Proceeding with vector store creation...")
else:
    print("Chunks or embedding model not available.")
