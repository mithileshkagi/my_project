import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents):
    """
    Splits the loaded documents into smaller chunks for processing.
    """
    if not documents:
        print("No documents to split.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # The maximum size of each chunk (in characters)
        chunk_overlap=200,    # The number of characters to overlap between chunks
        length_function=len,  # Function to calculate length (default is len)
        is_separator_regex=False, # Whether the separators are regex or not
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks

# Example usage (assuming 'documents' from Step 1):
# if documents:
#     chunks = split_documents(documents)
#     if chunks:
#         print(f"First chunk content: {chunks[0].page_content[:200]}...")

pdf_file_path = "budget_speech.pdf"
if os.path.exists(pdf_file_path):
    print("File found.")
else:
    print("File not found.")

