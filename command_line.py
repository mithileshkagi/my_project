import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Re-define functions for a single executable block ---

def load_documents(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from {file_path}")
        return documents
    except Exception as e: 
        print(f"Error loading document: {e}")
        return None

def split_documents(documents):
    if not documents:
        print("No documents to split.")
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks

def create_embeddings():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        print("Google Generative AI Embeddings model initialized.")
        return embeddings
    except Exception as e:
        print(f"Error initializing embeddings model: {e}")
        return None

def create_vector_store(chunks, embeddings):
    if not chunks or not embeddings:
        print("Chunks or embeddings model missing for vector store creation.")
        return None
    try:
        vector_store = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory="./chroma_db"
        )
        print("Vector store created and populated.")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def create_retriever(vector_store):
    if not vector_store:
        print("Vector store missing for retriever creation.")
        return None
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    print("Retriever created.")
    return retriever

def create_llm():
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        print("ChatGoogleGenerativeAI LLM initialized.")
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None

def create_rag_chain(retriever, llm):
    if not retriever or not llm:
        print("Retriever or LLM missing for RAG chain creation.")
        return None
    prompt_template = """
    You are a helpful AI assistant. Answer the question based ONLY on the provided context.
    If the answer cannot be found in the context, politely state that you don't have enough information.

    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain created.")
    return rag_chain

# --- Main Chatbot Logic ---

if __name__ == "__main__":
    load_dotenv() # Ensure .env is loaded at the start

    pdf_file_path = "example.pdf" # Make sure this PDF exists in your directory!
                                 # Or provide a full path: "/path/to/your/example.pdf"

    # Step 1: Load documents
    documents = load_documents(pdf_file_path)
    if not documents:
        print("Exiting: Could not load documents.")
        exit()

    # Step 2: Split documents
    chunks = split_documents(documents)
    if not chunks:
        print("Exiting: Could not split documents.")
        exit()

    # Step 3: Create embeddings
    embeddings_model = create_embeddings()
    if not embeddings_model:
        print("Exiting: Could not initialize embeddings model.")
        exit()

    # Step 4: Create vector store (or load if already persisted)
    # Check if a persisted ChromaDB exists to avoid re-embedding every time
    if os.path.exists("./chroma_db"):
        print("Loading existing vector store...")
        try:
            vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)
            print("Vector store loaded from disk.")
        except Exception as e:
            print(f"Error loading existing vector store, re-creating: {e}")
            vector_store = create_vector_store(chunks, embeddings_model)
    else:
        print("Creating new vector store...")
        vector_store = create_vector_store(chunks, embeddings_model)

    if not vector_store:
        print("Exiting: Could not create/load vector store.")
        exit()

    # Step 5: Create retriever
    retriever = create_retriever(vector_store)
    if not retriever:
        print("Exiting: Could not create retriever.")
        exit()

    # Step 6: Create LLM
    llm = create_llm()
    if not llm:
        print("Exiting: Could not initialize LLM.")
        exit()

    # Step 7: Create RAG chain
    rag_chain = create_rag_chain(retriever, llm)
    if not rag_chain:
        print("Exiting: Could not create RAG chain.")
        exit()

    print("\nPDF Chatbot is ready! Ask your questions about the document.")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() == 'exit':
            print("Exiting chatbot. Goodbye!")
            break

        if user_query.strip():
            try:
                response = rag_chain.invoke(user_query)
                print(f"Chatbot: {response}")
            except Exception as e:
                print(f"An error occurred during response generation: {e}")
        else:
            print("Please enter a question.")
