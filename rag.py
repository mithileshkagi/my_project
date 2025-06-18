from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings


def create_rag_chain(retriever, llm):
    """
    Creates the RAG chain that combines retrieval and generation.
    """
    if not retriever or not llm:
        print("Retriever or LLM missing for RAG chain creation.")
        return None

    # Define the prompt template
    # The prompt instructs the LLM on how to use the retrieved context.
    prompt_template = """
    You are a helpful AI assistant. Answer the question based ONLY on the provided context.
    If the answer cannot be found in the context, politely state that you don't have enough information.

    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Construct the RAG chain
    # The chain uses LCEL to define the flow:
    # 1. 'context': The retriever fetches documents, and we format them.
    # 2. 'question': The original user query is passed through.
    # 3. These are then passed to the 'prompt' template.
    # 4. The 'prompt' is sent to the 'llm' for generation.
    # 5. The 'StrOutputParser' extracts the string content from the LLM's response.
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain created.")
    return rag_chain

# Example usage (assuming 'retriever' from Step 5 and 'llm' from Step 6):
# if retriever and llm:
#     rag_chain = create_rag_chain(retriever, llm)
#     if rag_chain:
#         # You can test the chain
#         # response = rag_chain.invoke("What is the capital of France?") # This will only use context if available
#         print(f"RAG chain test response: {response}")
#         pass