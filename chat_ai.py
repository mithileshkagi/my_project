from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

def create_llm() -> BaseChatModel:
    """
    Initializes and returns the ChatGoogleGenerativeAI LLM.
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        print("ChatGoogleGenerativeAI LLM initialized.")
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None

# Example usage:
# llm = create_llm()
# if llm:
#     # test_response = llm.invoke("Hello, how are you?")
#     # print(f"LLM test response: {test_response.content}")
#     pass
