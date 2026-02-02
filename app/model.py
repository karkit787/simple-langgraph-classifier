from langchain_google_genai import ChatGoogleGenerativeAI
from .schema import ClassifierState

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

confidence_threshold = 0.7