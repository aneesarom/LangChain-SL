from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# model invoke accepts only str or list
# it wont accept PromptTemplate, ChatPromptTemplate, dict directly

# ------------------String input-------------------
response = model.invoke("Explain NN in simple words")
# print(response.content)


# ------------------List input-------------------
messages = [
    SystemMessage("You are a helpful AI assistant"),
    HumanMessage("Tell me about SpaceX"),
]

response = model.invoke(messages)
print(response.content)