from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# SystemMessage sets behavior.
# HumanMessage represents user input.
# AIMessage simulates previous model outputs.

# --------------SystemMessage & HumanMessage---------------

messages = [
    SystemMessage("You are a helpful AI assistant"),
    HumanMessage("Tell me about SpaceX"),
]

response = model.invoke(messages)
# print(response.content)

# --------------SystemMessage, HumanMessage & AIMessage---------------

messages = [
    SystemMessage("You are a helpful AI assistant"),
    HumanMessage("Tell me about SpaceX"),
    AIMessage("SpaceX is a groundbreaking company that has revolutionized the space industry. Its focus on innovation, reusability, and ambitious goals has made it a major player in space exploration and a driving force in shaping the future of humanity's presence in space."),
    HumanMessage("Tell me who is the founder of the company"),
]

response = model.invoke(messages)
# print(response.content)

# -------------- Messages as a String---------------

prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant"),
    ("human", "Tell me about SpaceX"),
    ("ai", "SpaceX is a groundbreaking company that has revolutionized the space industry. Its focus on innovation, reusability, and ambitious goals has made it a major player in space exploration and a driving force in shaping the future of humanity's presence in space."),
    ("human", "Tell me who is the founder of the company")])

prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant"),
    ("human", "Tell me about {topic} in few words")])

# print(prompt1.format_messages())
response = model.invoke(prompt1.format_messages())

chain = prompt2 | model
response = chain.invoke({"topic": "Twitter"})
# print(response.content)

chain = prompt1 | model
response = chain.invoke({})
print(response.content)

