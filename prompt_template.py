from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langsmith import Client
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# -------------Prompt Template---------------------

### Not recommended
prompt = PromptTemplate(template="Tell me about {person} in few words")

### Recommeded
prompt = PromptTemplate.from_template("Tell me about {person} about {topic} in few words")
response = prompt.invoke({"person": "Elon Musk", "topic": "SpaceX"})
# print(response)

# -------------ChatPromptTemplate---------------------

### ChatPromptTemplate : Multi-message formatting (system, human, AI roles), It is essential when 
### working with chat models
### By default it takes as Human Message
prompt = ChatPromptTemplate.from_template("Tell me about {person} about {topic} in few words")
response = prompt.invoke({"person": "Elon Musk", "topic": "SpaceX"})
# print(response)

# -------------ChatPromptTemplate Using Messages---------------------

### Not recommended
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are helpful AI assistant"),
    HumanMessage(content="Tell me about {person} about {topic} in few words")
])

### Recommeded
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful AI assistant"),
    ("human", "Tell me about {person} about {topic} in few words")
])

response = prompt.invoke({"person": "Elon Musk", "topic": "SpaceX"})
print(response)

# -------------PromptTemplate From Hub---------------------

client = Client()
prompt = client.pull_prompt("hyenaman263/react-med", include_model=True)
# print(prompt)

