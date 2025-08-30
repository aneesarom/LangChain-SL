from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
parser = StrOutputParser()

# --------------------- ⚙️ Manual Chain Execution (Under the Hood) ---------------------

def custom_chain(placeholder):
    prompt_string = prompt.invoke(placeholder)
    response = model.invoke(prompt_string)
    output = parser.invoke(response)
    return output

prompt = PromptTemplate.from_template("Tell me about {person} in few words")
chain = prompt | model | parser
response = custom_chain({"person": "Elon Musk"})
# print(response)

# --------------------- 🔁 Chain with Placeholders (Dynamic Inputs) ---------------------

prompt = PromptTemplate.from_template("Tell me about {person} about {topic} in few words")
chain = prompt | model | parser
response = chain.invoke({"person": "Elon Musk", "topic": "personal life"})
print(response)

# --------------------- 🧾 Chain without Placeholders (Static Prompt) ---------------------

prompt = PromptTemplate.from_template("Tell me about Elon musk about his personal life in few words")
chain = prompt | model | parser
response = chain.invoke({})
# print(response)

