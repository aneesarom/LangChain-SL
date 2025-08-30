from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful AI assistant"),
    ("human", "who is {person}")
])

# ------------------------------ StrOutputParser ------------------------------------

parser = StrOutputParser()

chain = prompt_template | model | parser
# response = chain.invoke({"person": "ratan tata"})
# print(response)

# ------------------------------ Using Pydantic ------------------------------------

class Output(BaseModel):
    profession: str = Field(description="Profession of the person")
    achievements: List[str] = Field(description="Achievements of the person")

parser_model = model.with_structured_output(Output)

chain = prompt_template | parser_model
response = chain.invoke({"person": "ratan tata"})
print(response.model_dump_json())

