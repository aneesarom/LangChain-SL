import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# ------------------------------ Image Input ---------------------------------

image_file_path1 = "1.jpg"
with open(image_file_path1, "rb") as image_file:
    encoded_image1 = base64.b64encode(image_file.read()).decode("utf-8")

image_file_path2 = "2.jpg"
with open(image_file_path2, "rb") as image_file:
    encoded_image2 = base64.b64encode(image_file.read()).decode("utf-8")

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful AI Assistant"),
#     ("human", [
#         {"type": "text", "text": "{question}"},
#         {"type": "image_url", "image_url": f"data:image/png;base64,{encoded_image1}"},
#     ])
# ])

### Simplified version
### Passing single image ####
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI Assistant"),
    ("human", [
        {"text": "{question}"},
        {"image_url": "data:image/png;base64,{encoded_image1}"},
    ])
])

chain = prompt | llm
response = chain.invoke({"encoded_image1": encoded_image1, "question": "Descibe the image"})

### Passing multiple image ####
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI Assistant"),
    ("human", [
        {"text": "{question}"},
        {"image_url": "data:image/png;base64,{encoded_image1}"},
        {"image_url": "data:image/png;base64,{encoded_image2}"},
    ])
])

chain = prompt | llm
# response = chain.invoke({"encoded_image1": encoded_image1, "encoded_image2": encoded_image2, "question": "Are two images are identical?"})

# print(response.content)

# https://python.langchain.com/docs/integrations/chat/google_generative_ai/
# https://python.langchain.com/docs/how_to/multimodal_inputs/
