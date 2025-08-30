from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory, ConversationKGMemory
from langchain.chains.conversation.base import ConversationChain
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import PromptTemplate

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

#------------------------- ConversationBufferMemory -------------------------

memory = ConversationBufferMemory()
chain = ConversationChain(memory=memory, verbose=False, llm=llm)

# while True:
#     question = input("Ask your question: ")
#     if question == "exit":
#         break
#     reply = chain.predict(input=question)
#     print(reply)

#------------------------- ConversationSummaryMemory -------------------------

memory = ConversationSummaryMemory(llm=llm)
chain = ConversationChain(memory=memory, verbose=True, llm=llm)

# while True:
#     question = input("Ask your question: ")
#     if question == "exit":
#         break
#     reply = chain.predict(input=question)
#     print(reply)

#------------------------- ConversationBufferWindowMemory -------------------------

memory = ConversationBufferWindowMemory(k=2)
chain = ConversationChain(memory=memory, verbose=True, llm=llm)

# while True:
#     question = input("Ask your question: ")
#     if question == "exit":
#         break
#     reply = chain.predict(input=question)
#     print(reply)

# ------------------------- ConversationSummaryBufferMemory -------------------------

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=200)
chain = ConversationChain(memory=memory, verbose=True, llm=llm)

# while True:
#     question = input("Ask your question: ")
#     if question == "exit":
#         break
#     reply = chain.predict(input=question)
#     print(reply)

# ------------------------------------------------------------------------------

# https://www.youtube.com/watch?v=sYlMD2OFEgc
