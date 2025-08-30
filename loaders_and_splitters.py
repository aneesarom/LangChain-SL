from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, TextSplitter, RecursiveJsonSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader
from langchain_core.documents.base import Document

#### .page_content gives the content of the document & .metadata gives the metadata of document

# ---------------------------TextLoader----------------------------------

loader = TextLoader("sample.txt")
text_documents = loader.load()
# print(text_documents)

# ---------------------------PyPDFLoader----------------------------------

### Load a pdf and returns a list of langchain Document
loader = PyPDFLoader("Use of Optimizers and its types.pdf")
documents = loader.load()
# print(documents[0].page_content)

# ---------------------------split_documents----------------------------------

#### If input documents is list of documents use "split_documents"

### Allows single separator
splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000, chunk_overlap=20)
chunks2 = splitter.split_documents(documents)
# print(chunks2)

### Allows list of separators
### If a chunk is too big, it recursively tries the next separator to break it down further.
splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=1000, chunk_overlap=20)
chunks1 = splitter.split_documents(text_documents)
# print(chunks1)

# ---------------------------split_text----------------------------------

#### If input text is string use "split_text"
chunks3 = splitter.split_text(text_documents[0].page_content)
# print(chunks3)

# ---------------------------create langchain documents----------------------------------

documents_list = list()
for idx, chunk in enumerate(chunks3):
    document = Document(page_content=chunk, metadata={"chunk": idx, "source": "sample.txt"})
    documents_list.append(document)

print(documents_list)


