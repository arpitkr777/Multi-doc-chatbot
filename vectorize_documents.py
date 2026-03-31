from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader

from langchain_text_splitters import CharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

# loading the embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

loader = DirectoryLoader(path="data", glob="*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()


text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)

text_chunks = text_splitter.split_documents(documents)

#creating the vector store
vectordb = Chroma.from_documents(documents=text_chunks, embedding=embeddings,persist_directory="vector_db_dir")

print("Documents vectorized")
