# documents.py
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(urls):
    docs = [doc for url in urls for doc in WebBaseLoader(url).load()]
    return docs

def split_documents(docs, chunk_size=100, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)