from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import TavilySearchResults
# from langchain.vectorstores import Chroma

def build_vectorstore(doc_chunks, embeddings, persist_dir="chromadb"):
    return Chroma.from_documents(
        documents=doc_chunks,
        embedding=embeddings,
        collection_name="rag-chroma",
        persist_directory=persist_dir
    )

# Load the persisted Chroma DB
def load_vector_db(embeddings, persist_dir="chromadb"):
    vectorstore = Chroma(
        persist_directory="./chroma_db",  # Path to your local directory
        embedding_function=embeddings
    )
    return vectorstore

def get_tools(vectorstore):
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        name="retrieve_blog_posts",
        description="Search Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks."
    )
    web_search_tool = TavilySearchResults(k=3)
    return retriever_tool, web_search_tool