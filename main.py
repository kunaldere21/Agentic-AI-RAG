# main.py
from src.embeddings import load_embeddings
from src.documents import load_documents, split_documents
from src.tools import build_vectorstore, get_tools, load_vector_db
from src.nodes import agent, rewrite, generate, web_search, grade_documents
from src.graph import build_graph
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI()
# from langchain_groq import ChatGroq
# llm=ChatGroq(model_name="Gemma2-9b-It")
embeddings = load_embeddings()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = load_documents(urls)
doc_chunks = split_documents(docs)
# vectorstore = build_vectorstore(doc_chunks, embeddings)
vectorstore = load_vector_db(embeddings=embeddings, persist_dir="Agentic_RAG/chromadb")
retriever_tool, search_tool = get_tools(vectorstore)

workflow = build_graph(
    agent(llm, [retriever_tool]),
    retriever_tool,
    generate(llm),
    rewrite(llm),
    web_search(search_tool),
    grade_documents(llm)
)

if __name__=="__main__":
    from IPython.display import Image, display

    graph_png = workflow.get_graph(xray=True).draw_mermaid_png()

    # Save it to a file
    with open("workflow_graph.png", "wb") as f:
        f.write(graph_png)



    inputs = "Todays, Weather of Hyderabad"
 
    print((20* '*') ,"Agent response", (20* '*'))
    result = workflow.invoke({"messages": [inputs]})
    print(result["messages"][-1].content)
