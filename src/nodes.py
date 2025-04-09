from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain import hub
from pydantic import BaseModel, Field


def agent(llm, tools):
    def _agent_node(state):
        print("--- AGENT NODE ---")
        model = llm.bind_tools(tools)
        return {"messages": [model.invoke(state["messages"])]}
    return _agent_node


def rewrite(llm):
    def _rewrite_node(state):
        print("--- REWRITE NODE ---")
        question = state["messages"][0].content
        msg = [HumanMessage(content=f"Rewrite this for clarity:\n\n{question}")]
        return {"messages": [llm.invoke(msg)]}
    return _rewrite_node


def generate(llm):
    def _generate_node(state):
        print("--- GENERATE NODE ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content
        rag_chain = hub.pull("rlm/rag-prompt") | llm | StrOutputParser()
        return {"messages": [rag_chain.invoke({"question": question, "context": docs})]}
    return _generate_node


def web_search(search_tool):
    def _search_node(state):
        print("--- WEB SEARCH NODE ---")
        question = state["messages"][0].content
        results = search_tool.invoke(question)

        # Convert the result to string (if dict or list) and wrap in a message
        return {"messages": [HumanMessage(content=str(results))]}
    return _search_node


def grade_documents(llm):
    def _grade_node(state):
        class Grade(BaseModel):
            binary_score: str = Field(description="'yes' or 'no' indicating document relevance")

        grade_chain = (
            PromptTemplate(
                template="""You are a grader. Here is the document:\n\n{context}\n\n
                And the user question: {question}\n
                If the content is related, return 'yes'. Otherwise, return 'no'.""",
                input_variables=["context", "question"]
            ) | llm.with_structured_output(Grade)
        )

        question = state["messages"][0].content
        docs = state["messages"][-1].content
        result = grade_chain.invoke({"question": question, "context": docs})
        return "generate" if result.binary_score == "yes" else "web_search"

    return _grade_node

