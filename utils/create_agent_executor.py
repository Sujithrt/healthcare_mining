from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain import hub
from langchain_openai import ChatOpenAI
import streamlit as st

from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.chat_models import ChatOpenAI
# from langchain_experimental import AutoGPT
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner


def agent_executor_1(astra_vector_store, openai_api_key):
    retriever = astra_vector_store.as_retriever(search_kwargs={"k": 3})

    retriever_tool = create_retriever_tool(
        retriever,
        "data_search",
        # "Search for information about the Indian budget 2023-2024. For any questions about the Indian budget, \
        # you must use this tool!",
        "Search for information about diseases, Prescription and allergies \
         management system, Peter Rabbit, information about customers, and answers to quiz questions \
         in the vector database and return the most relevant information."
    )

    agent_prompt = hub.pull("hwchase17/openai-functions-agent")
    chat_llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0.5,
    )

    tools = [retriever_tool]

    agent = create_openai_functions_agent(chat_llm, tools, agent_prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

def agent_executor_AutoGPT(astra_vector_store, openai_api_key):

    google_api_key=st.secrets['GOOGLE_API_KEY']
    google_cse_id=st.secrets['GOOGLE_API_KEY']
    search = GoogleSearchAPIWrapper(google_api_key=google_api_key,google_cse_id=google_cse_id)

    tools = [
    Tool(
        name = "search",
        func=search.run,
        description="Useful for when you need to answer questions about current events. You should ask targeted questions",
        return_direct=True
        ),
    WriteFileTool(),
    ReadFileTool(),

    ]

    chat_llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0.5,
    )

    agent = AutoGPT.from_llm_and_tools(
    ai_name="Jim",
    ai_role="Assistant",
    tools=tools,
    llm=chat_llm,
    memory=astra_vector_store.as_retriever()
    )

    agent.chain.verbose = True

    return agent

def agent_executor_PlanExecute(astra_vector_store, openai_api_key):
    retriever = astra_vector_store.as_retriever()
    retriever.search_kwargs['k'] = 3

    CUSTOM_TOOL_DOCS_SEPARATOR ="\n---------------\n" # how to join together the retrieved docs to form a single string

    def retrieve_n_docs_tool(query: str) -> str:
        """Searches for relevant documents that may contain the answer to the query."""
        docs = retriever.get_relevant_documents(query)
        texts = [doc.page_content for doc in docs]
        texts_merged = "---------------\n" + CUSTOM_TOOL_DOCS_SEPARATOR.join(texts) + "\n---------------"
        return texts_merged
    
    tools = [
    Tool(
        name="Search Database",
        func=retrieve_n_docs_tool,
        description="useful for when you need to answer questions from database"
            )
    ]

    chat_llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0.5,
    )

    planner = load_chat_planner(chat_llm)
    executor = load_agent_executor(chat_llm, tools, verbose=True)
    agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

    return agent



    




