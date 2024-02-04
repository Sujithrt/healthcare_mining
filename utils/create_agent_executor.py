from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain import hub
from langchain_openai import ChatOpenAI


def create_agent_executor(astra_vector_store, openai_api_key):
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
