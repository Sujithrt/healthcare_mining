from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

MEMORY_KEY = "chat_history"


def create_agent_executor(astra_vector_store, openai_api_key, stream_handler):
    retriever = astra_vector_store.as_retriever(search_kwargs={"k": 3})

    retriever_tool = create_retriever_tool(
        retriever,
        "data_search",
        # "Search for information about the Indian budget 2023-2024. For any questions about the Indian budget, \
        # you must use this tool!",
        "Search for information about diseases, Prescription and allergies, SSD (Single shot detectors) \
         management system, Peter Rabbit, information about customers, and answers to quiz questions \
         in the vector database and return the most relevant information."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are very powerful assistant with access to tools that can help you retrieve data \
            from the vector database, answer questions, and more. You can also upload files and submit links."),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    chat_llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0.5,
        streaming=True,
        callbacks=[stream_handler],
    )

    tools = [retriever_tool]
    llm_with_tools = chat_llm.bind_tools(tools)

    agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor
