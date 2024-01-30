import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor
import cassio
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
chat_llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

cassio.init(token=os.getenv('ASTRA_DB_APPLICATION_TOKEN'), database_id=os.getenv('ASTRA_DB_ID'))
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="budget_speech",
    session=None,
    keyspace=None,
)
retriever = astra_vector_store.as_retriever(search_kwargs={"k": 3})

retriever_tool = create_retriever_tool(
    retriever,
    "budget_speech_search",
    "Search for information about the Indian budget 2023-2024. For any questions about the Indian budget, \
    you must use this tool!",
)

prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [retriever_tool]

agent = create_openai_functions_agent(chat_llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "Who gave the budget speech?"})
