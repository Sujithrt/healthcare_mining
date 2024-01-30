import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain import hub
import cassio
import time

st.set_page_config(page_title="Healthcare Chatbot", page_icon=":robot_face:")
st.header('Healthcare Chatbot')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue
    role = None
    content = None
    if isinstance(message, HumanMessage):
        role = "user"
        content = message.content
    elif isinstance(message, AIMessage):
        role = "assistant"
        content = message.content

    with st.chat_message(role):
        st.markdown(content)

embedding = OpenAIEmbeddings()

cassio.init(token=st.secrets['ASTRA_DB_APPLICATION_TOKEN'], database_id=st.secrets['ASTRA_DB_ID'])
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

agent_prompt = hub.pull("hwchase17/openai-functions-agent")
chat_llm = ChatOpenAI(
    openai_api_key=st.secrets['OPENAI_API_KEY'],
    temperature=0.5,
)

tools = [retriever_tool]

agent = create_openai_functions_agent(chat_llm, tools, agent_prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# chat_history = []
# contextualize_q_system_prompt = """Given a chat history and the latest user question \
# which might reference context in the chat history, formulate a standalone question \
# which can be understood without the chat history. Do NOT answer the question, \
# just reformulate it if needed and otherwise return it as is."""
# contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         SystemMessagePromptTemplate.from_template(contextualize_q_system_prompt),
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessage(content="{question}"),
#     ]
# )
# contextualize_q_chain = contextualize_q_prompt | chat_llm | StrOutputParser()
# qa_system_prompt = """You are an assistant for question-answering tasks. \
# Use the following pieces of retrieved context to answer the question. \
# If you don't know the answer, just say that you don't know. \
# Use three sentences maximum and keep the answer concise.\
#
# {context}"""
# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         SystemMessagePromptTemplate.from_template(qa_system_prompt),
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessage(content="{question}"),
#     ]
# )
#
#
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)
#
#
# def contextualized_question(input: dict):
#     if input.get("chat_history"):
#         return contextualize_q_chain
#     else:
#         return input["question"]
#
#
# rag_chain = (
#     RunnablePassthrough.assign(
#         context=contextualized_question | retriever | format_docs
#     )
#     | qa_prompt
#     | chat_llm
# )

if user_input := st.chat_input("Ask me anything"):
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").markdown(user_input)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # response = rag_chain.invoke({"question": user_input, "chat_history": chat_history})
        # chat_history.extend([HumanMessage(content=user_input), response])
        response = agent_executor.invoke({"input": user_input})
        for chunk in response['output'].split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response)
    st.session_state.messages.append(AIMessage(content=response['output']))
