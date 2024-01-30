import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import cassio
import time


st.set_page_config(page_title="Healthcare Chatbot", page_icon=":robot_face:")
st.header('Healthcare Chatbot')

embedding = OpenAIEmbeddings()

cassio.init(token=st.secrets['ASTRA_DB_APPLICATION_TOKEN'], database_id=st.secrets['ASTRA_DB_ID'])
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="budget_speech",
    session=None,
    keyspace=None,
)
retriever = astra_vector_store.as_retriever(search_kwargs={"k": 3})

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are an AI Assistant with knowledge about healthcare")
    ]

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

if user_input := st.chat_input("Ask me anything"):
    st.chat_message("user").markdown(user_input)
    chat_llm = ChatOpenAI(
        openai_api_key=st.secrets['OPENAI_API_KEY'],
        temperature=0.5,
    )
    st.session_state.messages.append(HumanMessage(content=user_input))
    # chat_template = ChatPromptTemplate.from_messages(st.session_state.messages)
    template = """
    You are an AI Assistant with knowledge about healthcare
    
    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:"""
    chat_template = ChatPromptTemplate.from_template(template)
    # chain = LLMChain(llm=chat_llm, prompt=chat_template)
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | chat_template
            | chat_llm
            | StrOutputParser()
    )
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response = chain.invoke(user_input)
        print(response)
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response)
    st.session_state.messages.append(AIMessage(content=response))
