import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import ConversationalRetrievalChain
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
        SystemMessage(content="You are an AI Assistant with information about the Indian budget 2023-2024")
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
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").markdown(user_input)
    chat_llm = ChatOpenAI(
        openai_api_key=st.secrets['OPENAI_API_KEY'],
        temperature=0.5,
    )
    system_template = """
    You are an AI Assistant with information about the Indian budget 2023-2024
    CONTEXT:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="{question}"),
        ]
    )
    chain = ConversationalRetrievalChain.from_llm(llm=chat_llm, retriever=retriever,
                                                  combine_docs_chain_kwargs={"prompt": prompt})

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: RedisChatMessageHistory(session_id, url=st.secrets['REDIS_URL']),
        input_messages_key="question",
        history_messages_key="history",
    )
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response = chain_with_history.invoke(
            {"question": user_input, "chat_history": st.session_state.messages},
            config={"configurable": {"session_id": "1"}},
        )
        print(response)
        for chunk in response['answer'].split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response)
    st.session_state.messages.append(AIMessage(content=response["answer"]))
