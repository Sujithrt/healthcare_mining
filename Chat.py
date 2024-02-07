import os
import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.callbacks import OpenAICallbackHandler, get_openai_callback

from utils.data_loader import populate_vector_store, scrape_link
from utils.initialize_vector_store import initialize_vector_store
from utils.create_agent_executor import agent_executor_AutoGPT, agent_executor_1,agent_executor_PlanExecute

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if "messages" not in st.session_state:
        st.session_state.messages = []
        
def update_usage(cb: OpenAICallbackHandler) -> None:
    callback_properties = [
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_cost",
    ]
    print('Callback', cb)
    for prop in callback_properties:
        value = getattr(cb, prop, 0)
        st.session_state.usage[prop] += value


def main():
    if "usage" not in st.session_state:
        st.session_state.usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
        }
    

    
    astra_vector_store = initialize_vector_store(st.secrets['ASTRA_DB_APPLICATION_TOKEN'], st.secrets['ASTRA_DB_ID'])

    st.set_page_config(page_title="Healthcare Chatbot", page_icon=":robot_face:")
    st.header('Healthcare Chatbot')
    with st.sidebar:
        st.header("API Usage")
        if st.session_state["usage"]:
            st.metric("Total Tokens", st.session_state["usage"]["total_tokens"])
            st.metric("Total Costs in $", st.session_state["usage"]["total_cost"])
        with st.container(border=True):
            st.markdown("### Upload Files")
            uploaded_files = st.file_uploader("Upload a file",
                                              accept_multiple_files=True,
                                              type=['csv', 'pdf', 'json', 'html', 'md'],
                                              label_visibility='hidden')
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    populate_vector_store(uploaded_file, astra_vector_store)
                    st.success(f"Processed file {uploaded_file.name}. You may ask me questions about the file now.")

        with st.container(border=True):
            st.markdown("### Submit Links")
            new_link = st.text_input("Enter a link",
                                     key="new_link",
                                     placeholder="Paste your link here...",
                                     label_visibility='hidden')
            if st.button("Submit Link", key="submit_link"):
                try:
                    scrape_link(new_link, astra_vector_store)
                    st.success("Link successfully scraped and processed!")
                except Exception as e:
                    st.error(f"Failed to scrape link: {e}")

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

    agent_executor = agent_executor_1(astra_vector_store, st.secrets['OPENAI_API_KEY'])
    # agentAutoGPT=agent_executor_AutoGPT(astra_vector_store, st.secrets['OPENAI_API_KEY'])
    # agentPlanExecute=agent_executor_PlanExecute(astra_vector_store, st.secrets['OPENAI_API_KEY'])


    def update_chat_history(user_input, agent_response=None):
        if user_input:
            st.session_state.chat_history.append(f"User: {user_input}")
        if agent_response:
            st.session_state.chat_history.append(f"Agent: {agent_response}")

    def get_current_context():
        return "\n".join(st.session_state.chat_history)

    if user_input := st.chat_input("Ask me anything"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        st.chat_message("user").markdown(user_input)
        update_chat_history(user_input)
        context_with_input = get_current_context()
        with get_openai_callback() as cb:
            # response = agent_executor.invoke({"input": user_input})
            response = agent_executor.invoke({"input": context_with_input})
            # response=agentAutoGPT.run([user_input])
            # response=agentPlanExecute.run(user_input)
            update_usage(cb)
            # st.chat_message("assistant").markdown(response['output'])
            st.chat_message("assistant").markdown(response['output'])
            st.session_state.messages.append(AIMessage(content=response['output']))
            update_chat_history(None, response['output'])


if __name__ == "__main__":
    main()
