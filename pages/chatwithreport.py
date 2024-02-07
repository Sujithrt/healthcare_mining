import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from langchain_community.callbacks import get_openai_callback
from utils.initialize_vector_store import initialize_vector_store
from utils.create_agent_executor import agent_executor_1
from utils.data_loader import populate_vector_store

# Initialize session state for chat history and messages if not already present
if 'chat_history_report' not in st.session_state:
    st.session_state['chat_history_report'] = []
if "messages_report" not in st.session_state:
    st.session_state.messages_report = []

# Function to update chat history
def update_chat_history(user_input, agent_response=None):
    if user_input:
        st.session_state.chat_history_report.append(f"User: {user_input}")
    if agent_response:
        st.session_state.chat_history_report.append(f"Agent: {agent_response}")

# Function to get the current chat context
def get_current_context():
    return "\n".join(st.session_state.chat_history_report)

# Main function
def main():
    st.set_page_config(page_title="Healthcare Chatbot", page_icon=":robot_face:")
    st.header('Healthcare Chatbot')

    # Initialize vector store and agent executor
    astra_vector_store = initialize_vector_store(st.secrets['ASTRA_DB_APPLICATION_TOKEN'], st.secrets['ASTRA_DB_ID'])
    agent_executor = agent_executor_1(astra_vector_store, st.secrets['OPENAI_API_KEY'])

    # Chatbox
    user_input = st.chat_input("Ask me anything")
    if user_input:
        st.session_state.messages_report.append(HumanMessage(content=user_input))
        update_chat_history(user_input)
        context_with_input = get_current_context()
        with get_openai_callback() as cb:
            response = agent_executor.invoke({"input": context_with_input})
            update_chat_history(None, response['output'])
            st.session_state.messages_report.append(AIMessage(content=response['output']))

    # Display chat messages
    for message in st.session_state.messages_report:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    # File upload for vector store population
    st.markdown("### Upload Files")
    uploaded_files = st.file_uploader("Drag and drop files here", accept_multiple_files=True,
                                    type=['pdf', 'png', 'jpeg', 'jpg'])  # Allowed file types
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Check file type and process accordingly
            if uploaded_file.type in ["application/pdf", "image/png", "image/jpeg"]:
                populate_vector_store(uploaded_file, astra_vector_store)
                st.success(f"Processed file {uploaded_file.name}. You may ask questions about the file now.")
            else:
                st.error(f"Unsupported file type: {uploaded_file.type}")

# Run the main function
if __name__ == "__main__":
    main()
