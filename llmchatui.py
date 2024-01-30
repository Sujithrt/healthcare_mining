import streamlit as st
import openai
from langchain.callbacks import OpenAICallbackHandler, get_openai_callback
import pandas as pd 
import json
import logging
from langchain.document_loaders import CSVLoader, UnstructuredPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from tempfile import NamedTemporaryFile
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain,RetrievalQA
from streamlit_chat import message
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os
from dotenv import load_dotenv
import requests
from newspaper import Article
import time

load_dotenv()
# Statically initialize your API key
os.environ["OPENAI_API_KEY"] = "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"
os.environ["ACTIVELOOP_TOKEN"] = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwNTYxNjI1NSwiZXhwIjoxNzY4Nzc0NjQyfQ.eyJpZCI6InJvbml0cGF0aWwifQ.au8M98Fe76AWb_Lne5pVGVOf4azrdpbQuzyO_3BJtkl--PJYxrlgQVEFfQSIc7pdFX5rSyHaILojzhDXT6iSGA"
# Dummy usage statistics - replace these with your actual logic for calculating usage
# TOTAL_TOKENS_USED = 1782
# TOTAL_COST = 0.003564
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "ronitpatil"
my_activeloop_dataset_name = "langchain_course_qabot_with_source"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

try:
    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
except Exception as e:
    st.error(f"Failed to initialize DeepLake: {e}")
    raise

if "usage" not in st.session_state:
    st.session_state["usage"] = {}
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [
        SystemMessage(content="You are a an AI Assistant with the knowledge to help the user with any questions asked")
    ]
if "generated" not in st.session_state:
	st.session_state["generated"] = []
if "past" not in st.session_state:
	st.session_state["past"] = []

def handle_load_error(loader):
    if loader:
        st.error("Error occurred in loader")
    else:
        st.error("Loader is not initialized")

def get_loader_for_file(file_path):
    if file_path.endswith('.csv'):
        return CSVLoader(file_path, encoding="utf-8")
    elif file_path.endswith('.pdf'):
        return UnstructuredPDFLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

def load_documents(data_source, chunk_size):
    try:
        loader = get_loader_for_file(data_source)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        docs = loader.load_and_split(text_splitter)
        st.info(f"Loaded: {len(docs)} document chunks")
        return docs
    except Exception as e:
        handle_load_error(loader if 'loader' in locals() else None)
        return []  # Return an empty list if there's an error

def process_uploaded_file(uploaded_file, db):
    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    # Load and chunk the documents
    docs = load_documents(tmp_file_path, chunk_size=1000)
    
    # Check if docs is not None and is iterable
    if docs is not None:
        # Embed the chunks using OpenAIEmbeddings
        embedded_docs = [doc.page_content for doc in docs]
        db.add_texts(embedded_docs)

        # Store the embeddings in the vector database
        # for doc, embedding in zip(docs, embedded_docs):
        #     db.add_texts(doc, embedding)
    else:
        # If docs is None, log an error or handle it as per your logic
        st.error("Failed to load documents from the uploaded file.")

    os.unlink(tmp_file_path)  # Clean up the temporary file
    return docs if docs is not None else []

def scrape_link(url,db):
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }
    session = requests.Session()
    pages_content = [] # where we save the scraped article

    try:
        time.sleep(2) # sleep two seconds for gentle scraping
        response = session.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            article = Article(url)
            article.download() # download HTML of webpage
            article.parse() # parse HTML to extract the article text
            pages_content.append({ "url": url, "text": article.text })
        else:
            print(f"Failed to fetch article at {url}")
    except Exception as e:
        print(f"Error occurred while fetching article at {url}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    all_texts, all_metadatas = [], []
    for d in pages_content:
        chunks = text_splitter.split_text(d["text"])
        for chunk in chunks:
            all_texts.append(chunk)
            # all_metadatas.append({ "source": d["url"] })
    # db.add_texts(all_texts, all_metadatas)
    db.add_texts(all_texts)

def update_usage(cb: OpenAICallbackHandler) -> None:
    callback_properties = [
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_cost",
    ]
    for prop in callback_properties:
        value = getattr(cb, prop, 0)
        st.session_state["usage"].setdefault(prop, 0)
        st.session_state["usage"][prop] += value

def generate_response(prompt: str) -> str:
    """
    Call the OpenAI GPT-3 API to generate a response based on the given prompt.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")
        
    openai.api_key = api_key

    model = ChatOpenAI(
    model_name="gpt-4-turbo-preview",
    temperature=0.7,
    openai_api_key=os.environ["OPENAI_API_KEY"],
)
    chain = ConversationalRetrievalChain.from_llm(
    model,
    retriever=db.as_retriever(),
    chain_type="stuff",
    verbose=True,
    # we limit the maximum number of used tokens
    # to prevent running into the models token limit of 4096
    max_tokens_limit=150,
)
    response_data = chain({
        "question": prompt,
        "chat_history": st.session_state["chat_history"]
    })
    # Append the new interaction to the chat history
    st.session_state["chat_history"].append((prompt, response_data["answer"]))
    return response_data["answer"]

def main():
    # Sidebar for API usage
    with st.sidebar:
        st.header("API Usage")
        if st.session_state["usage"]:
            st.metric("Total Tokens", st.session_state["usage"]["total_tokens"])
            st.metric("Total Costs in $", st.session_state["usage"]["total_cost"])

    st.title("Chatbot Interface", anchor=None)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Upload Files")
        uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True, type=['csv', 'pdf'])
        if uploaded_files:
            for uploaded_file in uploaded_files:
                docs = process_uploaded_file(uploaded_file,db)
                st.write(f"Processed {len(docs)} records from the file {uploaded_file.name}")

    with col2:
        st.markdown("### Submit Links")
        new_link = st.text_input("", key="new_link", placeholder="Paste your link here...")
        if st.button("Submit Link", key="submit_link"):
            if "submitted_links" not in st.session_state:
                st.session_state.submitted_links = []
            st.session_state.submitted_links.append(new_link)
            st.success("Link submitted!")
            
            # Call the scrape_link function here
            try:
                scrape_link(new_link, db)
                st.success("Link successfully scraped and processed!")
            except Exception as e:
                st.error(f"Failed to scrape link: {e}")

    st.title("Ask a Question!")

    # User input for the question
    user_question = st.text_input("What would you like to ask?", key="user_question")

    if st.button('Submit'):
        if user_question:
            with st.spinner("Generating response"), get_openai_callback() as cb:
                # Generate the response
                answer = generate_response(user_question)
                st.session_state["chat_history"].append((user_question, answer))
                # st.session_state.past.append(user_question)
                # st.session_state.generated.append(answer) 
                st.session_state.past.insert(0, user_question)
                st.session_state.generated.insert(0, answer)
                update_usage(cb)
            st.success('Response generated!')
            

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))


if __name__ == "__main__":
    main()
