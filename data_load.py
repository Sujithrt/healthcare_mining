from langchain_community.vectorstores import Cassandra
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cassio
from PyPDF2 import PdfReader
import streamlit as st


def populate_vector_store():
    pdfreader = PdfReader('Documents/budget_speech.pdf')

    # read text from pdf
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    cassio.init(token=st.secrets['ASTRA_DB_APPLICATION_TOKEN'], database_id=st.secrets['ASTRA_DB_ID'])
    embedding = OpenAIEmbeddings()

    astra_vector_store = Cassandra(
        embedding=embedding,
        table_name="budget_speech",
        session=None,
        keyspace=None,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )

    texts = text_splitter.split_text(raw_text)

    astra_vector_store.add_texts(texts)


if __name__ == '__main__':
    populate_vector_store()