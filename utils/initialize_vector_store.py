from langchain_community.vectorstores import Cassandra
from langchain_openai import OpenAIEmbeddings
import cassio

from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
import faiss

def initialize_vector_store(astra_db_application_token, astra_db_id):
    embedding = OpenAIEmbeddings()
    cassio.init(token=astra_db_application_token, database_id=astra_db_id)
    astra_vector_store = Cassandra(
        embedding=embedding,
        table_name="general",
        session=None,
        keyspace=None,
    )
    return astra_vector_store

def vectorstore_AutoGPT(astra_db_application_token, astra_db_id):
    

    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    embedding_size = 1536

    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    return vectorstore