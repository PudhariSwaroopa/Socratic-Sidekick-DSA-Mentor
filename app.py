import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Socratic Sidekick", page_icon="🧠")

st.title("🧠 Socratic Sidekick DSA Mentor")
st.write("Ask a DSA question or paste your code.")


# -------------------------
# Load Environment Variables
# -------------------------

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# -------------------------
# Load Dataset (Cached)
# -------------------------

@st.cache_data
def load_documents():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "data", "leetcode_dataset.csv")

    df = pd.read_csv(csv_path)

    docs = []

    for _, row in df.iterrows():
        content = " ".join([str(v) for v in row.values])
        docs.append(Document(page_content=content))

    return docs


documents = load_documents()


# -------------------------
# Text Splitting
# -------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

text_chunks = splitter.split_documents(documents)


# -------------------------
# Embeddings
# -------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


# -------------------------
# Vectorstore (Cached)
# -------------------------

@st.cache_resource
def create_vectorstore():

    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_name = "dsa-mentor"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding_model
    )

    return vectorstore


vectorstore = create_vectorstore()

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2}
)

# -------------------------
# LLM
# -------------------------

llm = Ollama(
    model="llama3",
    temperature=0.2,
    num_predict=200
)


# -------------------------
# Prompt
# -------------------------

system_prompt = (
"You are Socratic Sidekick, a friendly AI mentor that teaches Data Structures and Algorithms (DSA). "
"Your goal is to help students understand problems and develop problem-solving skills."

"\n\nImportant rules:"
"\n• Never invent a problem if the user did not provide one."
"\n• If the user gives no code or problem description, politely ask them to provide one."
"\n• If the user asks for the solution code, refuse politely and guide them with hints."
"\n• If the user replies with short responses like 'yes', 'ok', or 'sure', continue guiding them."

"\n\nIf the student provides code:"
"\n1. Explain what the code is trying to do."
"\n2. Identify logical mistakes or inefficiencies."
"\n3. Explain the algorithm concept involved."
"\n4. Provide hints so the student can fix the code."

"\n\nAlways structure responses as:"
"\nCode Understanding"
"\nProblem Explanation"
"\nAlgorithm Pattern"
"\nSolution Logic"
"\nHints"

"\n\nContext:\n{context}"
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)


# -------------------------
# RAG Chain
# -------------------------

qa_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, qa_chain)


# -------------------------
# Chatbot Interface
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# user input
user_prompt = st.chat_input("Ask a DSA question or paste your code...")


if user_prompt:

    # store user message
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Mentor is thinking..."):

            response = rag_chain.invoke({"input": user_prompt})

            answer = response["answer"]

            st.markdown(answer)

    # store assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )