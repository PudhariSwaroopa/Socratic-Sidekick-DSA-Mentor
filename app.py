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

st.title("🧠 Socratic Sidekick DSA Mentor")
st.write("Paste your code or ask about a DSA problem.")

user_input = st.text_area("Your Question / Code", height=250)


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
    model_name="sentence-transformers/all-MiniLM-L6-v2"
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

    vectorstore = PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embedding_model,
        index_name=index_name
    )

    return vectorstore


vectorstore = create_vectorstore()

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8}
)


# -------------------------
# LLM
# -------------------------

llm = Ollama(model="llama3")


# -------------------------
# Prompt
# -------------------------

system_prompt = (
    "You are a Socratic mentor that teaches Data Structures and Algorithms (DSA). "
    "Your goal is to help students understand problems, analyze their code, and improve their reasoning skills. "
    "You must guide students to discover mistakes and solutions themselves instead of giving direct answers."

    "\n\nThe student may provide:"
    "\n• a DSA question"
    "\n• a problem description"
    "\n• their own code (which may contain logical or runtime errors)"

    "\n\nFollow these teaching steps carefully:\n"

    "1. If the student provides code, carefully analyze the code step-by-step.\n"
    "2. Identify logical mistakes or inefficiencies without fixing the code.\n"
    "3. Explain the problem clearly.\n"
    "4. Identify the algorithm pattern involved.\n"
    "5. Explain the reasoning behind a correct approach.\n"
    "6. Provide 2–3 hints that guide the student toward the solution.\n"
    "7. NEVER provide code."

    "\n\nStructure your response as:\n"
    "Code Understanding\n"
    "Problem Explanation\n"
    "Algorithm Pattern\n"
    "Solution Logic\n"
    "Hints\n"

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
# Run Mentor
# -------------------------

if st.button("Analyze Code"):

    if user_input.strip() == "":
        st.warning("Please enter a question or code")

    else:

        with st.spinner("Mentor is thinking..."):

            response = rag_chain.invoke({"input": user_input})

        st.subheader("Guidance")

        st.write(response["answer"])