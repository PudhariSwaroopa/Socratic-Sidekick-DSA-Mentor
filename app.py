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
# Reasoning Tracking
# -------------------------

if "reasoning_path" not in st.session_state:
    st.session_state.reasoning_path = []

# turn counter to avoid endless loops
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0


# -------------------------
# Sidebar Reasoning View
# -------------------------

st.sidebar.title("🧠 Student Reasoning Path")

for step in st.session_state.reasoning_path:
    st.sidebar.write("•", step)


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
"You are Socratic Sidekick, an AI mentor that teaches Data Structures and Algorithms (DSA) using the Socratic teaching method. "
"Your purpose is to guide students to discover solutions themselves through reasoning, questions, and hints."
"If the conversation continues for multiple turns, gradually provide stronger hints so the student can reach the correct reasoning within a few steps."

"\n\nStrict Rules:"
"\n• NEVER reveal or generate the final solution code."
"\n• NEVER directly fix the student's code."
"\n• Do NOT provide pseudocode or implementation."
"\n• Always guide the student through logical questions."
"\n• Help the student reason step-by-step."

"\n\nConversation Behavior:"
"\n• Track the student's reasoning across messages."
"\n• Refer to the student's previous attempts when guiding them."
"\n• If the student replies briefly (yes, ok, sure), continue guiding them."
"\n• If no problem or code is provided, politely ask for it."

"\n\nIf the student provides code:"
"\n1. Explain what the code is trying to accomplish."
"\n2. Identify possible logical mistakes or inefficiencies."
"\n3. Identify the algorithm pattern involved."
"\n4. Ask reasoning questions that help the student detect the mistake."
"\n5. Provide small hints but NEVER reveal the fix."

"\n\nIf the student code is inefficient:"
"\n• Help them analyze time complexity."
"\n• Ask questions that guide them toward a better approach."

"\n\nAlways structure responses exactly like this:"
"\n\nCode Understanding:"
"\nExplain what the student's code is attempting to do."

"\n\nProblem Explanation:"
"\nExplain the core idea of the problem."

"\n\nReasoning Question:"
"\nAsk a logical question that helps the student think."

"\n\nHint:"
"\nProvide a small conceptual hint but NOT the solution."

"\n\nNext Step for the Student:"
"\nTell the student what they should think about or try next."

"\n\nContext:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Conversation so far:\n{chat_history}\n\nNew student message:\n{input}")
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
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm Socratic Sidekick 🧠. Paste your DSA problem or code and I'll guide you through the reasoning."
        }
    ]


# display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# user input
user_prompt = st.chat_input("Ask a DSA question or paste your code...")


if user_prompt:

    st.session_state.turn_count += 1

    # store reasoning
    st.session_state.reasoning_path.append(user_prompt)

    # store user message
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Mentor is thinking..."):

            chat_history = ""

            for msg in st.session_state.messages:
                role = msg["role"]
                content = msg["content"]
                chat_history += f"{role}: {content}\n"

            response = rag_chain.invoke({
                "input": user_prompt,
                "chat_history": chat_history
            })

            answer = response["answer"]

            # stronger hint escalation after several turns
            if st.session_state.turn_count >= 4:
                answer += "\n\n💡 **Stronger Hint:** You're very close. Focus carefully on the boundary conditions or data structure being updated."

            # Safety filter to prevent code output
            if "def " in answer or "class " in answer:
                answer = (
                    "Let's focus on reasoning rather than jumping to the solution. "
                    "What do you think the issue might be in your approach?"
                )

            st.markdown(answer)

    # store assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )