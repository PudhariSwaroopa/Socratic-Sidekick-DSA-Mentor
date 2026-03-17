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
# UI
# -------------------------
st.set_page_config(page_title="Socratic Sidekick", page_icon="🧠")

st.title("🧠 Socratic Sidekick DSA Mentor")
st.write("Ask a DSA question or paste your code.")


# -------------------------
# ENV
# -------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# -------------------------
# SESSION STATE
# -------------------------
if "reasoning_path" not in st.session_state:
    st.session_state.reasoning_path = []

if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0


# -------------------------
# EMOTION DETECTION
# -------------------------
def detect_emotion(text):
    text = text.lower()

    if any(w in text for w in ["fed up", "frustrated", "give up", "tired", "stuck"]):
        return "frustrated"

    if any(w in text for w in ["i think", "maybe", "not sure"]):
        return "confused"

    if any(w in text for w in ["yes", "got it", "understood"]):
        return "progress"

    return "neutral"


# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("🧠 Student Reasoning Path")

for step in st.session_state.reasoning_path:
    st.sidebar.write("•", step)


# -------------------------
# DATA
# -------------------------
@st.cache_data
def load_documents():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "data", "leetcode_dataset.csv")

    df = pd.read_csv(csv_path)

    docs = []
    for _, row in df.iterrows():
        docs.append(Document(page_content=" ".join(map(str, row.values))))

    return docs


documents = load_documents()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
text_chunks = splitter.split_documents(documents)


# -------------------------
# EMBEDDINGS
# -------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


# -------------------------
# VECTORSTORE
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
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding_model
    )


vectorstore = create_vectorstore()
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2})


# -------------------------
# LLM
# -------------------------
llm = Ollama(model="llama3", temperature=0.2)


# -------------------------
# PROMPT
# -------------------------
system_prompt = (
"You are a friendly Socratic DSA mentor.\n\n"

"STYLE:\n"
"• Be conversational and supportive\n"
"• Encourage only when appropriate (not always)\n"
"• Avoid repeating phrases like 'Nice thinking' unnecessarily\n\n"

"RULES:\n"
"• NEVER give code\n"
"• NEVER reveal the answer directly\n"
"• Ask ONE precise reasoning question\n"
"• Build on previous conversation\n\n"

"FORMAT CONTROL:\n"
"If format_type = full → give full explanation\n"
"If format_type = minimal → ONLY guide (no repetition)\n\n"

"FULL FORMAT:\n"
"Code Understanding\n"
"Problem Explanation\n"
"Reasoning Question\n"
"Hint\n"
"Next Step\n\n"

"MINIMAL FORMAT:\n"
"Reasoning Question\n"
"Hint\n"
"Next Step\n\n"

"Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Format: {format_type}\nConversation:\n{chat_history}\n\nUser:\n{input}")
    ]
)


# -------------------------
# CHAINS
# -------------------------
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)


# -------------------------
# CHAT INIT
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm Socratic Sidekick 🧠. Let's learn together 🚀"}
    ]


# -------------------------
# DISPLAY CHAT
# -------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------------
# USER INPUT
# -------------------------
user_prompt = st.chat_input("Ask a DSA question or paste your code...")


if user_prompt:

    st.session_state.turn_count += 1
    st.session_state.reasoning_path.append(user_prompt)

    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            chat_history = "\n".join(
                f"{m['role']}: {m['content']}" for m in st.session_state.messages
            )

            format_type = "full" if st.session_state.turn_count == 1 else "minimal"

            response = rag_chain.invoke({
                "input": user_prompt,
                "chat_history": chat_history,
                "format_type": format_type
            })

            answer = response["answer"]

            # -------------------------
            # EMOTION HANDLING
            # -------------------------
            emotion = detect_emotion(user_prompt)

            if emotion == "frustrated":
                prefix = "I understand this is frustrating — you're actually very close. Let's take it step by step 👇\n\n"

            elif emotion == "confused":
                prefix = "You're on the right track — let's clarify one small part 👇\n\n"

            elif emotion == "progress":
                prefix = "Good progress! You're thinking in the right direction 👍\n\n"

            else:
                prefix = ""

            # -------------------------
            # SAFETY FILTER
            # -------------------------
            if "def " in answer or "class " in answer:
                answer = "Let's focus on reasoning. What do you think might be wrong?"

            st.markdown(prefix + answer)

    st.session_state.messages.append({"role": "assistant", "content": prefix + answer})