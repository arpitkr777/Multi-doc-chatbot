import os
import json
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# =========================
# CONFIG SETUP
# =========================
working_dir = os.path.dirname(os.path.abspath(__file__))

with open(f"{working_dir}/config.json", "r") as f:
    config_data = json.load(f)

GROQ_API_KEY = config_data.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ API key missing in config.json")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# =========================
# VECTOR STORE SETUP
# =========================
def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    return vectorstore

# =========================
# CHAT CHAIN SETUP
# =========================
def chat_chain(vectorstore):
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}   # 🔥 IMPORTANT FIX
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'

    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=True,
        return_source_documents=True   # 🔥 IMPORTANT
    )

    return chain

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="Multi Doc Chatbot",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Multi Document Chatbot")

# =========================
# SESSION STATE
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = chat_chain(
        st.session_state.vectorstore
    )

# =========================
# DISPLAY CHAT HISTORY
# =========================
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# USER INPUT
# =========================
user_input = st.chat_input("Ask a question about the documents...")

if user_input:
    # USER MESSAGE
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # ASSISTANT RESPONSE
    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({
            "question": user_input
        })

        assistant_response = response["answer"]

        # =========================
        # DEBUG (TERMINAL OUTPUT)
        # =========================
        print("\n====================")
        print("SOURCE DOCUMENTS:\n")

        for i, doc in enumerate(response["source_documents"]):
            print(f"\n--- Document {i+1} ---")
            print(doc.page_content[:500])
            print("----------------------")

        # =========================
        # SHOW ANSWER
        # =========================
        st.markdown(assistant_response)

        # =========================
        # SHOW SOURCES IN UI (🔥 IMPORTANT)
        # =========================
        st.write("### 📄 Retrieved Context:")

        for doc in response["source_documents"]:
            st.write(doc.page_content[:300])

        # SAVE TO HISTORY
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": assistant_response
        })