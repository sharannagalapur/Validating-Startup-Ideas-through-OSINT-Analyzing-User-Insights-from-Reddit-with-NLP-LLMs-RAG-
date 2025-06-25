import streamlit as st

# Set page config first
st.set_page_config(page_title="AI RAG Chatbot", layout="wide")

import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()




if not os.getenv("LANGCHAIN_API_KEY"):
    st.error("LANGCHAIN_API_KEY not found. Please check your .env file.")
    st.stop()

# Optional static config
os.environ["LANGCHAIN_TRACING_V2"] = "true"


# Prompt for Groq API key if not already set
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = st.text_input("Enter your Groq API key", type="password")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("rag_dataset.csv")
    df.drop(columns=['_id', 'id', 'ups', 'subreddit', 'created_utc', 'num_comments', 'url', 'response'], inplace=True, errors='ignore')
    df["combined_text"] = df["title"].fillna('') + " " + df["selftext"].fillna('') + " " + df["comments"].fillna('')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Convert text to documents
documents = [Document(page_content=text) for text in df["combined_text"].tolist()]

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)


# Load embeddings and FAISS
embeddings = OllamaEmbeddings(model="nomic-embed-text")
new_db = FAISS.load_local("Faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = new_db.as_retriever()


# Load LLM

llm = ChatGroq(model="llama-3.1-8b-instant")


# Prompts
system_prompt = "You are an AI assistant. Use the retrieved context to answer the question.\n\n{context}"

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question, reformulate it as a standalone question."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create chains
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Message history store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap chain with message history
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Streamlit UI
st.title("🔍 AI-Powered RAG Chatbot")

# Session state setup
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_session_1"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
query = st.chat_input("Ask me anything about AI...")

if query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # AI response
    try:
        response = conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": st.session_state.session_id}},
        )["answer"]

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    except Exception as e:
        st.error(f"Error during response generation: {e}")

