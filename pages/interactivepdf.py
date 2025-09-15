# streamlit: title = "📄🦜🔍 Talking PDF"
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAI
import tempfile
import os

st.set_page_config(page_title="💬 Chat with PDF", layout="wide")
st.title("💬 Chat with PDF")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Extract text
    pdf_reader = PdfReader(temp_path)
    text = "".join(page.extract_text() or "" for page in pdf_reader.pages)

    if not text.strip():
        st.error("Could not extract text from PDF.")
    else:
        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embeddings)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # Gemini LLM
        llm = GoogleGenerativeAI(model="gemini-2.5-flash")
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

        st.success("✅ PDF processed! You can now ask questions.")

        # Store chat history in session
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        query = st.text_input("Ask a question about the PDF:")

        if query:
            result = qa_chain({"question": query, "chat_history": st.session_state.chat_history})
            answer = result["answer"]

            st.session_state.chat_history.append((query, answer))

            for q, a in st.session_state.chat_history:
                st.markdown(f"**You:** {q}")
                st.markdown(f"**AI:** {a}")

    # Cleanup
    os.remove(temp_path)
