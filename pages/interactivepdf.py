
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("💬 Chat with PDF")

uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_text = ""
    for f in uploaded_files:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + "\n"

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(all_text)

    # Embeddings (in memory)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(chunks, embeddings)  # stored only in RAM

    # Conversation chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    st.success("✅ PDF ready! Start chatting below:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask something about the PDF:")

    if query:
        with st.spinner("Thinking..."):
            result = qa({"question": query})
            answer = result["answer"]

            # Save conversation
            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("Bot", answer))

    # Show history
    for role, msg in st.session_state.chat_history:
        st.markdown(f"**{role}:** {msg}")

else:
    st.info("Upload at least one PDF to begin.")
