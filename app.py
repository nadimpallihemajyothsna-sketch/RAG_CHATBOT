import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot")
st.title("🤖 PDF Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:

    # Read PDF
    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector store
    vector_store = FAISS.from_texts(
        chunks,
        embedding=embeddings
    )

    # User Question
    user_question = st.text_input("Ask a question from PDF")

    if user_question:

        # Search similar chunks
        docs = vector_store.similarity_search(user_question)

        # Combine context
        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        # Limit context size
        context = context[:500]

        # Prompt
        prompt = f"""
Answer the question based on the context below.

Context:
{context}

Question:
{user_question}
"""

        # Load local model
        generator = pipeline(
            task="text-generation",
            model="distilgpt2"
        )

        # Generate answer
        result = generator(
            prompt,
            max_new_tokens=50
        )

        # Display answer
        st.write("### Answer")
        st.write(result[0]["generated_text"])