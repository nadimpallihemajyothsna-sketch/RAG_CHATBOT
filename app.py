import streamlit as st

from ingest import process_pdf
from vector_store import create_vector_store
from chat import get_answer

#sidebar
st.sidebar.title("RAG chatbot")
st.sidebar.write(
    "AI-powered PDF Question Answering System"
)

st.set_page_config(page_title="RAG Chatbot")
st.title("📄 RAG PDF Chatbot")
st.markdown(
    "Upload a PDF and aask questions based on the document."
)

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

    chunks = process_pdf(uploaded_file)

    vector_store = create_vector_store(chunks)

    user_question = st.text_input(
        "Ask a question from PDF"
    )

    if user_question:

        docs = vector_store.similarity_search(
            user_question
        )

        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        context = context[:500]
        
        with st.spinner("Generating aanswer..."):            
          answer = get_answer(
            context,
            user_question
        )

        st.success("PDF uploaded successfully!")
        st.write("### Answer")
        st.info(answer)