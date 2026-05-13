# app.py
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import tempfile
import os

st.title('📄 PDF Q&A with Gemini')

# ====================== API KEY HANDLING ======================
if 'gemini_key' not in st.session_state:
    st.session_state.gemini_key = ""

# Sidebar for API Key
with st.sidebar:
    st.header("🔑 Configuration")
    
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",           # This hides the key
        value=st.session_state.gemini_key,
        placeholder="Enter your Gemini API key here"
    )
    
    if st.button("Save Key"):
        if api_key_input:
            st.session_state.gemini_key = api_key_input
            st.success("✅ API Key saved!")
        else:
            st.error("Please enter a valid key")

# Show status
if st.session_state.gemini_key:
    st.sidebar.success("✓ Key is set")
else:
    st.sidebar.warning("Enter your Gemini API Key to continue")

# ====================== Main App ======================
uploaded_file = st.file_uploader('Upload your PDF', type='pdf')

if uploaded_file and st.session_state.gemini_key:
    
    # Save uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # Load and process PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)
        texts = [c.page_content for c in chunks]

        # Embeddings + FAISS
        with st.spinner("Processing PDF..."):
            model_emb = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model_emb.encode(texts).astype('float32')

            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)

        st.success(f'✅ PDF processed! {len(texts)} chunks indexed.')

        # Question
        question = st.text_input('Ask a question about your PDF:')
        
        if st.button('Get Answer') and question:
            # Retrieve relevant chunks
            q_emb = model_emb.encode([question]).astype('float32')
            _, indices = index.search(q_emb, k=4)  # increased to 4
            context = '\n\n'.join([texts[i] for i in indices[0]])

            # Call Gemini
            genai.configure(api_key=st.session_state.gemini_key)
            llm = genai.GenerativeModel('gemini-1.5-flash')

            prompt = f"""Answer the question using only the provided context.
            Context:
            {context}

            Question: {question}
            Answer:"""

            with st.spinner('Gemini is thinking...'):
                response = llm.generate_content(prompt)
                st.write("**Answer:**")
                st.write(response.text)

            with st.expander("📋 View Context Used"):
                st.write(context)

    finally:
        os.unlink(tmp_path)

else:
    if not st.session_state.gemini_key:
        st.info("👈 Please enter your Gemini API Key in the sidebar")
