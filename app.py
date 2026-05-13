# app.py
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import os
from pypdf import PdfReader

st.title("📄 PDF Q&A with Gemini")

# ====================== API KEY ======================
if 'gemini_key' not in st.session_state:
    st.session_state.gemini_key = ""

with st.sidebar:
    st.header("🔑 Configuration")
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.gemini_key,
        placeholder="Paste your key here"
    )
    
    if st.button("Save Key"):
        if api_key_input.strip():
            st.session_state.gemini_key = api_key_input.strip()
            st.success("✅ API Key Saved!")
        else:
            st.error("Key cannot be empty")

if st.session_state.gemini_key:
    st.sidebar.success("✓ Key is ready")
else:
    st.sidebar.warning("Enter Gemini API Key")

# ====================== FILE UPLOAD ======================
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and st.session_state.gemini_key:
    
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # ================== EXTRACT TEXT WITHOUT LANGCHAIN ==================
        with st.spinner("Extracting text from PDF..."):
            reader = PdfReader(tmp_path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"

        # ================== SIMPLE CHUNKING ==================
        def chunk_text(text, chunk_size=500, overlap=50):
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end])
                start += chunk_size - overlap
            return chunks

        texts = chunk_text(full_text)

        # ================== EMBEDDINGS + FAISS ==================
        with st.spinner("Creating embeddings and index..."):
            model_emb = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model_emb.encode(texts).astype('float32')

            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)

        st.success(f"✅ PDF processed! {len(texts)} chunks created.")

        # ================== QUESTION ANSWERING ==================
        question = st.text_input("Ask a question about your PDF:")

        if st.button("Get Answer") and question:
            # Retrieve relevant chunks
            q_emb = model_emb.encode([question]).astype('float32')
            _, indices = index.search(q_emb, k=4)

            context = "\n\n".join([texts[i] for i in indices[0]])

            # Call Gemini
            genai.configure(api_key=st.session_state.gemini_key)
            llm = genai.GenerativeModel('gemini-1.5-flash')

            prompt = f"""Use only the following context to answer the question.
            
Context:
{context}

Question: {question}

Answer:"""

            with st.spinner("Gemini is thinking..."):
                response = llm.generate_content(prompt)
                st.subheader("Answer:")
                st.write(response.text)

            with st.expander("📋 Context Used"):
                st.write(context)

    finally:
        os.unlink(tmp_path)

else:
    if not st.session_state.gemini_key:
        st.info("👈 Please enter your Gemini API Key in the sidebar to continue.")
