import os
import tempfile
import json
import numpy as np
import faiss
from openai import OpenAI
from streamlit_chat import message
import streamlit as st
from embedder import get_embedding_function
from faiss_indexer import build_faiss_index
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Phase 1 code: page config, title, uploader, etc.
st.set_page_config(page_title="AI Reading Partner", layout="centered")
st.title("AI Reading Partner — Phase 1, 2, 3 & 4")
st.caption("Upload a book, set your reading progress, parse it into chunks, generate embeddings, and chat about it!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# File uploader
uploaded_file = st.file_uploader("Upload a PDF or EPUB file", type=["pdf", "epub"])
if uploaded_file is not None:
    # Save the uploaded file to a temp location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    tfile.close()
    st.success(f"✔️ File uploaded: {uploaded_file.name}")
    uploaded_path = tfile.name

    # Input for book title (Phase 1)
    book_title = st.text_input("Enter book title", placeholder="e.g., Pride and Prejudice")
    if book_title:
        st.write("Selected title:", book_title)

    # Reading progress controls (Phase 1)
    st.subheader("Set your current reading progress")
    chapter_num = st.number_input("Current chapter (integer)", min_value=1, step=1)
    page_num = st.number_input("Current page (integer)", min_value=1, step=1)
    percent = st.slider("Percentage read (%)", min_value=0, max_value=100, value=0)
    if st.button("Save Progress"):
        st.success(f"Progress saved: Chapter {chapter_num}, Page {page_num}, {percent}% read.")

    # Phase 2 — Parse into chunks
    if st.button("Parse Book into Chunks"):
        with st.spinner("Parsing and chunking book…"):
            from parser import parse_and_chunk
            try:
                chunks_json_path = parse_and_chunk(uploaded_path, output_dir=".")
                st.session_state["chunks_path"] = chunks_json_path
                st.success(f"✔️ Done! Chunks written to `{chunks_json_path}`")
            except Exception as e:
                st.error(f"Error during parsing: {e}")

    # Phase 3 — Generate embeddings
    if "chunks_path" in st.session_state:
        if st.button("Generate Embeddings (RAG Base)"):
            with st.spinner("Generating embeddings and building vector store…"):
                try:
                    from embedder import embed_chunks_to_chroma
                    storage_dir = embed_chunks_to_chroma(st.session_state["chunks_path"])
                    st.success(f"✔️ Vector store created at: `{storage_dir}`")
                except Exception as e:
                    st.error(f"Error during embedding: {e}")

    # Phase 4 — Build FAISS index
    if "chunks_path" in st.session_state:
        if st.button("Build FAISS Index"):
            with st.spinner("Building FAISS index…"):
                try:
                    build_faiss_index(st.session_state["chunks_path"])
                    st.session_state["faiss_ready"] = True
                    st.success("✔️ FAISS index built successfully!")
                except Exception as e:
                    st.error(f"Error building FAISS index: {e}")

    # Phase 4 — Chat interface
    if "faiss_ready" in st.session_state and st.session_state["faiss_ready"]:
        st.subheader("Ask a spoiler-safe question")
        
        # Load FAISS index and metadata
        index = faiss.read_index("faiss.index")
        with open("faiss_metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Get user question
        user_question = st.text_input("Your question:", key="user_question")
        
        if user_question:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            # Get embedding for question
            embedding_function = get_embedding_function()
            question_embedding = np.array([embedding_function([user_question])[0]]).astype('float32')
            
            # Search FAISS index
            k = 20  # Get top 20 results
            distances, indices = index.search(question_embedding, k)
            
            # Filter out spoilers
            filtered_chunks = []
            for idx in indices[0]:
                chunk = metadata[idx]
                if (chunk["chapter"] is None or chunk["chapter"] <= chapter_num) and \
                   (chunk["percent"] is None or chunk["percent"] <= percent):
                    filtered_chunks.append(chunk)
                    if len(filtered_chunks) >= 5:  # Use up to 5 chunks
                        break
            
            # Build prompt
            prompt = f"Based on the following passages from the book, please answer the question. " \
                    f"Do not reveal any information beyond what's in these passages:\n\n"
            
            for chunk in filtered_chunks:
                prompt += f"Passage: {chunk['text']}\n\n"
            
            prompt += f"\nQuestion: {user_question}\n\nAnswer:"
            
            # Get response using local model
            try:
                # Initialize the local model
                if "chat_model" not in st.session_state:
                    try:
                        # Load model and tokenizer
                        model_name = "gpt2"
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        model = AutoModelForCausalLM.from_pretrained(model_name)
                        
                        # Create pipeline
                        st.session_state.chat_model = pipeline(
                            "text-generation",
                            model=model,
                            tokenizer=tokenizer,
                            max_length=200
                        )
                        st.success("Model loaded successfully!")
                    except Exception as model_error:
                        st.error(f"Error loading model: {str(model_error)}")
                        st.info("Please make sure you have a stable internet connection for the first run to download the model.")
                        st.stop()
                
                # Generate response
                try:
                    response = st.session_state.chat_model(
                        prompt,
                        max_length=200,
                        num_return_sequences=1,
                        temperature=0.7,
                        pad_token_id=st.session_state.chat_model.tokenizer.eos_token_id
                    )
                    assistant_response = response[0]['generated_text']
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                except Exception as gen_error:
                    st.error(f"Error generating response: {str(gen_error)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

        # Display chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                message(msg["content"], is_user=True)
            else:
                message(msg["content"]) 