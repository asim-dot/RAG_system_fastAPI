# frontend.py - Streamlit app that uses your API
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("📚 RAG System (via API)")

# Upload section
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    if st.button("Process"):
        # Send to API
        files = {"file": uploaded_file}
        response = requests.post(f"{API_URL}/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            st.session_state.session_id = result["session_id"]
            st.success(f"✅ Processed! Session: {result['session_id']}")

# Q&A section
if "session_id" in st.session_state:
    question = st.text_input("Ask a question:")
    
    if question:
        response = requests.post(
            f"{API_URL}/ask",
            json={
                "question": question,
                "session_id": st.session_state.session_id
            }
        )
        
        if response.status_code == 200:
            answer = response.json()
            st.write("**Answer:**", answer["answer"])
            st.caption(f"Used {answer['source_count']} sources")