import streamlit as st
import requests


st.title("📄 Chat with Your Documents")

query = st.text_input("Ask a question:")

if st.button("Ask") and query:
    response = requests.get(
        "http://backend:8000/ask_llm_agent?query=" + query
    )

    data = response.json()

    st.write("### 🤖 Answer")
    st.write(data["answer"])
    
    if "sources" in data:
        st.write("### 📚 Sources")
        for s in data["sources"]:
            st.write("-", s)
