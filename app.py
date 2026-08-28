import streamlit as st
from supervisor import run_pipeline

st.set_page_config(page_title="Multi-Agent RAG Chatbot", page_icon="🤖")
st.title("🤖 Multi-Agent RAG Chatbot")
st.caption("Supervisor routes your question to PDF-RAG, SQL, and/or Web Search agents")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask about Lionsgate earnings, IMDb movies, or current news...")

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        st.caption(f"Tools used: {', '.join(entry['tools_called'])}")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_pipeline(question)
        st.write(result["final_answer"])
        st.caption(f"Tools used: {', '.join(result['tools_called'])}")

    st.session_state.history.append({
        "question": question,
        "answer": result["final_answer"],
        "tools_called": result["tools_called"],
    })