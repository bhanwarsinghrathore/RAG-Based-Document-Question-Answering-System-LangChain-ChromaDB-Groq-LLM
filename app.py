import streamlit as st

from rag_generation import ask_question

from pathlib import Path

st.title("📚 RAG Chatbot")

query = st.text_input(
    label="Ask a Question",
    placeholder="e.g. What is DNA replication?"
)

if query and query.strip():

    answer, docs = ask_question(query.strip())

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")

    for i, doc in enumerate(docs, 1):

        st.write(
            f"Source {i}: {Path(doc.metadata.get('source')).name}"
        )