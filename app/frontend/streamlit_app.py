import streamlit as st
import requests

st.set_page_config(page_title="Healthcare Assistant")

st.title("Healthcare Knowledge Assistant")

query = st.chat_input("Ask medical question")

if query:

    with st.chat_message("user"):
        st.write(query)

    response = requests.post(
        "http://localhost:8000/query",
        json={"query": query}
    )

    result = response.json()

    with st.chat_message("assistant"):
        st.write(result["response"])
