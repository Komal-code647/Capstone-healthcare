import streamlit as st
import requests


st.set_page_config(
    page_title="Healthcare Knowledge Assistant",
    layout="centered"
)

st.title("Healthcare Knowledge Assistant")

query = st.text_area(
    "Describe your symptoms",
    height=150
)

if st.button("Get Medical Advice"):

    if query.strip() == "":
        st.warning("Please enter symptoms.")
    else:

        with st.spinner("Analyzing symptoms..."):

            try:

                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"query": query}
                )

                result = response.json()

                st.subheader("Response")

                st.write(result["response"])

            except Exception as e:

                st.error(f"Error: {e}")
