from langchain_community.llms import ollama
import streamlit as st

llm = ollama(model = "llama3")

st.title("chatbot using llama3")

prompt = st.text_area("enter your prompt:")

if st.button("generate"):
    if prompt:
        with st.spinner("Generating response:"):
            st.write(llm.invoke(prompt,stop=['<|eot_id|>']))



