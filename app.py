# app.py
import streamlit as st

st.set_page_config(page_title="Sample App", layout="centered")

st.title("Sample App")
name = st.text_input("Han")
if name:
    st.write(f"Hello, {name}")