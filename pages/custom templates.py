# ./codespace/pages/custom templates.py
import streamlit as st
from llmonpy import llmonaid

st.set_page_config(page_title="custom templates", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("🍋custom templates")

if st.session_state['approved_login'] == True:
    with st.sidebar:
        llmonaid.memory_display()
else:
    st.image("./llmonpy/pie.png", caption="please login to continue")