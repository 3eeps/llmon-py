# ./codespace/pages/custom templates.py
import streamlit as st
from streamlit_extras.app_logo import add_logo
from llmonpy import llmonaid

st.set_page_config(page_title="custom templates", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("🍋custom templates")
add_logo("./llmonpy/pie.png", height=130)

if st.session_state['approved_login'] == True:
    llmonaid.memory_display()
else:
    st.image("./llmonpy/pie.png", caption="please login to continue")