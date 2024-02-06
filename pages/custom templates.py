# ./codespace/pages/custom templates.py
import streamlit as st
from streamlit_extras.app_logo import add_logo
from llmonpy import llmonaid
import GPUtil as GPU

st.set_page_config(page_title="custom templates", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("🍋custom templates")
add_logo("./llmonpy/pie.png", height=130)

if st.session_state['approved_login'] == True:
    GPUs = GPU.getGPUs()
    gpu = GPUs[0]
    st.progress((100 / gpu.memoryTotal) / (100 / int(gpu.memoryUsed)), "vram {0:.0f}/{1:.0f}mb".format(gpu.memoryUsed, gpu.memoryTotal))
    pass
else:
    st.image("./llmonpy/pie.png", caption="please login to continue")