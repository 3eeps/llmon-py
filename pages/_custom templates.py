# ./codespace/pages/_custom templates.py

import streamlit as st
from streamlit_extras.app_logo import add_logo
import keyboard
import time
import GPUtil as GPU

keyboard.unhook_all()
st.set_page_config(page_title="custom templates", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("llmon-py _custom templates")
add_logo("./llmon_art/lemon (15).png", height=150)
st.divider()

def popup_note():
        st.toast(':red[vram usage over 85%]')
        time.sleep(2.5)

GPUs = GPU.getGPUs()
gpu = GPUs[0]
vram_usage = float("{0:.0f}".format(gpu.memoryFree)) / float("{0:.0f}".format(gpu.memoryTotal))
if vram_usage > 0.85:
    popup_note()

st.progress(float("{0:.0f}".format(gpu.memoryFree)) / float("{0:.0f}".format(gpu.memoryTotal)), "vram {0:.0f}/{1:.0f}mb".format(gpu.memoryUsed, gpu.memoryTotal))
col1, col2 = st.columns(2)
with col1:
   st.header("custom template 0")
   custom_template0 = st.text_area(label='custom0')

with col2:
   st.header("custom template 1")
   custom_template1 = st.text_area(label='custom1')
