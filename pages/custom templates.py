# ./codespace/pages/custom templates.py

import streamlit as st
from streamlit_extras.app_logo import add_logo
import keyboard
import GPUtil as GPU
import time

keyboard.unhook_all()
st.set_page_config(page_title="custom templates", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("llmon-py - custom templates")
add_logo("./llmon_art/lemon (15).png")
st.divider()

GPUs = GPU.getGPUs()
gpu = GPUs[0]

def popup_note(message=str):
        st.toast(message)
        time.sleep(2.0)

check_vram = float("{0:.0f}".format(gpu.memoryUsed)) / float("{0:.0f}".format(gpu.memoryTotal))
if check_vram > 0.85:
    popup_note(message='😭 vram limit is being reached')
st.progress(float("{0:.0f}".format(gpu.memoryFree)) / float("{0:.0f}".format(gpu.memoryTotal)), "vram {0:.0f}/{1:.0f}mb".format(gpu.memoryUsed, gpu.memoryTotal))

col1, col2 = st.columns(2)
with col1:
   st.header("custom template 0")
   custom_template0 = st.text_area(label='custom0')

with col2:
   st.header("custom template 1")
   custom_template1 = st.text_area(label='custom1')
