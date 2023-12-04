# ./codespace/pages/image generation.py

import streamlit as st
from streamlit_extras.app_logo import add_logo

st.set_page_config(page_title="image generation", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("llmon-py - image generation")
add_logo("./llmon_art/lemon (12).png")
st.divider()

import GPUtil as GPU
import time
import keyboard
import torch
from diffusers import AutoPipelineForText2Image

enable_popups = st.session_state.enable_popups
enable_sdxl = st.session_state.enable_sdxl
enable_cpu_only = st.session_state.enable_cpu_only
GPUs = GPU.getGPUs()
gpu = GPUs[0]
popup_delay = 1.0

def popup_note(message=str):
    if enable_popups:
        st.toast(message)
        time.sleep(popup_delay)

with st.sidebar:
    notepad = st.text_area(label='notepad', label_visibility='collapsed')
    steps = st.slider('steps', 1, 32, 1)
    
if 'image_pipe' not in st.session_state and enable_sdxl:
    popup_note(message='👊 hiting up sdxl turbo...')
    st.session_state.image_pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    if not enable_cpu_only:
        st.session_state.image_pipe.to('cuda')

def create_image(image_prompt=str):
    image = st.session_state.image_pipe(prompt=image_prompt, num_inference_steps=steps, guidance_scale=0.0).images[0]
    image.save(f'_image.png')
    time.sleep(0.1)
    st.image(f'_image.png')

def on_space():
    keyboard.send('enter')

check_vram = float("{0:.0f}".format(gpu.memoryUsed)) / float("{0:.0f}".format(gpu.memoryTotal))
if check_vram > 0.85:
    st.warning(body='🔥 vram limit is being reached')
st.progress(float("{0:.0f}".format(gpu.memoryFree)) / float("{0:.0f}".format(gpu.memoryTotal)), "vram {0:.0f}/{1:.0f}mb".format(gpu.memoryUsed, gpu.memoryTotal))

if enable_sdxl:
    keyboard.add_hotkey('space', on_space)
    image_prompt = st.text_input(label='real time(ish) image generation')
    create_image(image_prompt)
