# ./codespace/pages/image generation.py

import streamlit as st
from streamlit_extras.app_logo import add_logo
from st_keyup import st_keyup

st.set_page_config(page_title="image generation", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("llmon-py - image generation")
add_logo("./llmon_art/pie.png", height=130)

import time
import torch
from diffusers import DiffusionPipeline, AutoPipelineForText2Image
from datetime import datetime

enable_popups = st.session_state['enable_popups']
enable_sdxl_turbo = st.session_state['enable_sdxl_turbo']
enable_sdxl = st.session_state['enable_sdxl']
lora_to_load = st.session_state['lora_to_load']
enable_cpu_only = st.session_state['enable_cpu_only']
lora_path = "./loras"

def popup_note(message=str):
    if enable_popups:
        st.toast(message)

default_step = 0
if enable_sdxl:
    default_step = 50
if enable_sdxl_turbo:
    default_step = 1

with st.sidebar:
    st.text(f"lora: {lora_to_load}")
    notepad = st.text_area(label='notepad', label_visibility='collapsed')
    steps = st.slider('steps', 1, 50, default_step)
    iter_count = st.slider('number of images', 1, 32, 1)
    gen_buffer = st.slider('sdxl turbo gen buffer', 50, 5000, 1000)
    
if 'image_pipe_sdxl' not in st.session_state and enable_sdxl:
    popup_note(message='👊 hiting up sdxl 1.O...')
    st.session_state.image_pipe_sdxl = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to('cuda')
    st.session_state.image_pipe_sdxl.load_lora_weights(pretrained_model_name_or_path_or_dict=lora_path, weight_name=lora_to_load)
    st.session_state.image_pipe_sdxl.enable_vae_slicing()
    st.session_state.image_pipe_sdxl.enable_model_cpu_offload()

if 'image_pipe_turbo' not in st.session_state and enable_sdxl_turbo:
    popup_note(message='👊 hiting up sdxl turbo...')
    st.session_state.image_pipe_turbo = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    if not enable_cpu_only:
        st.session_state.image_pipe_turbo.to('cuda')

def create_image_turbo():
    image_prompt = st_keyup(label='real time(ish) image generation using sdxl turbo', debounce=gen_buffer)
    image = st.session_state.image_pipe_turbo(prompt=image_prompt, num_inference_steps=steps, guidance_scale=0.0).images[0]

    image.save('image_turbo.png')
    time.sleep(0.2)
    st.image('image_turbo.png')

def create_image_sdxl(image_prompt=str, image_count=int):
    
    image_list = []
    while image_count:     
        image = st.session_state.image_pipe_sdxl(prompt=image_prompt, num_inference_steps=50).images[0]
        output_file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        image.save(f"{output_file_name}-{image_count}.png")
        image_list.append(f"{output_file_name}-{image_count}.png")
        st.image(image_list)
        image_count = image_count - 1
        if len(image_list) > 32:
            image_list = []

if enable_sdxl_turbo:
    create_image_turbo()

if enable_sdxl:
    sdxl_prompt = st.text_input(label='image generation with sdxl 1.0')
    if st.button('submit'):
        create_image_sdxl(image_prompt=sdxl_prompt, image_count=iter_count)
