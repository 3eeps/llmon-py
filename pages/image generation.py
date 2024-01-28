# ./codespace/pages/image generation.py

import streamlit as st
from streamlit_extras.app_logo import add_logo
from st_keyup import st_keyup
import os

st.set_page_config(page_title="image generation", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("llmon-py - image generation")
add_logo("./llmon_art/pie.png", height=130)

import time
import torch
from diffusers import DiffusionPipeline, AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image
from datetime import datetime

def scan_dir(directory):
    directory_list = []
    count = 0
    for file in os.scandir(f'{directory}'):
        if file.is_file():
            directory_list.append(file.name)
            count += count
    return directory_list

def popup_note(message=str):
    if st.session_state['enable_popups']:
        st.toast(message)

lora_path = "./loras"
image_list = scan_dir('./images')

default_step = 0
if st.session_state['enable_sdxl']:
    default_step = 50
if st.session_state['enable_sdxl_turbo']:
    default_step = 1
if st.session_state['img2img_on']:
    default_step = 1

with st.sidebar:
    st.text(f"lora: {st.session_state['lora_to_load']}")
    notepad = st.text_area(label='notepad', label_visibility='collapsed')
    if st.session_state['img2img_on']:
        image_selected = st.selectbox('image for img2img', image_list)
    steps = st.slider('steps', 1, 50, default_step)
    iter_count = st.slider('number of images', 1, 32, 1)
    gen_buffer = st.slider('sdxl turbo gen buffer', 50, 5000, 1000)

if 'img2img_pipe' not in st.session_state and st.session_state['img2img_on']:   
    st.session_state['img2img_pipe'] = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")

if 'image_pipe_sdxl' not in st.session_state and st.session_state['enable_sdxl']:
    popup_note(message='👊 hiting up sdxl 1.O...')
    st.session_state.image_pipe_sdxl = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to('cuda')
    st.session_state.image_pipe_sdxl.load_lora_weights(pretrained_model_name_or_path_or_dict=lora_path, weight_name=st.session_state['lora_to_load'])
    st.session_state.image_pipe_sdxl.enable_vae_slicing()
    st.session_state.image_pipe_sdxl.enable_model_cpu_offload()

if 'image_pipe_turbo' not in st.session_state and st.session_state['enable_sdxl_turbo']:
    popup_note(message='👊 hiting up sdxl turbo...')
    st.session_state.image_pipe_turbo = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
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

def create_img2img(prompt=str, image_count=int):
    while image_count:
        init_image = load_image(f"./images/{image_selected}").resize((512, 512))
        image = st.session_state['img2img_pipe'](prompt, image=init_image, num_inference_steps=steps, strength=0.5, guidance_scale=0.0).images[0]
        output_file_name = datetime.now().strftime("img2img-%d-%m-%Y-%H-%M-%S")
        image.save(f"img2img-{output_file_name}.png")
        image_count = image_count - 1

if st.session_state['enable_sdxl_turbo']: 
    create_image_turbo()

if st.session_state['enable_sdxl']:
    sdxl_prompt = st.text_input(label='image generation with sdxl 1.0')
    if st.button('submit'):
        create_image_sdxl(image_prompt=sdxl_prompt, image_count=iter_count)

if st.session_state['img2img_on']:
    img2img_prompt = st.text_input(label='image to image generation with sdxl turbo')
    img2img_button = st.button("submit", key='img2img')
    st.image(f"./images/{image_selected}")
    if img2img_button:
        create_img2img(prompt=img2img_prompt, image_count=iter_count)
