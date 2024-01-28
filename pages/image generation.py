# ./codespace/pages/image generation.py

import streamlit as st
from streamlit_extras.app_logo import add_logo
from st_keyup import st_keyup

st.set_page_config(page_title="image generation", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("image generation")
add_logo("./llmonpy/pie.png", height=130)

from llmonpy import generation

image_list = generation.scan_dir('./images')
lora_list = generation.scan_dir("./loras")

with st.sidebar:
    notepad = st.text_area(label='notepad', label_visibility='collapsed')
    if st.session_state['enable_sdxl']:
        lora_selected = st.selectbox('lora for sdxl', lora_list)
    if st.session_state['img2img_on']:
        image_selected = st.selectbox('image for img2img', image_list)
    if st.session_state['enable_sdxl'] or st.session_state['img2img_on']:
        iter_count = st.slider('number of images', 1, 32, 1)

if 'image_pipe_turbo' not in st.session_state and st.session_state['enable_sdxl_turbo']:
    generation.popup_note(message='👊 hiting up sdxl turbo...')
    generation.load_sdxl_turbo()

if 'image_pipe_sdxl' not in st.session_state and st.session_state['enable_sdxl']:
    generation.popup_note(message='👊 hiting up sdxl 1.O...')
    generation.load_sdxl()

if 'img2img_pipe' not in st.session_state and st.session_state['img2img_on']:
    generation.popup_note(message='👊 hiting up sdxl turbo img2img...')
    generation.load_turbo_img2img()

if st.session_state['enable_sdxl_turbo']:
    turbo_prompt = st_keyup(label='real time(ish) image generation using sdxl turbo', debounce=1000) 
    generation.create_image_turbo(prompt=turbo_prompt)

if st.session_state['enable_sdxl']:
    sdxl_prompt = st.text_input(label='image generation with sdxl 1.0')
    if st.button('submit'):
        st.image(generation.create_image_sdxl(prompt=sdxl_prompt, iterations=iter_count, lora_name=lora_selected))
        
if st.session_state['img2img_on']:
    img2img_prompt = st.text_input(label='image to image generation with sdxl turbo')
    send_img2img_prompt = st.button("submit", key='img2img')
    st.image(f"./images/{image_selected}")
    if send_img2img_prompt:
        st.image(generation.create_image_img2img(from_image=image_selected, prompt=img2img_prompt, iterations=iter_count))
