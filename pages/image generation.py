# ./codespace/pages/image generation.py
import streamlit as st
from streamlit_extras.app_logo import add_logo
from st_keyup import st_keyup

st.set_page_config(page_title="image generation", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("🍋image generation")
add_logo("./llmonpy/pie.png", height=130)

from llmonpy import generation, llmonaid

if st.session_state['approved_login'] == True:
    lora_list = llmonaid.scan_dir("./loras")
    lora_selected = str
    iter_count = 1
    steps = 1
    with st.sidebar:
        notepad = st.text_area(label='notepad', label_visibility='collapsed')
        if st.session_state['enable_sdxl']:
            lora_selected = st.selectbox('lora for sdxl', lora_list)
        if st.session_state['img2img_on']:
            uploaded_file = st.file_uploader(label="Choose a image file")
            if uploaded_file is not None:
                st.session_state.bytes_data = uploaded_file.getbuffer
            else:
                st.session_state.buffer = ""
            steps= st.slider('number of steps', 1, 50, 1)
            iter_count = st.slider('number of images', 1, 32, 1)

        if st.session_state['enable_sdxl']:
            steps= st.slider('number of steps', 1, 50, 1)
            iter_count = st.slider('number of images', 1, 32, 1)

        if st.session_state['enable_sdxl_turbo']:
            steps= st.slider('number of steps', 1, 32, 1)

    if 'image_pipe_turbo' not in st.session_state and st.session_state['enable_sdxl_turbo']:
        llmonaid.popup_note(message='👊 hiting up sdxl turbo...')
        generation.load_sdxl_turbo()

    if 'image_pipe_sdxl' not in st.session_state and st.session_state['enable_sdxl']:
        llmonaid.popup_note(message='👊 hiting up sdxl 1.O...')
        generation.load_sdxl(lora_name=lora_selected)

    if 'img2img_pipe' not in st.session_state and st.session_state['img2img_on']:
        llmonaid.popup_note(message='👊 hiting up sdxl turbo img2img...')
        generation.load_turbo_img2img()

    if st.session_state['enable_sdxl_turbo']:
        turbo_prompt = st_keyup(label='real time(ish) image generation using sdxl turbo', debounce=1000) 
        generation.create_image_turbo(prompt=turbo_prompt)

    if st.session_state['enable_sdxl']:
        sdxl_prompt = st.text_input(label='image generation with sdxl 1.0')
        if st.button('submit'):
            st.image(generation.create_image_sdxl(prompt=sdxl_prompt, iterations=iter_count, steps=steps))
        
    if st.session_state['img2img_on']:
        img2img_prompt = st.text_input(label='image to image generation with sdxl turbo')
        send_img2img_prompt = st.button("submit", key='img2img')
        if st.session_state.bytes_data is not None:
            show_image = st.image(st.session_state.bytes_data, clamp=True)
        else:
            show_image = st.image('./llmonpy/pie.png', clamp=True)
        if send_img2img_prompt:
            # from_image=st.session_state.bytes_data, prompt=img2img_prompt, steps=steps, iterations=iter_count
            st.image(generation.create_image_img2img(from_image=show_image))
else:
    st.image('./llmonpy/pie.png', caption='please login to continue')
