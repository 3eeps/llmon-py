# ./codespace/pages/image generation.py
import streamlit as st
from st_keyup import st_keyup

st.set_page_config(page_title="image generation", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("🍋image generation")

from llmonpy import generation, llmonaid

if st.session_state['approved_login'] == True:
    llmonaid.memory_display()
    with st.sidebar:
        if st.session_state['img2img_on']:
            uploaded_file = st.file_uploader(label="Choose a image file")
            if uploaded_file is not None:
                st.session_state.bytes_data = uploaded_file.getvalue()
            st.session_state['img2img_steps'] = st.slider('number of steps', 1, 50, st.session_state['img2img_steps'], key='img2img_step_key')
            st.session_state['img2img_iter_count'] = st.slider('number of images', 1, 32, st.session_state['img2img_iter_count'], key='img2img_iter_key')

        if st.session_state['enable_sdxl']:
            st.session_state['sdxl_steps'] = st.slider('number of steps', 1, 50, st.session_state['sdxl_steps'], key='sdxl_step_key')
            st.session_state['sdxl_iter_count'] = st.slider('number of images', 1, 32, st.session_state['sdxl_iter_count'], key='sdxl_iter_key')

        if st.session_state['enable_sdxl_turbo']:
            st.session_state['sdxl_turbo_steps'] = st.slider('number of steps', 1, 32, st.session_state['sdxl_turbo_steps'], key='turbo_step_key')

    if 'image_pipe_turbo' not in st.session_state and st.session_state['enable_sdxl_turbo']:
        llmonaid.popup_note(message='👊 hiting up sdxl turbo...')
        generation.load_sdxl_turbo()

    if 'image_pipe_sdxl' not in st.session_state and st.session_state['enable_sdxl']:
        llmonaid.popup_note(message='👊 hiting up sdxl 1.O...')
        generation.load_sdxl(lora_name=st.session_state['lora_selected'])

    if 'img2img_pipe' not in st.session_state and st.session_state['img2img_on']:
        llmonaid.popup_note(message='👊 hiting up sdxl turbo img2img...')
        generation.load_turbo_img2img()

    if st.session_state['enable_sdxl_turbo']:
        st.session_state['turbo_prompt'] = st_keyup(label='real time(ish) image generation with sdxl turbo', debounce=1000, value=st.session_state['turbo_prompt'], )
        generation.create_image_turbo(prompt=st.session_state['turbo_prompt'])

    if st.session_state['enable_sdxl']:
        st.session_state['sdxl_prompt'] = st.text_area(label='image generation with sdxl 1.0', value=st.session_state['sdxl_prompt'])
        if st.button('submit'):
            st.image(generation.create_image_sdxl(prompt=st.session_state['sdxl_prompt'], iterations=st.session_state['sdxl_iter_count'], steps=st.session_state['sdxl_steps']))
        
    if st.session_state['img2img_on']:
        st.session_state['img2img_prompt'] = st.text_input(label='image to image generation with sdxl turbo', value=st.session_state['img2img_prompt'])
        send_img2img_prompt = st.button("submit", key='img2img')
        if st.session_state.bytes_data is not None:
            show_image = st.image(st.session_state.bytes_data, clamp=True)
        else:
            show_image = st.image('./llmonpy/pie.png', clamp=True)
        if send_img2img_prompt:
            st.image(image=f"./{generation.create_image_img2img(prompt=st.session_state['img2img_prompt'], steps=st.session_state['img2img_steps'])}", clamp=True)

if st.session_state['approved_login'] == False:
    st.image('./llmonpy/pie.png', caption='please login to continue')
if st.session_state['enable_sdxl'] == False and st.session_state['enable_sdxl_turbo'] == False and st.session_state['img2img_on'] == False:
    st.image('./llmonpy/pie.png', caption="enable an image model to continue")