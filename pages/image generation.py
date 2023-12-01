# ./codespace/pages/image generation.py

import streamlit as st
from streamlit_extras.streaming_write import write as stream_write
from threading import Thread
import time
import torch
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image

def popup_note(message=str, delay=int):
    if st.session_state.enable_popups == 'yes':
        st.toast(message)
        time.sleep(delay)

def stream_text(text=str):
    for word in text.split():
        yield word + " "
        if st.session_state.text_stream_speed != 0:
            speed = (st.session_state.text_stream_speed / 10)
            time.sleep(speed)

if 'image_pipe' not in st.session_state and st.session_state.enable_sdxl == 'yes':
    popup_note(message='👊 hiting up sdxl turbo...', delay=1.0)
    st.session_state.image_pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    st.session_state.image_pipe.to('cuda')

def create_image(image_prompt=str):
    popup_note(message=f'🗿 generating image...', delay=1.0)  
    image = st.session_state.image_pipe(prompt=image_prompt, num_inference_steps=st.session_state.steps, guidance_scale=0.0).images[0]
    image.save(f'sdxl_image.png')
    popup_note(message=f'🙅 image created...', delay=1.0)

if st.session_state.enable_sdxl == 'yes':
    image_input = st.text_input(label='sdxl turbo')
    if image_input != "":
        create_image(image_prompt=image_input)
        st.image(f'sdxl_image.png')
    stream_write(stream_text(image_input))
    
