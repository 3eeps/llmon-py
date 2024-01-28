# ./codespace/pages/custom templates.py

import streamlit as st
from streamlit_extras.app_logo import add_logo

st.set_page_config(page_title="custom templates", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("image reconition")
add_logo("./llmon_art/pie.png", height=130)

from moondream import VisionEncoder, TextModel
from PIL import Image
import os
from threading import Thread
from transformers import TextIteratorStreamer
import re
import time

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

image_list = scan_dir('./images')
with st.sidebar:
    notepad = st.text_area(label='notepad', label_visibility='collapsed')
    image_selected = st.selectbox('pick image', image_list)

model_path = "c:\codespace\moondream1"

if 'vision_encoder' not in st.session_state or st.session_state.vision_encoder == None:
    popup_note(message='👀 looking at you moondream1...')
    st.session_state.vision_encoder = VisionEncoder(model_path, run_on=st.session_state['ocr_device'])
    st.session_state.text_model = TextModel(model_path, run_on=st.session_state['ocr_device'])

moondream_prompt = st.text_input(label="image reconition with :orange[moondream1]")
button = st.button("submit")

st.image(f"./images/{image_selected}")

if button:
    ocr_time = time.time()
    image = Image.open(f"c:\codespace\images\{image_selected}")
    image_embeds = st.session_state.vision_encoder(image)
    streamer = TextIteratorStreamer(st.session_state.text_model.tokenizer, skip_special_tokens=True)
    generation_kwargs = dict(image_embeds=image_embeds, question=moondream_prompt, streamer=streamer)
    thread = Thread(target=st.session_state.text_model.answer_question, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
    st.write(re.sub("<$", "", re.sub("END$", "", buffer)))
    with st.sidebar:
        st.write(f"elapsed time : {int(time.time()-ocr_time)} secs")