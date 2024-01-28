# ./codespace/pages/image reconition.py

import streamlit as st
from streamlit_extras.app_logo import add_logo

st.set_page_config(page_title="image reconition", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("image reconition")
add_logo("./llmonpy/pie.png", height=130)

import re
import time
from llmonpy import reconition

with st.sidebar:
    notepad = st.text_area(label='notepad', label_visibility='collapsed')
    image_selected = st.selectbox(label='pick image', options=reconition.scan_dir(directory='./images'))

if 'vision_encoder' not in st.session_state:
    reconition.popup_note(enable=st.session_state['enable_popups'])
    reconition.load_vision_encoder(enable_cpu=st.session_state['ocr_device'])

moondream_prompt = st.text_input(label="image reconition with :orange[moondream1]")
send_prompt = st.button("submit")
st.image(f"./images/{image_selected}")

if send_prompt:
    start_time = time.time()
    response = reconition.generate_response(image=image_selected, prompt=moondream_prompt)

    buffer = ""
    for word in response:
        buffer += word
    st.write(re.sub("<$", "", re.sub("END$", "", buffer)))

    with st.sidebar:
        st.write(f"elapsed time: {int(time.time()-start_time)} secs")
