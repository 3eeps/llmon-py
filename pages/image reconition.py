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

    uploaded_file = st.file_uploader(label="Choose a image file")
    if uploaded_file is not None:
        st.session_state.bytes_data = uploaded_file.getvalue()
    else:
        st.session_state.buffer = ""

if 'vision_encoder' not in st.session_state:
    reconition.popup_note(enable=st.session_state['enable_popups'])
    reconition.load_vision_encoder(enable_cpu=st.session_state['ocr_device'])

moondream_prompt = st.text_input(label="image reconition with :orange[moondream1]", value='can you describe this image?')
send_prompt = st.button("submit")

if uploaded_file is not None:
    st.image(image=st.session_state.bytes_data, caption=re.sub("<$", "", re.sub("END$", "", st.session_state.buffer)))
else:
    st.image(image=f"./llmonpy/pie.png", caption=re.sub("<$", "", re.sub("END$", "", st.session_state.buffer)))

if send_prompt:
    start_time = time.time()
    response = reconition.generate_response(image_data=st.session_state.bytes_data, prompt=moondream_prompt)

    st.session_state.buffer = ""
    for word in response:
        st.session_state.buffer += word

    with st.sidebar:
        st.write(f"elapsed time: {int(time.time()-start_time)} secs")
    st.rerun()
