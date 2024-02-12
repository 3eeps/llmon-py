# ./codespace/pages/image reconition.py
import streamlit as st
import re
from llmonpy import reconition, llmonaid

st.set_page_config(page_title="image reconition", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("🍋image reconition")

if st.session_state['approved_login'] and st.session_state['enable_ocr']:
    st.markdown("image reconition with :orange[moondream1]")
    with st.sidebar:
        uploaded_file = st.file_uploader(label="Choose a image file")
        if uploaded_file is not None:
            st.session_state.bytes_data = uploaded_file.getvalue()

    if 'vision_encoder' not in st.session_state:
        llmonaid.popup_note(message='🌒 looking up at you, moondream...')
        reconition.load_vision_encoder(enable_cpu=st.session_state['ocr_device'])
        
    llmonaid.memory_display()

    if uploaded_file is not None:
        st.image(image=st.session_state.bytes_data)
        st.write_stream(llmonaid.stream_text(text=re.sub("<$", "", re.sub("END$", "", st.session_state.buffer))))
      
    user_input = st.chat_input("ask questions about your images")
    if user_input:
        response = reconition.generate_response(image_data=st.session_state.bytes_data, prompt=user_input)
        llmonaid.message_boop()
        st.session_state.buffer = ""
        for word in response:
            st.session_state.buffer += word
        st.rerun()
if st.session_state['approved_login'] == False:
    st.image('./llmonpy/pie.png', caption='please login to continue')
if st.session_state['enable_ocr'] == False:
    st.image('./llmonpy/pie.png', caption="enable a vision model to continue")
st.session_state.buffer = ""