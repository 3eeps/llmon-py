# ./codespace/pages/image reconition.py
import streamlit as st
from llmonpy import reconition, llmonaid

st.set_page_config(page_title="image reconition", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("🍋image reconition")

if st.session_state['approved_login'] and st.session_state['enable_vision']:
    st.markdown("image reconition with :violet[moon]:orange[dream2]")
    with st.sidebar:
        llmonaid.memory_display()
        uploaded_file = st.file_uploader(label="Choose a image file")
        if uploaded_file is not None:
            st.session_state.bytes_data = uploaded_file.getvalue()

    if 'moondream' not in st.session_state:
        llmonaid.popup_note(message='🌒 describe me to the moon...')
        reconition.load_vision_encoder()

    if uploaded_file is not None:
        llmonaid.message_boop()
        st.image(image=st.session_state.bytes_data)
        st.write_stream(llmonaid.stream_text(text=st.session_state.buffer))
      
    user_input = st.chat_input("ask questions about your images")
    if user_input:
        response = reconition.generate_response(image_data=st.session_state.bytes_data, prompt=user_input)
        st.session_state.buffer = ""
        for word in response:
            st.session_state.buffer += word
        st.rerun()
if st.session_state['approved_login'] == False:
    st.image('./llmonpy/pie.png', caption='please login to continue')
if st.session_state['enable_vision'] == False:
    st.image('./llmonpy/pie.png', caption="enable moondream to continue")
st.session_state.buffer = ""