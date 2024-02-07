# ./codespace/pages/image reconition.py
import streamlit as st
from streamlit_extras.app_logo import add_logo

st.set_page_config(page_title="image reconition", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("🍋image reconition")
add_logo("./llmonpy/pie.png", height=130)

if st.session_state['approved_login'] == True and st.session_state['enable_ocr'] == True:

    import re
    from llmonpy import reconition, llmonaid

    st.markdown("image reconition with :orange[moondream1]")
    with st.sidebar:
        st.session_state['notepad'] = st.text_area(label='notepad', label_visibility='collapsed', value=st.session_state['notepad'])
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
        st.session_state.buffer = ""
        for word in response:
            st.session_state.buffer += word
        st.rerun()
if st.session_state['approved_login'] == False:
    st.image('./llmonpy/pie.png', caption='please login to continue')
if st.session_state['enable_ocr'] == False:
    st.image('./llmonpy/pie.png', caption="enable a vision model to continue")
st.session_state.buffer = ""