# ./codespace/pages/music generation.py
import streamlit as st
from llmonpy import llmonaid

st.set_page_config(page_title="music generation", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("🍋music generation")

from transformers import pipeline
import scipy

if st.session_state['approved_login'] and st.session_state['enable_music']:
    st.markdown(":orange[music generation with] :blue[facebook-musicgen medium]")
    with st.sidebar:
        llmonaid.memory_display()

    if 'musicgen_synthesiser' not in st.session_state:
        llmonaid.popup_note(message='👊 bringing back the boom bap...')
        st.session_state['musicgen_synthesiser'] = pipeline("text-to-audio", "meta-musicgen", device='cuda:0')
      
    user_input = st.chat_input("ex: lo-fi music with a soothing melody")
    if user_input:
        llmonaid.popup_note(message='👊 generating a tune...')
        music = st.session_state['musicgen_synthesiser'](user_input, forward_params={"do_sample": True})
        scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
        st.audio("musicgen_out.wav")

if st.session_state['approved_login'] == False:
    st.image('./llmonpy/pie.png', caption='please login to continue')
if st.session_state['enable_music'] == False:
    st.image('./llmonpy/pie.png', caption="enable musicgen to continue")