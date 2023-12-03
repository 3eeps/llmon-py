# ./codespace/main.py

import streamlit as st
from streamlit_extras.app_logo import add_logo

import os
import keyboard

keyboard.unhook_all()
st.set_page_config(page_title="llmon-py", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("llmon-py")
add_logo("logo.png", height=150)
st.divider()

def scan_dir(directory):
    directory_list = []
    count = 0
    for file in os.scandir(f'{directory}'):
        if file.is_file():
            directory_list.append(file.name)
            count += count
    return directory_list

model_box_data = scan_dir('models')
voice_box_data = scan_dir('voices')
tab1, tab2, tab3, tab4 = st.tabs(["🔊audio", "💭chat model", "🔗advanced", '👀 sdxl turbo'])
    
with tab1:
    st.header("audio")          
    st.session_state.enable_microphone = st.toggle('enable microphone', value=False)
    st.session_state.enable_voice = st.toggle('enable tts model', value=False)
    st.session_state.enable_code_voice = st.toggle('enable coding mode', value=False)
    st.session_state.user_audio_length = st.slider("microphone rec time(sec)", 2, 25, 8)
    st.session_state.audio_cuda_or_cpu = st.selectbox('audio inference to', ["cuda", "cpu"])
    st.session_state.voice_select = st.selectbox('voice file', [voice_box_data[0], voice_box_data[1], voice_box_data[2], voice_box_data[3], voice_box_data[4], voice_box_data[5], voice_box_data[6]])

with tab2:
    st.header("chat model")
    st.session_state.model_select = st.selectbox('model file', [model_box_data[0], model_box_data[1], model_box_data[2], model_box_data[3]])   
    st.session_state.template_select = st.selectbox('chat template', ['user_assist_rick', 'vicuna_based', 'user_assist_duke', 'ajibawa_python', 'user_assist_art', 'user_assist_kyle', 'user_assist_hlsci'])
    st.session_state.char_name = st.selectbox('model name', ['Assistant', 'Dr. Rosenburg', 'Duke', 'Rick', 'Cortana', 'Kyle Katarn', 'Art Bell', 'Bot', 'AI', 'Model'])
    st.session_state.model_language = st.selectbox('tts language', ["en", 'es'])
    st.session_state.text_stream_speed = st.slider('streaming text speed', 0, 2, 1)

with tab3:
    st.header("advanced")
    st.session_state.enable_popups = st.toggle('enable system popups', value=True)
    st.session_state.max_context_prompt = int(st.selectbox('max token gen', ['1536', '256', '512', '1024', '2048', '4096', '8096', '16384', '32768']))
    st.session_state.max_context = int(st.selectbox('max context size', ['4096', '8096', '16384', '32768']))
    st.session_state.torch_audio_cores = st.slider('torch audio cores', 2, 64, 8)
    st.session_state.gpu_layer_count = st.slider('gpu layers', -1, 128, -1)
    st.session_state.cpu_core_count = st.slider('cpu cores', 1, 128, 12)
    st.session_state.cpu_batch_count = st.slider('cpu batch cores', 1, 128, 12)
    st.session_state.batch_size= st.slider('batch size', 0, 1024, 128)
    st.session_state.stream_chunk_size = st.slider('stream chunk size', 20, 200, 40)
    st.session_state.chunk_buffer = st.slider('chunk buffers', 0, 2, 1)
    st.session_state.console_warnings = st.selectbox('hide console warnings', ['ignore', 'default'])
    st.session_state.verbose_chat = st.toggle('enable verbose console', value=False)

with tab4:
    st.header("sdxl turbo")
    st.session_state.enable_sdxl = st.toggle('enable sdxl turbo', value=False)
    st.session_state.enable_cpu_only = st.toggle('run on cpu', value=False)
