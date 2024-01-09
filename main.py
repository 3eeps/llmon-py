# ./codespace/main.py

import streamlit as st
from streamlit_extras.app_logo import add_logo
import GPUtil as GPU
import os
import torch
import keyboard
import random

keyboard.unhook_all()
st.set_page_config(page_title="llmon-py", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
llmon_list = ['lemon (1).png', 'lemon (3).png', 'lemon (4).png', 'lemon (5).png', 'lemon (6).png', 'lemon (7).png', 'lemon (8).png', 'lemon (10).png', 'lemon (10).png', 'lemon (11).png', 'lemon (13).png', 'lemon (14).png', 'lemon (15).png', 'lemon (16).png', 'lemon (18).png']
st.title('llmon-py')
add_logo(f'./llmon_art/{llmon_list[random.randint(0,14)]}')
st.divider()

def scan_dir(directory):
    directory_list = []
    count = 0
    for file in os.scandir(f'{directory}'):
        if file.is_file():
            directory_list.append(file.name)
            count += count
    return directory_list

GPUs = GPU.getGPUs()
gpu = GPUs[0]

model_box_data = scan_dir('./models')
voice_box_data = scan_dir('./voices')
boop_box_data = scan_dir('./ui-tones')
lora_list = scan_dir("./loras")

chat_templates = ['vicuna_based', 'deepseek', 'user_assist_rick', 'user_assist_duke', 'ajibawa_python', 'user_assist_art', 'user_assist_kyle', 'user_assist_hlsci', 'custom']
model_names = ['Assistant', 'Dr. Rosenburg', 'Duke', 'Rick', 'Cortana', 'Kyle Katarn', 'Art Bell', 'Bot', 'AI', 'Model', 'Johnny 5', 'Codebot']

st.progress(float("{0:.0f}".format(gpu.memoryFree)) / float("{0:.0f}".format(gpu.memoryTotal)), "vram {0:.0f}/{1:.0f}mb".format(gpu.memoryUsed, gpu.memoryTotal))
st.session_state.clear_vram = st.toggle('clear vram', value=False)
tab1, tab2, tab3, tab4 = st.tabs(["🔊audio", "💭chat model", "🔗advanced", '👀 sdxl'])

with tab1:
    st.header("audio")
    st.session_state.boop_select = st.selectbox('message beep', boop_box_data)
    st.audio(f"./ui-tones/{st.session_state.boop_select}")          
    st.session_state.enable_microphone = st.toggle('enable microphone', value=False)
    st.session_state.microphone_hotkey = st.selectbox('microphone hotkey', ['ctrl'])
    st.session_state.enable_voice = st.toggle('enable tts model', value=False)
    st.session_state.enable_code_voice = st.toggle('enable coding mode', value=False)
    st.session_state.user_audio_length = st.slider("microphone rec time(sec)", 2, 25, 8)
    st.session_state.audio_cuda_or_cpu = st.selectbox('audio inference to', ["cuda", "cpu"])
    st.session_state.voice_select = st.selectbox('voice file', voice_box_data)
    st.audio(f"./voices/{st.session_state.voice_select}")

with tab2:
    st.header("chat model")
    st.session_state.model_select = st.selectbox('model file', model_box_data)
    st.session_state.template_select = st.selectbox('chat template', chat_templates)
    st.session_state.char_name = st.selectbox('model name', model_names)
    st.session_state.model_language = st.selectbox('tts language', ["en", 'es'])

with tab3:
    st.header("advanced")
    #st.session_state.enable_summarizer = st.toggle('summarize text', value=False)
    st.session_state.enable_popups = st.toggle('enable system popups', value=True)
    st.session_state.console_warnings = st.selectbox('hide console warnings', ['ignore', 'default'])
    st.session_state.verbose_chat = st.toggle('enable verbose console', value=True)
    st.session_state.max_context_prompt = int(st.selectbox('max token gen', ['1024', '256', '512', '1536', '2048', '4096', '8096', '16384', '32768']))
    st.session_state.max_context = int(st.selectbox('max context size', ['4096', '8096', '16384', '32768']))
    st.session_state.torch_audio_cores = st.slider('torch audio cores', 2, 64, 8)
    st.session_state.gpu_layer_count = st.slider('gpu layers', -1, 128, -1)
    st.session_state.cpu_core_count = st.slider('cpu cores', 1, 128, 8)
    st.session_state.cpu_batch_count = st.slider('cpu batch cores', 1, 128, 8)
    st.session_state.batch_size= st.slider('batch size', 0, 1024, 256)
    st.session_state.stream_chunk_size = st.slider('stream chunk size', 20, 200, 40)
    st.session_state.chunk_pre_buffer = st.slider('chunk buffers', 0, 2, 2)
    
with tab4:
    st.header("sdxl turbo")
    st.session_state.enable_sdxl_turbo = st.toggle('enable sdxl turbo', value=False)
    st.session_state.enable_cpu_only = st.toggle('run turbo on cpu', value=False)
    st.header("sdxl 1.0")
    st.session_state.enable_sdxl = st.toggle('enable sdxl 1.0', value=False)
    st.session_state.lora_to_load = st.session_state.lora_to_load = st.selectbox('lora to load', lora_list)

st.write('session state details')
st.json(st.session_state, expanded=False)

if st.session_state.clear_vram:
    del st.session_state.chat_model
    torch.cuda.empty_cache()
