# ./codespace/settings.py

import streamlit as st
from streamlit_extras.app_logo import add_logo
import os
import torch

st.set_page_config(page_title="llmon-py", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title('llmon-py')
add_logo("./llmon_art/pie.png", height=130)

def scan_dir(directory):
    directory_list = []
    count = 0
    for file in os.scandir(f'{directory}'):
        if file.is_file():
            directory_list.append(file.name)
            count += count
    return directory_list

model_box_data = scan_dir('./models')
voice_box_data = scan_dir('./voices')
lora_list = scan_dir("./loras")

# pull from text files?
chat_templates = ['alpaca_noro', 'vicuna_based', 'none', 'deepseek', 'user_assist_rick', 'user_assist_duke', 'ajibawa_python', 'user_assist_art', 'user_assist_kyle', 'user_assist_hlsci']
model_names = ['Assistant', 'Nora', 'Dr. Rosenburg', 'Duke', 'Rick', 'Cortana', 'Kyle Katarn', 'Art Bell', 'Bot', 'AI', 'Model', 'Johnny 5', 'Codebot']
default_context_list = [128, 256, 512, 1024, 2048, 4096, 8096, 16384, 32768]
default_max_context_list = [2048, 4096, 8096, 16384, 32768]

model_box_dict = {}
voice_box_dict = {}
chat_template_dict = {}
model_name_dict = {}
lora_list_dict = {}
default_context_dict = {}
default_max_context_dict = {}

counter = 0
for i in model_box_data:
    model_box_dict[counter] = model_box_data[counter]
    counter += 1

counter = 0
for i in voice_box_data:
    voice_box_dict[counter] = voice_box_data[counter]
    counter += 1

counter = 0
for i in chat_templates:
    chat_template_dict[counter] = chat_templates[counter]
    counter += 1

counter = 0
for i in model_names:
    model_name_dict[counter] = model_names[counter]
    counter += 1

counter = 0
for i in lora_list:
    lora_list_dict[counter] = lora_list[counter]
    counter += 1

counter = 0
for i in default_context_list:
    default_context_dict[counter] = default_context_list[counter]
    counter += 1

counter = 0
for i in default_max_context_list:
    default_max_context_dict[counter] = default_max_context_list[counter]
    counter += 1

st.session_state.clear_vram = st.toggle('clear vram', value=False)

default_settings_state = {'enable_message_beep': True,
                          'enable_microphone': False,
                          'enable_voice': False,
                          'enable_code_voice': False,
                          'user_audio_length': 8,
                          'voice_select': voice_box_data[0],
                          'model_select': model_box_data[0],
                          'template_select': chat_templates[0],
                          'char_name': model_names[0],
                          'enable_popups': True,
                          'verbose_chat': True,
                          'max_context_prompt': 1024,
                          'max_context': 4096,
                          'torch_audio_cores': 8,
                          'gpu_layer_count': -1,
                          'cpu_core_count': 8,
                          'cpu_batch_count': 8,
                          'batch_size': 256,
                          'stream_chunk_size': 35,
                          'chunk_pre_buffer': 3,
                          'enable_sdxl_turbo': False,
                          'enable_cpu_only': False,
                          'enable_sdxl': False,
                          'lora_to_load': lora_list[0]}

for key, value in default_settings_state.items():
    if key not in st.session_state:
        st.session_state[key] = value
        
tab1, tab2, tab3, tab4 = st.tabs(["🔊audio", "💭chat model", "🔗advanced", '👀 sdxl'])
with tab1:
    st.header("audio")
    st.session_state['enable_message_beep'] = st.toggle('enable message boop', value=st.session_state['enable_message_beep'])        
    st.session_state['enable_microphone'] = st.toggle('enable microphone', value=st.session_state['enable_microphone'])
    st.session_state['enable_voice'] = st.toggle('enable tts model', value=st.session_state['enable_voice'])

    if st.session_state['enable_voice'] == True:
        tts_coding_button = False
    else:
        tts_coding_button = True

    st.session_state['enable_code_voice'] = st.toggle('enable tts coding mode', value=st.session_state['enable_code_voice'], disabled=tts_coding_button)
    st.session_state['user_audio_length'] = st.slider("microphone rec time(sec)", 2, 25, st.session_state['user_audio_length'])

    set_voice_index = 0
    for key, value in voice_box_dict.items():
        if value == st.session_state['voice_select']:
            set_voice_index = key

    st.session_state['voice_select'] = st.selectbox('voice file', voice_box_data, index=set_voice_index)
    st.audio(f"./voices/{st.session_state['voice_select']}")

with tab2:
    st.header("chat model")

    set_model_index = 0
    for key, value in model_box_dict.items():
        if value == st.session_state['model_select']:
            set_model_index = key

    st.session_state['model_select'] = st.selectbox('model file', model_box_data, index=set_model_index)

    set_template_index = 0
    for key, value in chat_template_dict.items():
        if value == st.session_state['template_select']:
            set_template_index = key

    st.session_state['template_select'] = st.selectbox('chat template', chat_templates, index=set_template_index)

    set_char_name_index = 0
    for key, value in model_name_dict.items():
        if value == st.session_state['char_name']:
            set_char_name_index = key

    st.session_state['char_name'] = st.selectbox('model character name', model_names, index=set_char_name_index)

with tab3:
    st.header("advanced")
    #st.session_state.enable_summarizer = st.toggle('summarize text', value=False)
    st.session_state['enable_popups'] = st.toggle('enable system popups', value=st.session_state['enable_popups'])
    st.session_state['verbose_chat'] = st.toggle('enable verbose console', value=st.session_state['verbose_chat'])

    set_context_index = 0
    for key, value in default_context_dict.items():
        if value == st.session_state['max_context_prompt']:
            set_context_index = key

    st.session_state['max_context_prompt'] = st.selectbox('max token gen', default_context_list, index=set_context_index)

    set_max_context_index = 0
    for key, value in default_max_context_dict.items():
        if value == st.session_state['max_context']:
            set_max_context_index = key

    st.session_state['max_context']= st.selectbox('max context size', default_max_context_list, index=set_max_context_index)
    st.session_state['torch_audio_cores'] = st.slider('torch audio cores', 2, 32, st.session_state['torch_audio_cores'])
    st.session_state['gpu_layer_count'] = st.slider('gpu layers', -1, 128, st.session_state['gpu_layer_count'])
    st.session_state['cpu_core_count'] = st.slider('cpu cores', 1, 128, st.session_state['cpu_core_count'])
    st.session_state['cpu_batch_count'] = st.slider('cpu batch cores', 1, 128,  st.session_state['cpu_batch_count'])
    st.session_state['batch_size'] = st.slider('batch size', 0, 1024, st.session_state['batch_size'])
    st.session_state['stream_chunk_size'] = st.slider('stream chunk size', 20, 200, st.session_state['stream_chunk_size'])
    st.session_state['chunk_pre_buffer'] = st.slider('chunk buffers', 0, 4, st.session_state['chunk_pre_buffer'])
    
with tab4:
    st.header("sdxl turbo")
    st.session_state['enable_sdxl_turbo'] = st.toggle('enable sdxl turbo', value=st.session_state['enable_sdxl_turbo'])
    st.session_state['enable_cpu_only'] = st.toggle('run turbo on cpu', value=st.session_state['enable_cpu_only'])
    st.header("sdxl 1.0")
    st.session_state['enable_sdxl'] = st.toggle('enable sdxl 1.0', value=st.session_state['enable_sdxl'])

    set_lora_index = 0
    for key, value in lora_list_dict.items():
        if value == st.session_state['lora_to_load']:
            set_lora_index = key

    st.session_state['lora_to_load'] = st.session_state.lora_to_load = st.selectbox('lora to load', lora_list, index=set_lora_index)

st.write('session state details')
st.json(st.session_state, expanded=False)

if st.session_state.clear_vram:
    del st.session_state.model
    del st.session_state.config
    del st.session_state['chat_model']
    torch.cuda.empty_cache()
    st.session_state['chat_model'] = None
    st.session_state.model = None
