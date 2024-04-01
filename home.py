# ./codespace/home.py
import streamlit as st
from llmonpy import llmonaid
st.set_page_config(page_title="llmon-py", page_icon="🍋", layout="centered", initial_sidebar_state="auto")
st.title('🍋llmon-py')

model_box_data = llmonaid.scan_dir('./models')
voice_box_data = llmonaid.scan_dir('./voices')
lora_list = llmonaid.scan_dir("./loras")
chat_templates = ['code_deepseek', 'chat_mistral', 'code_mistral', 'chat_mixtral_base', 'chat_redguard', 'chat_artbell', 'chat_halflife']
llmonaid.init_state(model_box_data, voice_box_data, lora_list, chat_templates)

import json
with open("functions.json", "r") as file:
    st.session_state.functions = json.load(file)

default_max_context_list = [2048, 4096, 6144, 8192, 10240, 12288]
model_box_dict = {}
voice_box_dict = {}
lora_dict = {}
chat_template_dict = {}
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
for i in lora_list:
    lora_dict[counter] = lora_list[counter]
    counter += 1

counter = 0
for i in chat_templates:
    chat_template_dict[counter] = chat_templates[counter]
    counter += 1

counter = 0
for i in default_max_context_list:
    default_max_context_dict[counter] = default_max_context_list[counter]
    counter += 1

with st.sidebar:
    llmonaid.memory_display()
    clear_vram = st.button(label=':green[clear vram]')
    inference = st.switch_page('')
tab2, tab1, tab3 = st.tabs(["💭chat", "🔊audio", '👀image'])
with tab1:
    st.session_state['enable_microphone'] = st.toggle(':orange[enable microphone]', value=st.session_state['enable_microphone'])
    st.session_state['enable_voice'] = st.toggle(':orange[enable] :green[xttsv2 tts]', value=st.session_state['enable_voice'])
    st.session_state['enable_voice_melo'] = st.toggle(':orange[enable] :blue[melotts]', value=st.session_state['enable_voice_melo'])
    st.session_state['user_audio_length'] = st.text_input(":orange[microphone rec time(sec)]", value=st.session_state['user_audio_length'])

    if st.session_state['enable_voice']:
        set_voice_index = 0
        for key, value in voice_box_dict.items():
            if value == st.session_state['voice_select']:
                set_voice_index = key
        st.session_state['voice_select'] = st.selectbox(':orange[voice file]', voice_box_data, index=set_voice_index)
        st.audio(f"./voices/{st.session_state['voice_select']}")
        st.session_state['stream_chunk_size'] = st.text_input(':orange[tts stream chunk size]', value=int(st.session_state['stream_chunk_size']))
        st.session_state['chunk_pre_buffer'] = st.text_input(':orange[tts chunk buffers]', value=int(st.session_state['chunk_pre_buffer']))

with tab2:
    loader_list = ['llama-cpp-python', 'exllamav2']
    loader_index = 0
    if st.session_state['loader_type'] == 'llama-cpp-python':
        loader_index = 0
    else:
        loader_index = 1
    st.session_state['loader_type'] = st.selectbox(':orange[backend]', loader_list, index=loader_index)

    set_model_index = 0
    for key, value in model_box_dict.items():
        if value == st.session_state['model_select']:
            set_model_index = key
    st.session_state['model_select'] = st.selectbox(':orange[model file]', model_box_data, index=set_model_index)
    st.session_state['gpu_layer_count'] = -1
    if st.session_state['model_select'] == "mixtral-8x7b-instruct-v0.1.Q5_0.gguf":
        st.session_state['gpu_layer_count'] = 15

    if st.session_state['model_select'] == "deepseek-coder-33b-instruct.Q5_K_M.gguf":
        st.session_state['gpu_layer_count'] = 40         

    #llmonaid.get_gguf_info(file_path=f"./models/{st.session_state['model_select']}")

    set_template_index = 0
    for key, value in chat_template_dict.items():
        if value == st.session_state['template_select']:
            set_template_index = key
    st.session_state['template_select'] = st.selectbox(':orange[chat template]', chat_templates, index=set_template_index)

    set_max_context_index = 0
    for key, value in default_max_context_dict.items():
        if value == st.session_state['max_context']:
            set_max_context_index = key
    st.session_state['max_context']= st.selectbox(':orange[max context size]', default_max_context_list, index=set_max_context_index)
    st.session_state['gpu_layer_count'] = st.text_input(label=':orange[gpu layers]', value=st.session_state['gpu_layer_count'])

with tab3:
    st.session_state['enable_sdxl_turbo'] = st.toggle(':orange[enable sdxl] :red[turbo]', value=st.session_state['enable_sdxl_turbo'], disabled=True)
    st.session_state['img2img_on'] = st.toggle(':orange[enable sdxl] :red[turbo] :rainbow[img2img]', value=st.session_state['img2img_on'], disabled=True)
    st.session_state['enable_sdxl'] = st.toggle(':orange[enable] :violet[sdxl 1.0]', value=st.session_state['enable_sdxl'], disabled=True)
    if st.session_state['enable_sdxl']:
        st.session_state['use_lora'] = st.toggle(':orange[enable lora]', value=st.session_state['use_lora'], disabled=False)

        set_lora_index = 0
        for key, value in lora_dict.items():
            if value == st.session_state['lora_selected']:
                set_lora_index = key
        st.session_state['lora_selected'] = st.selectbox(':orange[lora for sdxl]', lora_list, index=set_lora_index)

if clear_vram:
    llmonaid.clear_vram()
    st.rerun()