# ./codespace/home.py
import streamlit as st
from streamlit_extras.app_logo import add_logo

st.set_page_config(page_title="llmon-py", page_icon="🍋", layout="centered", initial_sidebar_state="auto")
st.title('🍋llmon-py')
add_logo("./llmonpy/pie.png", height=130)

from llmonpy import llmonaid

model_box_data = llmonaid.scan_dir('./models')
voice_box_data = llmonaid.scan_dir('./voices')
lora_list = llmonaid.scan_dir("./loras")
chat_templates = ['chat_default', 'code_deepseek', 'chat_mixtral', 'chat_redguard', 'chat_artbell', 'chat_halflife']

if 'approved_login' not in st.session_state:
    st.session_state['approved_login'] = False

if st.session_state['approved_login'] == False:
    llmonaid.attempt_login(model_box_data, voice_box_data, lora_list, chat_templates)

if st.session_state['approved_login']:
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

    clear_vram = st.button(label='clear models', help='clear all vram and system memory of all models and clears current context/messages')
    llmonaid.memory_display()
    tab2, tab1, tab4, tab3 = st.tabs(["💭chat model", "🔊audio", '👀image gen/vison', "🔗advanced"])
    with tab1:
        st.session_state['enable_microphone'] = st.toggle('enable microphone', value=st.session_state['enable_microphone'], disabled=llmonaid.check_user_type())
        st.session_state['enable_voice'] = st.toggle('enable tts model', value=st.session_state['enable_voice'], disabled=llmonaid.check_user_type())
        if st.session_state['enable_voice'] == True:
            tts_coding_button = False
            disable_cpu_button = False
        else:
            tts_coding_button = True
            disable_cpu_button = True

        st.session_state['enable_code_voice'] = st.toggle('enable tts coding mode', value=st.session_state['enable_code_voice'], disabled=tts_coding_button)
        st.session_state['user_audio_length'] = st.slider("microphone rec time(sec)", 2, 25, st.session_state['user_audio_length'], disabled=llmonaid.check_user_type())

        set_voice_index = 0
        for key, value in voice_box_dict.items():
            if value == st.session_state['voice_select']:
                set_voice_index = key
        st.session_state['voice_select'] = st.selectbox('voice file', voice_box_data, index=set_voice_index, disabled=llmonaid.check_user_type())
        if st.session_state['user_type'] == 'admin':
            st.audio(f"./voices/{st.session_state['voice_select']}")

    with tab2:
        set_model_index = 0
        for key, value in model_box_dict.items():
            if value == st.session_state['model_select']:
                set_model_index = key
        st.session_state['model_select'] = st.selectbox('model file', model_box_data, index=set_model_index)
        if st.session_state['model_select'] == "mixtral-8x7b-instruct-v0.1.Q5_0.gguf":
            st.session_state['gpu_layer_count'] = 14

        if st.session_state['model_select'] == "deepseek-coder-33b-instruct.Q5_K_M.gguf":
            st.session_state['gpu_layer_count'] = 35    

        set_template_index = 0
        for key, value in chat_template_dict.items():
            if value == st.session_state['template_select']:
                set_template_index = key
        st.session_state['template_select'] = st.selectbox('chat template', chat_templates, index=set_template_index)
        st.session_state['char_name'] = st.text_input(label='model name', value=st.session_state['char_name'])

    with tab3:
        set_max_context_index = 0
        for key, value in default_max_context_dict.items():
            if value == st.session_state['max_context']:
                set_max_context_index = key
        st.session_state['max_context']= st.selectbox('max context size', default_max_context_list, index=set_max_context_index)
        st.session_state['gpu_layer_count'] = st.text_input(label='gpu layers', value=st.session_state['gpu_layer_count'])
        st.session_state['stream_chunk_size'] = st.slider('tts stream chunk size', 20, 200, st.session_state['stream_chunk_size'], disabled=llmonaid.check_user_type())
        st.session_state['chunk_pre_buffer'] = st.slider('tts chunk buffers', 0, 6, st.session_state['chunk_pre_buffer'], disabled=llmonaid.check_user_type())
    
    with tab4:
        if st.session_state['enable_sdxl']:
            st.session_state['enable_sdxl_turbo'] = st.toggle('enable :orange[sdxl] :red[turbo]', value=st.session_state['enable_sdxl_turbo'], disabled=True)
            st.session_state['img2img_on'] = st.toggle('enable :orange[sdxl] :red[turbo] :rainbow[img2img]', value=st.session_state['img2img_on'], disabled=True)
        else:
            st.session_state['enable_sdxl_turbo'] = st.toggle('enable :orange[sdxl] :red[turbo]', value=st.session_state['enable_sdxl_turbo'], disabled=False)
            st.session_state['img2img_on'] = st.toggle('enable :orange[sdxl] :red[turbo] :rainbow[img2img]', value=st.session_state['img2img_on'], disabled=False)

        if st.session_state['enable_sdxl_turbo']:
            st.session_state['enable_sdxl'] = st.toggle('enable :orange[sdxl 1.0]', value=st.session_state['enable_sdxl'], disabled=True)
        else:
            st.session_state['enable_sdxl'] = st.toggle('enable :orange[sdxl 1.0]', value=st.session_state['enable_sdxl'], disabled=False)

        set_lora_index = 0
        for key, value in lora_dict.items():
            if value == st.session_state['lora_selected']:
                set_lora_index = key
        st.session_state['lora_selected'] = st.selectbox('lora for :orange[sdxl]', lora_list, index=set_lora_index)

        st.session_state['enable_ocr'] = st.toggle('enable :violet[moon]:orange[dream]', value=st.session_state['enable_ocr'])
        st.session_state['ocr_device'] = st.toggle('run on cpu', value=st.session_state['ocr_device'], key='ocr_cpu_button')

    if clear_vram:
        llmonaid.clear_vram()
        st.rerun()