import os
import streamlit as st

def scan_dir(directory):
    directory_list = []
    count = 0
    for file in os.scandir(f'{directory}'):
        if file.is_file():
            directory_list.append(file.name)
            count = count + 1
    return directory_list

st.title("🍋llmon-py")

if 'settings' not in st.session_state:

    llmon_config = st.form('setup')

    model_box_data = scan_dir('models')
    model_select = st.selectbox('model file', [model_box_data[0], model_box_data[1], model_box_data[2]])

    character_name = st.selectbox('character name', ['Dr. Rosenburg', 'Cortana', 'Kyle Katarn', 'Art Bell', 'Bot', 'Assistant', 'AI', 'Model'])
    voice_box_data = scan_dir('voices')
    voice_select = st.selectbox('voice file', [voice_box_data[0], voice_box_data[1], voice_box_data[2], voice_box_data[3], voice_box_data[4], voice_box_data[5], voice_box_data[6], voice_box_data[7], voice_box_data[8]])

    #template_select = st.selectbox('chat template', ['ajibawa_python', 'instruction', 'user_assist_art', 'user_assist_kyle', 'user_assist_hlsci', 'vicuna'])
    enable_code_voice = st.selectbox('coding mode', ['no', 'yes'])

    submit_config_button = llmon_config.form_submit_button('save')

    st.session_state.model_select = model_select
    st.session_state.char_name = character_name
    st.session_state.voice_select = voice_select
    #st.session_state.template_select = template_select
    st.session_state.enable_code_voice = enable_code_voice
    st.session_state.config_set = True