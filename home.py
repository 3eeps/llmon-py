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
    st.header('settings')
    model_box_data = scan_dir('models')
    model_select = st.selectbox('model file', [model_box_data[0], model_box_data[1], model_box_data[2]])

    character_name = st.selectbox('character name', ['Dr. Rosenburg', 'Cortana', 'Kyle Katarn', 'Art Bell', 'Bot', 'Assistant', 'AI', 'Model'])
    voice_box_data = scan_dir('voices')
    voice_select = st.selectbox('voice file', [voice_box_data[0], voice_box_data[1], voice_box_data[2], voice_box_data[3], voice_box_data[4], voice_box_data[5], voice_box_data[6], voice_box_data[7], voice_box_data[8]])

    model_avatar_img = '🤖'
    user_avatar_img = '🙅'
                                                   
    template_select = st.selectbox('chat template', ['ajibawa_python', 'user_assist_art', 'user_assist_kyle', 'user_assist_hlsci'])
    enable_code_voice = st.selectbox('coding mode', ['no', 'yes'])
    user_avatar_img = st.selectbox('user avatar', ['🙅', '🤖', '😈', '🗿', '💩', '💀', '👾', '👽', '👤', '🎅'])
    model_avatar_img = st.selectbox('model avatar', ['🤖', '🙅', '😈', '🗿', '💩', '💀', '👾', '👽', '👤', '🎅'])
    st.header('advanced')
    verbose_chat = st.selectbox('verbose mode', ['no', 'yes'])
    max_prompt_context = st.selectbox('max token gen', ['default', '256', '512', '1024', '1536', '2048', '4096', '8096', '16384', '32768'])
    max_context = st.selectbox('max context size', ['default', '4096', '8096', '16384', '32768'])
    gpu_layer_count = st.slider('gpu layers', -1, 128, -1)
    cpu_core_count = st.slider('cpu cores', 1, 128, 12)
    cpu_batch_count = st.slider('cpu batch cores', 1, 128, 12)
    stream_chunk_size = st.slider('stream chunk size', 20, 200, 40)
    chunk_buffer = st.slider('chunk buffers', 0, 2, 1)
    
    submit_config_button = llmon_config.form_submit_button('save')
     
    if verbose_chat == 'yes':
        chat_verbose = True
    else:
        chat_verbose = False

    if max_prompt_context == 'default':
        max_prompt_context = 1536

    if max_context  == 'default':
        max_context  = 4096

    st.session_state.verbose_chat = chat_verbose 
    st.session_state.model_select = model_select
    st.session_state.char_name = character_name
    st.session_state.model_avatar = user_avatar_img
    st.session_state.user_avatar = model_avatar_img
    st.session_state.voice_select = voice_select
    st.session_state.context_max_prompt = int(max_prompt_context)
    st.session_state.max_context = int(max_context)
    st.session_state.gpu_layer_count = gpu_layer_count
    st.session_state.template_select = template_select
    st.session_state.enable_code_voice = enable_code_voice
    st.session_state.chunk_buffer = chunk_buffer
    st.session_state.stream_chunk_size = stream_chunk_size
    st.session_state.config_set = True
    st.session_state.cpu_core_count = cpu_core_count
    st.session_state.cpu_batch_count = cpu_batch_count
