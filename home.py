import streamlit as st
import os

st.set_page_config(
    page_title="llmon-py",
    page_icon="🍋",
    layout="wide",
    initial_sidebar_state="expanded")
st.title("🍋llmon-py")

def scan_dir(directory):
    directory_list = []
    count = 0
    for file in os.scandir(f'{directory}'):
        if file.is_file():
            directory_list.append(file.name)
            count = count + 1
    return directory_list

model_box_data = scan_dir('models')
voice_box_data = scan_dir('voices')

if 'settings' not in st.session_state:
    llmon_config = st.form('setup')
    tab1, tab2, tab3 = st.tabs(["🔊audio", "💭chat model", "🔗advanced"])
    
    with tab1:
        st.header("audio")
        st.session_state.voice_select = st.selectbox('voice file', [voice_box_data[0], voice_box_data[1], voice_box_data[2], voice_box_data[3], voice_box_data[4], voice_box_data[5], voice_box_data[6]])          
        st.session_state.enable_code_voice = st.selectbox('coding mode', ['no', 'yes'])
        st.session_state.enable_voice = st.selectbox('enable audio', ['yes', 'no'])
        st.session_state.voice_word = st.selectbox('start mic word', ['default', 'talk', 'vc', 'stt'])
        st.session_state.user_audio_length = st.slider("mic rec time(sec)", 2, 25, 8)
        st.session_state.audio_cuda_or_cpu = st.selectbox('audio inference to', ["cuda", "cpu"])

    with tab2:
        st.header("chat model")
        st.session_state.model_select = st.selectbox('model file', [model_box_data[0], model_box_data[2]])   
        st.session_state.template_select = st.selectbox('chat template', ['ajibawa_python', 'user_assist_art', 'user_assist_kyle', 'user_assist_hlsci'])
        st.session_state.char_name = st.selectbox('model name', ['Dr. Rosenburg', 'Cortana', 'Kyle Katarn', 'Art Bell', 'Bot', 'Assistant', 'AI', 'Model'])
        st.session_state.user_avatar = st.selectbox('user avatar', ['🙅', '🤖', '😈', '🗿', '💩', '💀', '👾', '👽', '👤', '🎅'])
        st.session_state.model_avatar = st.selectbox('model avatar', ['🤖', '🙅', '😈', '🗿', '💩', '🗿', '👾', '👽', '👤', '🎅'])
        st.session_state.model_language = st.selectbox('tts language', ["en"])
        st.session_state.text_stream_speed = st.slider('text streaming speed', 0, 2, 1)

    with tab3:
        st.header("advanced")
        st.session_state.enable_popups = st.selectbox('enable system messages', ['yes', 'no'])
        max_prompt_context = st.selectbox('max token gen', ['default', '256', '512', '1024', '1536', '2048', '4096', '8096', '16384', '32768'])
        max_context = st.selectbox('max context size', ['default', '4096', '8096', '16384', '32768'])
        st.session_state.torch_audio_cores = st.slider('torch audio cores', 2, 64, 8)
        st.session_state.gpu_layer_count = st.slider('gpu layers', -1, 128, -1)
        st.session_state.cpu_core_count = st.slider('cpu cores', 1, 128, 12)
        st.session_state.cpu_batch_count = st.slider('cpu batch cores', 1, 128, 12)
        st.session_state.batch_count = st.slider('cpu batch cores', 0, 1024, 128)
        st.session_state.stream_chunk_size = st.slider('stream chunk size', 20, 200, 40)
        st.session_state.chunk_buffer = st.slider('chunk buffers', 0, 2, 1)
        st.session_state.console_warnings = st.selectbox('hide console warnings', ['ignore', 'default'])
        st.session_state.verbose_chat = st.selectbox('verbose mode', ['no', 'yes'])
        
    submit_config_button = llmon_config.form_submit_button('save setup')
     
    if st.session_state.verbose_chat == 'yes':
        st.session_state.verbose_chat = True
    else:
        st.session_state.verbose_chat = False

    # defaults
    if max_prompt_context == 'default':
        max_prompt_context = 1536

    if max_context  == 'default':
        max_context  = 4096

    if st.session_state.voice_word  == 'default':
        st.session_state.voice_word = 'voice'
        
    st.session_state.context_max_prompt = int(max_prompt_context)
    st.session_state.max_context = int(max_context)
    st.session_state.config_set = True
