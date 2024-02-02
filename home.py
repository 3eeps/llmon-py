# ./codespace/home.py
import streamlit as st
from streamlit_extras.app_logo import add_logo

st.set_page_config(page_title="llmon-py", page_icon="🍋", layout="centered", initial_sidebar_state="collapsed")
st.title('🍋llmon-py')
add_logo("./llmonpy/pie.png", height=130)

from llmonpy import llmonaid

model_box_data = llmonaid.scan_dir('./models')
voice_box_data = llmonaid.scan_dir('./voices')
chat_templates = ['default', 'deepseek', 'user_assist_rick', 'user_assist_duke', 'ajibawa_python', 'user_assist_art', 'user_assist_kyle', 'user_assist_hlsci']

if 'approved_login' not in st.session_state:
    st.session_state['approved_login'] = False

if st.session_state['approved_login'] == False:
    
    st.write("please login to continue")
    username = st.text_input("username")
    password = st.text_input("password", type="password")
    login_button = st.button('sign in')
    if login_button == True and username == "chad" and password == "chadc":
        llmonaid.popup_note(message=f"you have logged in {username}!")
        st.session_state['approved_login'] = True    
        st.session_state['user_type'] = 'admin'
        llmonaid.init_state(model_box_data, voice_box_data, chat_templates, user_type=st.session_state['user_type'])
        st.rerun()
    if login_button == True and username == "mikey" and password == "mikey42069":
        llmonaid.popup_note(message=f"you have logged in {username}!")
        st.session_state['approved_login'] = True    
        st.session_state['user_type'] = 'user_basic'
        llmonaid.init_state(model_box_data, voice_box_data, chat_templates, user_type=st.session_state['user_type'])
        st.rerun()

if st.session_state['approved_login']:
    # be the same list? ...
    default_context_list = [128, 256, 512, 1024, 2048, 4096, 8096, 16384, 32768]
    default_max_context_list = [1024, 2048, 4096, 8096, 16384, 32768]
    if st.session_state['user_type'] == 'user_basic':
        default_context_list = [128, 256, 512, 1024, 2048]
        default_max_context_list = [4096]

    model_box_dict = {}
    voice_box_dict = {}
    chat_template_dict = {}
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
    for i in default_context_list:
        default_context_dict[counter] = default_context_list[counter]
        counter += 1
    counter = 0
    for i in default_max_context_list:
        default_max_context_dict[counter] = default_max_context_list[counter]
        counter += 1

    clear_vram = st.toggle('clear vram')     
    tab2, tab1, tab4, tab3 = st.tabs(["💭chat model", "🔊audio", '👀 image gen/ocr', "🔗advanced"])
    with tab1:
        st.header("🔊audio")        
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
        st.audio(f"./voices/{st.session_state['voice_select']}")

    with tab2:
        st.header("💭chat model")
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
        st.session_state['char_name'] = st.text_input(label='model name', value='Johnny 5')

    with tab3:
        st.header("🔗advanced")
        st.session_state['verbose_chat'] = st.toggle('enable verbose console', value=st.session_state['verbose_chat'], disabled= llmonaid.check_user_type())

        set_context_index = 0
        for key, value in default_context_dict.items():
            if value == st.session_state['max_context_prompt']:
                set_context_index = key
        st.session_state['max_context_prompt'] = st.selectbox('max token gen', default_context_list, index=set_context_index, disabled=llmonaid.check_user_type())

        set_max_context_index = 0
        for key, value in default_max_context_dict.items():
            if value == st.session_state['max_context']:
                set_max_context_index = key
        st.session_state['max_context']= st.selectbox('max context size', default_max_context_list, index=set_max_context_index, disabled=llmonaid.check_user_type())

        st.session_state['torch_audio_cores'] = st.slider('torch audio cores', 2, 32, st.session_state['torch_audio_cores'], disabled=llmonaid.check_user_type())
        
        if st.session_state['user_type'] == 'user_basic':
            st.session_state['gpu_layer_count'] = st.slider('gpu layers', -1, 128, st.session_state['gpu_layer_count'])
            st.session_state['cpu_core_count'] = st.slider('cpu cores', 1, 4, st.session_state['cpu_core_count'])
            st.session_state['cpu_batch_count'] = st.slider('cpu batch cores', 1, 4, st.session_state['cpu_batch_count'])
        else:
            st.session_state['gpu_layer_count'] = st.slider('gpu layers', -1, 128, st.session_state['gpu_layer_count'])
            st.session_state['cpu_core_count'] = st.slider('cpu cores', 1, 128, st.session_state['cpu_core_count'])
            st.session_state['cpu_batch_count'] = st.slider('cpu batch cores', 1, 128,  st.session_state['cpu_batch_count'])
        st.session_state['batch_size'] = st.slider('batch size', 0, 1024, st.session_state['batch_size'])
        st.session_state['stream_chunk_size'] = st.slider('stream chunk size', 20, 200, st.session_state['stream_chunk_size'], disabled=llmonaid.check_user_type())
        st.session_state['chunk_pre_buffer'] = st.slider('chunk buffers', 0, 6, st.session_state['chunk_pre_buffer'], disabled=llmonaid.check_user_type())
    
    with tab4:
        st.header("👀image gen/ocr")
        st.header("sdxl turbo")
        st.session_state['enable_sdxl_turbo'] = st.toggle('enable sdxl turbo', value=st.session_state['enable_sdxl_turbo'], disabled=False)
        st.session_state['img2img_on'] = st.toggle('enable sdxl turbo img2img', value=st.session_state['img2img_on'], disabled=llmonaid.check_user_type())
        st.header("sdxl 1.0")
        st.session_state['enable_sdxl'] = st.toggle('enable sdxl 1.0', value=st.session_state['enable_sdxl'], disabled=llmonaid.check_user_type())
        st.header("moondream1")
        # disable if chat model or sdxl turbo etc, loaded
        st.session_state['enable_ocr'] = st.toggle('enable image ocr', value=st.session_state['enable_ocr'])
        st.session_state['ocr_device'] = st.toggle('run on cpu', value=st.session_state['ocr_device'], key='ocr_cpu_button')

    llmonaid.clear_buffers()
    if clear_vram:
        llmonaid.clear_vram()
        st.rerun()

st.json(st.session_state, expanded=False)
