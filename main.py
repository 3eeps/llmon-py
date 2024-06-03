import streamlit as st
import llmon
import os
from pywhispercpp.model import Model
from llama_cpp import Llama

st.set_page_config(page_title="llmon-py", page_icon="page_icon.png", layout="centered", initial_sidebar_state="collapsed")

if 'init_app' not in st.session_state:
    llmon.init_state()

if st.session_state.start_app == False:
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.image('llmon_logo.png')
        submit_button = st.button(":orange[click here to start]")
        if submit_button:
            st.session_state.start_app = True
            st.rerun()

lcol1, lcol2, lcol3 = st.columns([1,1,1])            
with lcol2:
    with st.spinner('🍋 initializing...'):
        if "messages" not in st.session_state and st.session_state.start_app:
            st.session_state.messages = []

        if "chat_model" not in st.session_state and st.session_state.start_app:
            st.session_state["chat_model"] = Llama(model_path=f"./{st.session_state['model_select']}",
                n_batch=st.session_state['batch_size'],
                n_threads=st.session_state['cpu_core_count'],
                n_threads_batch=st.session_state['cpu_batch_count'],
                n_gpu_layers=int(st.session_state['gpu_layer_count']),
                n_ctx=st.session_state['max_context'])

        if "moondream" not in st.session_state and st.session_state.start_app:
            llmon.Moondream.init()

        if "sdxl_turbo" not in st.session_state and st.session_state.start_app:
            llmon.SDXLTurbo.init()

        if 'speech_tt_model' not in st.session_state and st.session_state.start_app:
            st.session_state['speech_tt_model'] = Model(models_dir='./speech models', n_threads=10)

if st.session_state.start_app:
    with st.sidebar:
        llmon.sidebar()

if st.session_state.start_app:
    for message in st.session_state.messages:
        with st.chat_message(name=message['role']):
            st.write(message["content"])

if st.session_state.start_app:
    if st.session_state.show_start_card:
        st.session_state.show_start_card = False
        with lcol2:
            st.markdown('🍋 llmon-py')

    if user_text_input:= st.chat_input(placeholder=''):
        user_input = user_text_input
        if user_input == " ":
            try: user_input = llmon.Audio.voice_to_text()
            except: pass

        with st.chat_message(name="user"):
            st.write(user_input)
            if st.session_state.bytes_data:
                try:
                    st.image("ocr_upload_image.png", clamp=True)
                    st.session_state.bytes_data = None
                    os.remove("ocr_upload_image.png")
                except: pass
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state['message_list'].append(f"""user: {user_input}""")

        final_template = llmon.ChatTemplate.chat_template(prompt=user_input)
        model_output_text = llmon.model_inference(prompt=final_template)
        try:
            if st.session_state.function_calling:
                model_output_dict = eval(model_output_text)
                model_dict_values = [value for key, value in model_output_dict.items()]
                func_value_dict = model_dict_values.pop(1)
                output_function = model_dict_values.pop(0)
                value_name = [value for key, value in func_value_dict.items()]
                print(f"""{model_output_dict}\n{func_value_dict}\n{output_function}\n{value_name}""")
                
                if output_function == "user_chat":
                    st.session_state.function_results = 'func_reply'
                    text_output = True

                if output_function == "describe_image":
                    st.session_state.function_results = llmon.Moondream.generate_response(prompt=value_name[0])

                if output_function == "create_image":
                    llmon.SDXLTurbo.generate_image(prompt=value_name[0])
                    text_output = False
                
                if output_function == "video_player":
                    st.session_state.video_link = llmon.Functions.find_youtube_link(user_query=value_name[0])
                    st.session_state.first_watch = False
                    display_link = True
                    st.session_state.messages.append({"role": "assistant", "content": value_name[0]})
                    st.session_state['message_list'].append(f"You: {value_name[0]}")

                final_function_template = llmon.ChatTemplate.chat_template(prompt=user_input, function_result=st.session_state.function_results)
                if text_output:
                    model_output_text = llmon.model_inference(prompt=final_function_template)
                if text_output == False: 
                    model_output_text = ""
                if display_link:
                    model_output_text = value_name[0]

        except: pass

        with st.chat_message(name="assistant"):
            st.write(model_output_text)
            try:
                st.image('image_turbo.png')
                os.remove('image_turbo.png')
            except: pass
            if st.session_state.video_link and st.session_state.first_watch == False:
                st.session_state.first_watch = True
                st.video(data=st.session_state.video_link)
            st.caption('4096/8192')
        st.session_state.messages.append({"role": "assistant", "content": model_output_text})
        st.session_state['message_list'].append(f"You: {model_output_text}")