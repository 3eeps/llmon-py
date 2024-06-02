import streamlit as st
import llmon
import os
from pywhispercpp.model import Model
from llama_cpp import Llama

st.set_page_config(page_title="llmon-py", page_icon="🍋", layout="centered", initial_sidebar_state="collapsed")
st.title('🍋 llmon-py', anchor='https://github.com/3eeps/llmon-py')

if 'init_app' not in st.session_state:
    llmon.init_state()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_model" not in st.session_state:
    st.session_state["chat_model"] = Llama(model_path=f"./llama-3-8b-instruct.gguf",
                                            n_batch=st.session_state['batch_size'],
                                            n_threads=st.session_state['cpu_core_count'],
                                            n_threads_batch=st.session_state['cpu_batch_count'],
                                            n_gpu_layers=int(st.session_state['gpu_layer_count']),
                                            n_ctx=st.session_state['max_context'],
                                            verbose=True)

if "moondream" not in st.session_state:
    llmon.Moondream.init()

if "sdxl_turbo" not in st.session_state:
    llmon.SDXLTurbo.init()

if 'speech_tt_model' not in st.session_state:
    st.session_state['speech_tt_model'] = Model(models_dir='./speech models', n_threads=10)

with st.sidebar:
    llmon.sidebar()

for message in st.session_state.messages:
    with st.chat_message(name=message["role"]):
        st.markdown(message["content"])

if user_text_prompt:= st.chat_input(placeholder=''):
    user_prompt = user_text_prompt
    if user_text_prompt == " ":
        user_prompt = llmon.Audio.voice_to_text()
    final_prompt = llmon.ChatTemplate.chat_template(prompt=user_prompt)

    with st.chat_message(name="user"):
        st.markdown(user_prompt)
        if st.session_state.bytes_data:
            try:
                st.image("ocr_upload_image.png", clamp=True)
                st.session_state.bytes_data = None
                os.remove("ocr_upload_image.png")
            except: pass
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.session_state['message_list'].append(f"""user: {user_prompt}""")

    model_output_text, st.session_state['model_output_tokens'] = llmon.model_inference(prompt=final_prompt)
    try:
        if st.session_state.function_calling:
            model_output_dict = eval(model_output_text)
            model_dict_values = [value for key, value in model_output_dict.items()]
            func_value_dict = model_dict_values.pop(1)
            output_function = model_dict_values.pop(0)
            value_name = [value for key, value in func_value_dict.items()]
            print(f"""{model_output_dict}\n{func_value_dict}\n{output_function}\n{value_name}""")
            
            if output_function == "answer_other_questions":
                st.session_state.function_results = "func_reply"

            if output_function == "describe_image":
                st.session_state.function_results = llmon.Moondream.generate_response(prompt=value_name[0])

            if output_function == "create_image":
                llmon.SDXLTurbo.generate_image(prompt=value_name[0])
                no_text_output = True
            
            if output_function == "video_player":
                st.session_state.video_link = llmon.Functions.find_youtube_link(user_query=value_name[0])
                st.session_state.first_watch = False
                no_text_output = True

            final_prompt = llmon.ChatTemplate.chat_template(prompt=user_prompt, function_result=st.session_state.function_results)
            if no_text_output == False:
                model_output_text, st.session_state['model_output_tokens'] = llmon.model_inference(prompt=final_prompt)
            if no_text_output: 
                model_output_text = ""
    except: pass

    with st.chat_message(name="assistant", avatar="🍋"):
        try:
            st.image('image_turbo.png')
            os.remove('image_turbo.png')
        except: pass
        if st.session_state.video_link and st.session_state.first_watch == False:
            try:
                st.session_state.first_watch = True
                st.video(data=st.session_state.video_link)
            except: pass
        st.markdown(model_output_text)
    st.session_state.messages.append({"role": "assistant", "content": model_output_text})
    st.session_state['message_list'].append(f"You: {model_output_text}")