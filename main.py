# ./codespace/main.py
import streamlit as st
import llmon
import os
from melo_tts.api import TTS
from pywhispercpp.model import Model
from llama_cpp import Llama



# add saving context.. easy? save to json



st.set_page_config(page_title="model inference", page_icon="🍋", layout="wide", initial_sidebar_state="expanded")
if 'app_state_init' not in st.session_state:
    llmon.init_state()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "image_pipe_turbo" not in st.session_state and st.session_state.enable_sdxl_turbo:
    llmon.SDXLTurbo.init()

if "moondream" not in st.session_state and st.session_state.enable_moondream:
    llmon.Moondream.load_vision_encoder()

if 'melo_model' not in st.session_state:
    st.session_state['melo_model'] = TTS(language='EN', device='auto')
    st.session_state['speaker_ids'] = st.session_state['melo_model'].hps.data.spk2id
    
if 'speech_tt_model' not in st.session_state:
    st.session_state['speech_tt_model'] = Model(models_dir='./speech models', n_threads=10)

if "chat_model" not in st.session_state and st.session_state.model_loaded:
    st.toast(body=f"🍋 :orange[loading {st.session_state['model_select']}...]")
    st.session_state["chat_model"] = Llama(model_path=f"./models/{st.session_state['model_select']}",
                                            n_batch=st.session_state['batch_size'],
                                            n_threads=st.session_state['cpu_core_count'],
                                            n_threads_batch=st.session_state['cpu_batch_count'],
                                            n_gpu_layers=int(st.session_state['gpu_layer_count']),
                                            n_ctx=st.session_state['max_context'],
                                            verbose=True)
    st.session_state.disable_chat = False

with st.sidebar:
    llmon.SidebarConfig.memory_display()
    llmon.SidebarConfig.display_buttons()
    llmon.SidebarConfig.select_model()
    st.caption(f"context: :orange[{st.session_state['model_output_tokens']}/{st.session_state['max_context']}]")
    if st.session_state.enable_moondream:
        uploaded_file = st.file_uploader(label='file uploader', label_visibility='collapsed', type=['png', 'jpeg'])
        if uploaded_file:
            st.session_state.bytes_data = uploaded_file.getvalue()
            with open("ocr_upload_image.png", 'wb') as file:
                file.write(st.session_state.bytes_data)
    llmon.SidebarConfig.advanced_settings()
    llmon.SidebarConfig.app_exit_button()

if st.session_state.model_loaded:
    for message in st.session_state.messages:
        with st.chat_message(name=message["role"]):
            st.markdown(message["content"])
    no_text_output = False
    if user_text_prompt:= st.chat_input(placeholder='', disabled=st.session_state.disable_chat):
        st.toast(body='🍋 :orange[generating...]')
        user_prompt = user_text_prompt
        if user_text_prompt == " ":
            user_prompt = llmon.Audio.voice_to_text()

        final_prompt = llmon.ChatTemplates.update_chat_template(prompt=user_prompt, template_type=st.session_state['template_select'])
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

        model_output = st.session_state["chat_model"](prompt=final_prompt,
                                                        repeat_penalty=float(st.session_state['repeat_penalty']),
                                                        max_tokens=st.session_state['max_context'], 
                                                        top_k=int(st.session_state['model_top_k']),
                                                        top_p=float(st.session_state['model_top_p']),
                                                        min_p=float(st.session_state['model_min_p']),
                                                        temperature=float(st.session_state['model_temperature']))
        model_response = model_output['choices'][0]['text']
        st.session_state['model_output_tokens'] = model_output['usage']['total_tokens']

        try:
            output_dict = eval(model_response)
            values_list = [value for key, value in output_dict.items()]
            if values_list is not None:
                func_dict = values_list.pop(1)
                func_name = values_list.pop(0)
                value_name = [value for key, value in func_dict.items()]
                print(f"""{output_dict}\n{func_dict}\n{func_name}\n{value_name}""")
                remove_slash = func_name.replace("\\", "")
                func_name = remove_slash
                
                if func_name == "describe_image":
                    st.session_state.function_results = llmon.Moondream.generate_response(prompt=value_name[0])

                if func_name == "create_image":
                    llmon.SDXLTurbo.generate_image(prompt=value_name[0])
                    no_text_output = True
                
                if func_name == "video_player":
                    st.session_state.video_link = llmon.Functions.youtube_download(link=value_name[0])
                    st.session_state.first_watch = False
                    no_text_output = True

                final_prompt = llmon.ChatTemplates.update_chat_template(prompt=user_prompt, template_type=st.session_state['template_select'], function_result=st.session_state.function_results)
                if no_text_output == False:
                    model_output = st.session_state["chat_model"](prompt=final_prompt,
                                                        repeat_penalty=float(st.session_state['repeat_penalty']),
                                                        max_tokens=st.session_state['max_context'], 
                                                        top_k=int(st.session_state['model_top_k']),
                                                        top_p=float(st.session_state['model_top_p']),
                                                        min_p=float(st.session_state['model_min_p']),
                                                        temperature=float(st.session_state['model_temperature']))
                    model_response = model_output['choices'][0]['text']
                if no_text_output: 
                    model_response = ""
                st.session_state['model_output_tokens'] = model_output['usage']['total_tokens']
        except: pass

        if st.session_state['mute_melo'] == False and no_text_output == False:
            llmon.Audio.melo_gen_message(message=model_response)

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
            st.markdown(model_response)
            st.session_state.messages.append({"role": "assistant", "content": model_response})
            st.session_state['message_list'].append(f"You: {model_response}")