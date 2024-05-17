# ./codespace/main.py
import streamlit as st
import llmon
import os

st.set_page_config(page_title="model inference", page_icon="🍋", layout="wide", initial_sidebar_state="expanded")
if 'app_state_init' not in st.session_state:
    llmon.init_state()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "image_pipe_turbo" not in st.session_state and st.session_state.enable_sdxl_turbo:
    st.toast(body='🍋 :orange[loading sdxl turbo...]')
    llmon.SDXLTurbo.init()

if "moondream" not in st.session_state and st.session_state.enable_moondream:
    st.toast(body='🍋 :orange[loading moondream2...]')
    llmon.Moondream.load_vision_encoder()

if 'melo_model' not in st.session_state and st.session_state['enable_melo'] and st.session_state.bite_llmon:
    st.toast(body='🍋 :orange[loading melotts...]')
    from melo_tts.api import TTS
    st.session_state['melo_model'] = TTS(language='EN', device='auto')
    st.session_state['speaker_ids'] = st.session_state['melo_model'].hps.data.spk2id

if 'xtts_model' not in st.session_state and st.session_state['enable_xtts'] and st.session_state.bite_llmon:
    st.toast(body='🍋 :orange[loading xtts...]')
    llmon.XttsTTS.load_xtts()
    
if 'speech_tt_model' not in st.session_state:
    st.toast(body='🍋 :orange[loading pywhispercpp...]')
    from pywhispercpp.model import Model
    st.session_state['speech_tt_model'] = Model(models_dir='./speech models', n_threads=10)

if "chat_model" not in st.session_state and st.session_state.bite_llmon:
    st.toast(body=f"🍋 :orange[loading {st.session_state['model_select']}...]")
    from llama_cpp import Llama
    st.session_state["chat_model"] = Llama(model_path=f"./models/{st.session_state['model_select']}",
                                            n_batch=st.session_state['batch_size'],
                                            n_threads=st.session_state['cpu_core_count'],
                                            n_threads_batch=st.session_state['cpu_batch_count'],
                                            n_gpu_layers=int(st.session_state['gpu_layer_count']),
                                            n_ctx=st.session_state['max_context'],
                                            verbose=True)

with st.sidebar:
    llmon.memory_display()
    st.markdown("""
            <style>
                div[data-testid="column"] {
                    width: fit-content !important;
                    flex: unset;
                }
                div[data-testid="column"] {
                    width: fit-content !important;
                }
            </style>
            """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.button(label=':orange[load/refresh]')
    with col2:
        if st.button(label=':orange[clear context]'):
            st.session_state['message_list'] = []
            st.session_state.messages = []
            st.session_state['model_output_tokens'] = 0
    with col3:
        if st.button(label=':orange[unload chat model]'):
            llmon.clear_vram()
            st.rerun()

    @st.experimental_fragment
    def select_model():
        st.session_state['model_select'] = st.selectbox(label=':orange[model]', options=st.session_state.model_list, disabled=st.session_state.model_loader, label_visibility=st.session_state.model_label)
        if st.session_state.model_picked != st.session_state['model_select']:
            try:
                del st.session_state['chat_model']
                st.session_state.model_picked = None
                st.rerun()
            except: pass
        st.session_state.enable_moondream = st.toggle(label=':orange[moondream]', value=st.session_state.enable_moondream)
        st.session_state.enable_sdxl_turbo = st.toggle(label=':orange[sdxl turbo]', value=st.session_state.enable_sdxl_turbo)
        st.session_state.bite_llmon = True
        st.session_state.model_picked = st.session_state['model_select']
    select_model()

    if st.session_state['model_select'] == 'mistral-7b-instruct.gguf':
        st.session_state['template_select'] = st.session_state.chat_templates[1]
        st.session_state['model_temperature'] = 0.1
    if st.session_state['model_select'] == 'llama-3-8b-instruct.gguf':
        st.session_state['template_select'] = st.session_state.chat_templates[0]
        st.session_state['model_temperature'] = 0.85

    @st.experimental_fragment
    def advanced_settings():
        if st.checkbox(label=':orange[advanced settings]'):
            if st.button(label=':orange[unload image/vision models]'):
                llmon.clear_vram(models_to_unload=['moondream', 'image_pipe_turbo'])
                st.rerun()
            st.caption(f"context length: :orange[{st.session_state['model_output_tokens']}/{st.session_state['max_context']}]")
            st.session_state.sys_prompt = st.text_area(label=':orange[custom prompt]', value="")

            uploaded_file = st.file_uploader(label='file uploader', label_visibility='collapsed')
            if uploaded_file:
                st.session_state.bytes_data = uploaded_file.getvalue()
                with open("ocr_upload_image.png", 'wb') as file:
                    file.write(st.session_state.bytes_data)

            disable_xtts = False
            disable_melo = False
            if st.session_state['enable_melo']:
                disable_xtts = True
                disable_melo = False
            if st.session_state['enable_xtts']:
                disable_melo = True
                disable_xtts = False
            st.caption(body="text-to-speech")
            st.session_state['enable_xtts'] = st.checkbox(':orange[enable xttsv2]', value=st.session_state['enable_xtts'], disabled=disable_xtts)
            st.session_state['enable_melo'] = st.checkbox(':orange[enable melotts]', value=st.session_state['enable_melo'], disabled=disable_melo)

            st.caption(body="model parameters")
            st.session_state['model_temperature'] = st.text_input(label=':orange[temperature]', value=st.session_state['model_temperature'])
            st.session_state['model_top_p'] = st.text_input(label=':orange[top p]', value=st.session_state['model_top_p'] )
            st.session_state['model_top_k'] = st.text_input(label=':orange[top k]', value=st.session_state['model_top_k'])
            st.session_state['model_min_p'] = st.text_input(label=':orange[min p]', value=st.session_state['model_min_p'])
            st.session_state['repeat_penalty'] = st.text_input(label=':orange[repeat_penalty]', value=st.session_state['repeat_penalty'])
    advanced_settings()

st.title(f"🍋model inference")
if st.session_state.bite_llmon:
    for message in st.session_state.messages:
        with st.chat_message(name=message["role"]):
            st.markdown(message["content"])

    model_response = ""
    user_prompt = ""
    if user_text_prompt:= st.chat_input():
        user_prompt = user_text_prompt
        
        if user_text_prompt == " ":
            user_prompt = llmon.voice_to_text()

        final_prompt = llmon.update_chat_template(prompt=user_prompt, template_type=st.session_state['template_select'])
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

        func_call = False
        no_text_output = False
        try:
            output_dict = eval(model_response)
            print(output_dict)
            values_list = [value for key, value in output_dict.items()]
            if values_list is not None:
                func_dict = values_list.pop(1)
                func_name = values_list.pop(0)
                value_name = [value for key, value in func_dict.items()]
                func_call = True

                print(func_dict)
                remove_slash = func_name.replace("\\", "")
                func_name = remove_slash
                print(func_name)
                print(value_name)

                if func_name == "get_stock_price":
                    st.toast(body='🍋 :orange[invest to impress...]')
                    st.session_state.function_results = llmon.FunctionCall.get_stock_price(symbol=value_name[0])

                if func_name == "get_city_weather":
                    st.toast(body='🍋 :orange[weathermen never lie...]')
                    st.session_state.function_results = f"""Use Markdown language to make the weather data quickly readable for the user. Data: {llmon.FunctionCall.get_weather(city=value_name[0])}"""

                if func_name == "describe_image":
                    st.toast(body='🍋 :orange[taking a look...]')
                    st.session_state.function_results = llmon.Moondream.generate_response(prompt=value_name[0])

                if func_name == "get_news":
                    st.toast(body='🍋 :orange[assembling news team 6...]')
                    st.session_state.function_results = f"Breakdown the information inside each news story into a professional presentation using the Markdown. Remove text that is not part of the articles paragraph. Use Markdown to present the image urls (example: ![news story Image](http://www.url.com/article_image.png)): {llmon.FunctionCall.get_news(value_name[0])}"""

                if func_name == "create_image":
                    st.toast(body='🍋 :orange[generating image...]')
                    llmon.SDXLTurbo.generate_image(prompt=value_name[0])
                    no_text_output = True

                if func_name == "answer_other_questions":
                    st.session_state.function_results = "func_reply"

                if func_name == "video_player":
                    st.toast(body='🍋 :orange[finding that 240p video from 2007...]')
                    st.session_state.video_link = llmon.FunctionCall.youtube_download(link=value_name[0])
                    st.session_state.first_watch = False
                    no_text_output = True

                final_prompt = llmon.update_chat_template(prompt=user_prompt, template_type=st.session_state['template_select'], function_result=st.session_state.function_results)
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

        if st.session_state['enable_melo']:
            st.toast(body='🍋 :orange[generating audio with melotts...]')              
            llmon.melo_gen_message(message=model_response)
            
        if st.session_state['enable_xtts']:
            st.toast(body='🍋 :orange[generating audio with xtts...]')
            llmon.XttsTTS.play_back_speech(prompt=model_response)

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