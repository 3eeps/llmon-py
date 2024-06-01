# ./codespace/main.py
import streamlit as st
import llmon
import os
from melo_tts.api import TTS
from pywhispercpp.model import Model
from llama_cpp import Llama


######################## make sure to 
# add a check (a file) and if the file is there, you are the host... hopefully i can load model right away when connected remotely!!!


# make mini app to run streamlit from phone, etc, just run bat
    #os.system(r"C:/Users/User/desktop/llmonpy.bat ")

# add saving context.. easy? save to json

# add support for mistral code model or wavecoder6.7b

# try new moondream, also try ggufs!

# have llm react to enabling things instead of using 'toasts?'
# ex: you want a yt vid open, have to reply with soemthing liek: 'ahh, let me find that for you...' 

# add option to upload pdfs and access whats inside? scan the images too?

# set app page defaults
st.set_page_config(page_title="llmon-py", page_icon="🍋", layout="wide", initial_sidebar_state="expanded")

# init default app settings
if 'init_app' not in st.session_state:
    llmon.init_state()

# init empty message list on first run
if "messages" not in st.session_state:
    st.session_state.messages = []

# load models
if "chat_model" not in st.session_state:
    st.toast(body=f"🍋 :orange[loading {st.session_state['model_select']}...]")
    st.session_state["chat_model"] = Llama(model_path=f"./model/llama-3-8b-instruct.gguf",
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

if 'melo_model' not in st.session_state:
    st.session_state['melo_model'] = TTS(language='EN', device='auto')
    st.session_state['speaker_ids'] = st.session_state['melo_model'].hps.data.spk2id

if 'speech_tt_model' not in st.session_state:
    st.session_state['speech_tt_model'] = Model(models_dir='./speech models', n_threads=10)

# load this last to keep user from trying to use it, while models first load
with st.sidebar:
    llmon.SidebarConfig.sidebar()

# update chat window
for message in st.session_state.messages:
    with st.chat_message(name=message["role"]):
        st.markdown(message["content"])

# send user message through model template, starting inference
if user_text_prompt:= st.chat_input(placeholder=''):
    st.toast(body='🍋 :orange[generating...]')
    user_prompt = user_text_prompt
    if user_text_prompt == " ":
        user_prompt = llmon.Audio.voice_to_text()
    final_prompt = llmon.ChatTemplate.chat_template(prompt=user_prompt)

    # add user message to chat window
    with st.chat_message(name="user"):
        st.markdown(user_prompt)
        if st.session_state.bytes_data:
            try:
                st.image("ocr_upload_image.png", clamp=True)
                st.session_state.bytes_data = None
                os.remove("ocr_upload_image.png")
            except: pass
    # append user messages
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.session_state['message_list'].append(f"""user: {user_prompt}""")

    # inference model
    model_output_text, st.session_state['model_output_tokens'] = llmon.model_inference(prompt=final_prompt)
    try:
        if st.session_state.function_calling:
            # get function name and function values from model output
            model_output_dict = eval(model_output_text)
            model_dict_values = [value for key, value in model_output_dict.items()]
            func_value_dict = model_dict_values.pop(1)
            output_function = model_dict_values.pop(0)
            value_name = [value for key, value in func_value_dict.items()]
            print(f"""{model_output_dict}\n{func_value_dict}\n{output_function}\n{value_name}""")
            
            if output_function == "answer_other_questions":
                st.session_state.function_results = "func_reply"

            # output vision model response of uploaded image
            if output_function == "describe_image":
                st.session_state.function_results = llmon.Moondream.generate_response(prompt=value_name[0])

            # output sdxl_turbo image from users prompt
            if output_function == "create_image":
                llmon.SDXLTurbo.generate_image(prompt=value_name[0])
                no_text_output = True
            
            # search google for user youtube search query, disable text response
            if output_function == "video_player":
                st.session_state.video_link = llmon.Functions.find_youtube_link(user_query=value_name[0])
                st.session_state.first_watch = False
                no_text_output = True

            # rebuild model template, inference model to output function results
            final_prompt = llmon.ChatTemplate.chat_template(prompt=user_prompt, function_result=st.session_state.function_results)
            if no_text_output == False:
                model_output_text, st.session_state['model_output_tokens'] = llmon.model_inference(prompt=final_prompt)
            if no_text_output: 
                model_output_text = ""
    except: pass

    # play back model response with tts
    if st.session_state['mute_melo'] == False and no_text_output == False:
        llmon.Audio.melo_gen_message(message=model_output_text)

    # add model output to chat window
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
    # append model responses
    st.session_state.messages.append({"role": "assistant", "content": model_output_text})
    st.session_state['message_list'].append(f"You: {model_output_text}")