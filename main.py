import streamlit as st
st.set_page_config(page_title="llmon-py", page_icon="page_icon.png", layout="centered", initial_sidebar_state="collapsed")
lcol1, lcol2, lcol3 = st.columns([1,1,1])
with lcol2:
    with st.spinner('🍋 initializing...'):
        import llmon
        from pywhispercpp.model import Model
        from llama_cpp import Llama

llmon.hide_deploy_button()
if 'init_app' not in st.session_state:
    llmon.init_state()
if st.session_state.init_app == False:
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.write('🍋 welcome to llmon-py!')
        model_picked = st.selectbox(label='first model list', options=st.session_state.model_list, label_visibility='hidden')
        if st.button(":orange[click here to start!]"):
            st.session_state['model_select'] = model_picked
            st.session_state.start_app = True
            st.session_state.init_app = True
            st.rerun()

lcol1, lcol2, lcol3 = st.columns([1,1,1])            
with lcol2:
    with st.spinner('🍋 loading your model...'):
        if "messages" not in st.session_state and st.session_state.start_app:
            st.session_state.messages = []

        if "chat_model" not in st.session_state and st.session_state.start_app:
            st.session_state["chat_model"] = Llama(model_path=f"./{st.session_state['model_select']}",
                n_batch=st.session_state['batch_size'],
                n_threads=st.session_state['cpu_core_count'],
                n_threads_batch=st.session_state['cpu_batch_count'],
                n_gpu_layers=int(st.session_state['gpu_layer_count']),
                n_ctx=st.session_state['max_context'])

        #if "sd3_medium" not in st.session_state and st.session_state.start_app and st.session_state['model_select'] != 'DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf':
            #llmon.SD3Medium.init()

        if 'speech_tt_model' not in st.session_state and st.session_state.start_app:
            st.session_state['speech_tt_model'] = Model(models_dir='./speech models', n_threads=10)

if st.session_state.start_app:
    with st.sidebar:
        llmon.sidebar()
    for message in st.session_state.messages:
        with st.chat_message(name=message['role']):
            try:
                st.markdown(f"""<img src="data:png;base64,{message['image']}">""", unsafe_allow_html=True)
            except: pass
            st.write(message["content"])

    if user_text_input:= st.chat_input(placeholder=''):
        user_input = user_text_input
        if user_input == " ":
            try: user_input = llmon.Audio.voice_to_text()
            except: pass

        with st.chat_message(name="user"):
            st.write(user_input)
            if st.session_state.show_uploaded_image:
                st.session_state.messages.append({"role": "user", "content": user_input, "image": st.session_state.bytes_data})
            elif st.session_state.show_uploaded_image == False:
                st.session_state.messages.append({"role": "user", "content": user_input})

            st.session_state['message_list'].append(f"""user: {user_input}""")
            st.session_state.show_uploaded_image = False

        with st.spinner('🍋 generating...'):
            final_template = llmon.ChatTemplate.chat_template(prompt=user_input)
            model_output_text, st.session_state.token_count = llmon.model_inference(prompt=final_template) 
            user_chat = False
            if st.session_state.function_calling:
                try:
                    model_output_dict = eval(model_output_text)
                    model_dict_values = [value for key, value in model_output_dict.items()]
                    func_value_dict = model_dict_values.pop(1)
                    output_function = model_dict_values.pop(0)
                    value_name = [value for key, value in func_value_dict.items()]
                    print(f"""{model_output_dict}\n{func_value_dict}\n{output_function}\n{value_name}""")
                
                    if output_function == "no_function_message":
                        user_chat = True
                        text_output = True

                    if output_function == "change_voice_style":
                        llmon.Functions.change_llm_voice(voice_description=value_name[0])
                        voice_reply = True

                    if output_function == "describe_image":
                        st.session_state.show_uploaded_image = True
                        text_output = True

                    if output_function == "create_image":
                        llmon.SD3Medium.generate_image(prompt=value_name[0])
                        st.session_state.show_generated_image = True
                    
                    if output_function == "video_player":
                        st.session_state.video_link = llmon.Functions.find_youtube_link(user_query=value_name[0])

                    final_function_template = llmon.ChatTemplate.chat_template(prompt=user_input, function_result=st.session_state.function_results)
                    model_output_text = ""
                    if user_chat:
                        st.session_state.function_calling = False
                        final_template = llmon.ChatTemplate.chat_template(prompt=user_input)
                        model_output_text, st.session_state.token_count = llmon.model_inference(prompt=final_template)
                        st.session_state.function_calling = True
                    if text_output and user_chat == False:
                        model_output_text, st.session_state.token_count = llmon.model_inference(prompt=final_function_template)
                    if voice_reply:
                        st.session_state.function_calling = False
                        voice_prompt = f"Pretend that I just gave you a new text to speech voice with this description: {value_name[0]}. Say something fun to show it off!"
                        final_template = llmon.ChatTemplate.chat_template(prompt=voice_prompt)
                        model_output_text, st.session_state.token_count = llmon.model_inference(prompt=final_function_template)
                        st.session_state.function_calling = True
                except: pass

        with st.chat_message(name="assistant"):
            st.write_stream(llmon.stream_text(text=model_output_text))
            if st.session_state.show_generated_image:
                st.markdown(f"""<img src="data:png;base64,{st.session_state.sdxl_base64}">""", unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": model_output_text, "image": st.session_state.sdxl_base64})
                st.session_state['message_list'].append(f"You: Generated an image based on the users request.")

            if st.session_state.show_generated_image == False:
                st.session_state.messages.append({"role": "assistant", "content": model_output_text})
                st.session_state['message_list'].append(f"You: {model_output_text}")

            if st.session_state.video_link:
                st.video(data=st.session_state.video_link)
                st.session_state.messages.append({"role": "assistant", "content": f"{st.session_state.video_link}"})
                st.session_state['message_list'].append(f"You: Displayed a Youtube video based on the users search request.")  

            st.session_state.show_generated_image = False
            st.session_state.video_link = None