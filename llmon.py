import streamlit as st
import os
import sounddevice
import psutil
import json
import keyboard
from scipy.io.wavfile import write as write_wav
from googlesearch import search
from pywhispercpp.model import Model
from diffusers import AutoPipelineForText2Image
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

def model_inference(prompt=""):
    if st.session_state.function_calling:
        st.session_state['model_tempurature'] = 0.85
        st.session_state['model_min_p'] = 0.06
        st.session_state['model_top_p'] = 0.0
        st.session_state['model_top_k'] = 0
        st.session_state['repeat_penalty'] = 1.1

    model_output = st.session_state["chat_model"](prompt=prompt,
                                                        repeat_penalty=float(st.session_state['repeat_penalty']),
                                                        max_tokens=st.session_state['max_context'], 
                                                        top_k=int(st.session_state['model_top_k']),
                                                        top_p=float(st.session_state['model_top_p']),
                                                        min_p=float(st.session_state['model_min_p']),
                                                        temperature=float(st.session_state['model_temperature']))

    return model_output['choices'][0]['text']

def clear_vram():
    model_list = ['chat_model', 'sdxl_turbo', 'moondream']
    for model_in_vram in model_list:
        del st.session_state[model_in_vram]

def init_state():
    default_settings_state = {
        'max_context': 8192,
        'gpu_layer_count': -1,
        'cpu_core_count': 8,
        'cpu_batch_count': 8,
        'batch_size': 256,
        'model_temperature': 0.85,
        'model_top_p': 0.0,
        'model_top_k': 0,
        'model_min_p': 0.06,
        'repeat_penalty': 1.1,
        'message_list': [],
        'init_app': True,
        'model_select': 'Meta-Llama-3-8B-Instruct.Q6_K.gguf'}
    st.session_state.show_start_card = True
    st.session_state.start_app = False
    st.session_state.function_calling = False
    st.session_state.function_results = ""
    st.session_state.custom_template = ""
    st.session_state.video_link = None
    st.session_state.first_watch = False
    st.session_state.bytes_data = None

    with open("functions.json", "r") as file:
        st.session_state.functions = json.load(file)

    for key, value in default_settings_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

def sidebar(): 
    col1, col2, col3 = st.columns([1,1,1])
    with col3:
        if st.button(label=':orange[new chat]'):
            st.session_state['message_list'] = []
            st.session_state.messages = []
            st.session_state.function_calling = False
            st.session_state.custom_template = ""

    st.title('🍋 llmon-py', anchor='https://github.com/3eeps/llmon-py')

    uploaded_file = st.file_uploader(label='file uploader', label_visibility='collapsed', type=['png', 'jpeg'])
    if uploaded_file:
        st.session_state.bytes_data = uploaded_file.getvalue()
        with open("ocr_upload_image.png", 'wb') as file:
            file.write(st.session_state.bytes_data)

    st.session_state.function_calling = st.toggle(':orange[enable function calling] :green[(beta)]', value=st.session_state.function_calling)
    st.caption(body="custom model template")
    st.session_state.custom_template = st.text_area(label='custom prompt', value="", label_visibility='collapsed', disabled=st.session_state.function_calling)
    st.caption(body="model parameters")
    temp_help = "determinines whether the output is more random and creative or more predictable. :green[a higher temperature will result in lower probability], i.e more creative outputs."
    top_p_help = "controls the diversity of the generated text by only considering tokens with the highest probability mass. :green[top_p = 0.1: only tokens within the top 10% probability are considered. 0.9: considers tokens within the top 90% probability]."
    top_k_help = "limits the model's output to the top-k most probable tokens at each step. This can help reduce incoherent or nonsensical output by restricting the model's vocabulary. :green[a top-K of 1 means the next selected token is the most probable among all tokens in the model's vocabulary]."
    min_p_help = "different from top k or top p, sets a minimum percentage requirement to consider tokens relative to the largest token probability. :green[for example, min p = 0.1 is equivalent to only considering tokens at least 1/10th the top token probability]."
    rep_help = "helps the model generate more diverse content instead of repeating previous phrases. Repetition is prevented by applying a high penalty to phrases or words that tend to be repeated. :green[a higher penalty generally results in more diverse outputs, whilst a lower value might lead to more repetition]."
    st.session_state['model_temperature'] = st.text_input(label=':orange[temperature]', value=st.session_state['model_temperature'], disabled=st.session_state.function_calling, help=temp_help)
    st.session_state['model_top_p'] = st.text_input(label=':orange[top p]', value=st.session_state['model_top_p'], disabled=st.session_state.function_calling, help=top_p_help)
    st.session_state['model_top_k'] = st.text_input(label=':orange[top k]', value=st.session_state['model_top_k'], disabled=st.session_state.function_calling, help=top_k_help)
    st.session_state['model_min_p'] = st.text_input(label=':orange[min p]', value=st.session_state['model_min_p'], disabled=st.session_state.function_calling, help=min_p_help)
    st.session_state['repeat_penalty'] = st.text_input(label=':orange[repetition penalty]', value=st.session_state['repeat_penalty'], disabled=st.session_state.function_calling, help=rep_help)
    
    col_1, col_2, col_3 = st.columns([1,1,1])
    with col_2:
        if st.button(":orange[shutdown]", help='shut down app on server side'):
            clear_vram()
            try:
                os.remove('.google-cookie')
            except: pass
            try:
                os.remove('server_side.log')
            except: pass
            keyboard.press_and_release('ctrl+w')
            llmon_process_id = os.getpid()
            process = psutil.Process(llmon_process_id)
            process.terminate()

class ChatTemplate:
    def chat_template(prompt="", function_result=""):
        system_message = f"""You are an helpful AI assistant, answer any request the user may have. Share postive or negative considerations when appropriate, but keep them concise. Conversation history: {st.session_state['message_list']}"""
        template = f"""<|begin_of_text|<|start_header_id|>system<|end_header_id|>{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        if st.session_state.function_calling:
            system_message = f"""As an AI assistant with function calling support, you are provided with the following functions: {json.dumps(st.session_state.functions)}
            When the user asks a question that can be answered with a function, do not describe the function or wrap the function in "||", only output the function filled with the appropriate data required as a Python dictionary."""
            system_message_plus_example = """Example message: 'Hello there!' Your reply: '{'function_name': 'user_chat', 'parameters': {'user_message': 'Hello there!'}}'"""
            template = f"""<|begin_of_text|<|start_header_id|>system<|end_header_id|>{system_message + system_message_plus_example}<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            normal_reply = False
            if function_result == "func_reply":
                normal_reply = True
                system_message = f"""You are an helpful AI assistant, answer any request the user may have. Share postive or negative considerations when appropriate. Conversation history: {st.session_state['message_list']}"""
                template = f"""<|begin_of_text|<|start_header_id|>system<|end_header_id|>{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            if len(function_result) > 1 and normal_reply == False:
                system_message = f"""Using this data that is up to date, reply to the user using it: {function_result}. The question the user asked was:"""
                template = f"""<|begin_of_text|<|start_header_id|>system<|end_header_id|>{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        if len(st.session_state.custom_template) > 1:
            system_message = f"""{st.session_state.custom_template} Chat history: {st.session_state['message_list']}"""
            template = f"""<|begin_of_text|<|start_header_id|>system<|end_header_id|>{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""                 

        return template

class Audio:
    def voice_to_text():
        rec_user_voice = sounddevice.rec(int(st.session_state['user_audio_length']) * 44100, samplerate=44100, channels=2)
        sounddevice.wait()
        write_wav(filename='user_output.wav', rate=44100, data=rec_user_voice)
        st.session_state['speech_tt_model'] = Model(models_dir='./ggml-tiny.bin', n_threads=10)
        user_voice_data = st.session_state['speech_tt_model'].transcribe('user_output.wav', speed_up=True)
        os.remove(f"user_output.wav")
        text_data = []
        for voice in user_voice_data:        
            text_data.append(voice.text)
        combined_text = ' '.join(text_data)
        return combined_text

class Functions:
    def find_youtube_link(user_query=str):
        search_helper = 'youtube'
        search_query = user_query + search_helper
        for youtube_link in search(query=search_query, tld="co.in", num=1, stop=1, pause=2):
            print(youtube_link)
        try:
            os.remove('.google-cookie')
        except: pass
        return youtube_link

class SDXLTurbo:
    def init():
        st.session_state['sdxl_turbo'] = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", variant="fp16").to('cpu')
    
    def generate_image(prompt="", steps=1):
        _image = st.session_state['sdxl_turbo'](prompt=prompt, num_inference_steps=steps, guidance_scale=0.0).images[0]
        _image.save('image_turbo.png')

class Moondream:
    def init():
        model_id = "vikhyatk/moondream2"
        revision = "2024-05-20"
        st.session_state['moondream'] = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision)
        st.session_state.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def generate_response(prompt=str):
        filename = "ocr_upload_image.png"
        image = Image.open(filename)
        enc_image = st.session_state['moondream'].encode_image(image)
        return st.session_state['moondream'].answer_question(enc_image, prompt, st.session_state.tokenizer)