# ./codespace/llmon.py
import streamlit as st
import os
import torch
import sounddevice
import GPUtil as GPU
import psutil
import base64
import json
import keyboard
from scipy.io.wavfile import write as write_wav
from googlesearch import search
from pywhispercpp.model import Model
from diffusers import AutoPipelineForText2Image
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

def scan_dir(directory):
    directory_list = []
    for file in os.scandir(f'{directory}'):
        if file.is_file():
            directory_list.append(file.name)
    return directory_list or None

def clear_vram():
    model_list = ['chat_model', 'image_pipe_turbo', 'moondream', 'melo_model', 'speaker_ids', 'speech_tt_model']
    toggled_on_list = ['enable_moondream', 'enable_sdxl_turbo']
    for toggle in toggled_on_list:
        if st.session_state[toggle]:
            st.session_state[toggle] = False
    for model in model_list:
        try:
            del st.session_state[model]
        except: pass

def init_state():
    default_settings_state = {'user_audio_length': 8,
                            'max_context': 8192,
                            'gpu_layer_count': -1,
                            'cpu_core_count': 8,
                            'cpu_batch_count': 8,
                            'batch_size': 256,
                            'model_output_tokens': 0,
                            'model_temperature': 0.85,
                            'model_top_p': 0.0,
                            'model_top_k': 0,
                            'model_min_p': 0.06,
                            'repeat_penalty': 1.1,
                            'message_list': [],
                            'app_state_init': True,
                            'mute_melo': True,
                            'model_select': 'llama-3-8b-instruct.gguf'}
    st.session_state.model_picked = 'llama-3-8b-instruct.gguf'
    st.session_state.enable_moondream = False
    st.session_state.enable_sdxl_turbo = False
    st.session_state.pause_model_load = False
    with open("functions.json", "r") as file:
        st.session_state.functions = json.load(file)
    st.session_state.disable_chat = True
    st.session_state.model_loaded = False
    st.session_state.sys_prompt = ""
    st.session_state.model_list = scan_dir('./models')
    st.session_state.chat_templates = ['func_llama3', 'func_mistral']
    st.session_state.video_link = None
    st.session_state.first_watch = False
    st.session_state.function_results = ""
    st.session_state.bytes_data = None
    for key, value in default_settings_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

class ChatTemplates:
    def update_chat_template(prompt="", template_type="", function_result=""):
        template = ""
        if template_type == "func_mistral":
            func_mistral = f"""<s>[INST]You are a function calling AI model. You are provided with the following functions: 
            Functions: {json.dumps(st.session_state.functions)}
            When the user asks a question that can be answered with one of the above functions, only output the function filled with the appropriate data required as a python dictionary.
            Do not describe the function. Only output functions found in the json file provided to you.
            {prompt} [/INST]"""
            template = func_mistral

            normal_reply = False
            if function_result == "func_reply":
                normal_reply = True
                func_mistral = f"""<s>[INST]You are an AI assistant who acts more as the users best friend.
                You do not tell the user you are going to answer any request they may have, but will also maintain a laid back and chilled out response to any inquiries the user has.
                The user is roughly 35 years old and is well versed in most topics. Do not answer questions in long paragraphs but more quick and precise. Act more as a close friend then a AI assistant. Conversation history: {st.session_state['message_list']}
                {prompt} [/INST]"""
                template = func_mistral

            if len(function_result) > 1 and normal_reply == False:
                func_mistral = f"""<s>[INST]The user has asked this question: {prompt}. {function_result}. [/INST]"""
                template = func_mistral

            if len(st.session_state.sys_prompt) > 1:
                func_mistral = f"""<s>[INST]{st.session_state.sys_prompt} Chat history: {st.session_state['message_list']}

                {prompt} [/INST]"""
                template = func_mistral

        if template_type == "func_llama3":
            system_message = f"""As an AI assistant with function calling support, you are provided with the following functions: 
            Functions: {json.dumps(st.session_state.functions)}
            Chat history: {st.session_state['message_list']}
            When the user asks a question that can be answered with one of the above functions, only output the function filled with the appropriate data required as a python dictionary.
            Only output functions found in the json file provided to you. Do not describe the function, 'present it', or wrap it in markdown. Do not say: Here is the function call:"""
            system_message_plus_example = """Example: User: 'play brother ali take me home' You: '{'function_name': 'video_player', 'parameters': {'youtube_query': 'brother ali take me home'}}'"""
            func_llama3 = f"""<|begin_of_text|<|start_header_id|>system<|end_header_id|>
            {system_message + system_message_plus_example}<|eot_id|><|start_header_id|>user<|end_header_id|>
            {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            template = func_llama3

            normal_reply = False
            if function_result == "func_reply":
                normal_reply = True
                system_message = f"""You are an AI assistant who acts more as the users best friend.
                You do not tell the user you are going to answer any request they may have, but will also maintain a laid back and chilled out response to any inquiries the user has.
                The user is roughly 35 years old and is well versed in most topics. Do not answer questions in long paragraphs but more quick and precise. Act more as a close friend then a AI assistant. Conversation history: {st.session_state['message_list']}"""
                func_llama3 = f"""<|begin_of_text|<|start_header_id|>system<|end_header_id|>
                {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
                {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
                template = func_llama3

            if len(function_result) > 1 and normal_reply == False:
                system_message = f"""The user has asked this question: {prompt}. Answer with this data that is up to date: {function_result}."""
                func_llama3 = f"""<|begin_of_text|<|start_header_id|>system<|end_header_id|>
                {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
                {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
                template = func_llama3

            if len(st.session_state.sys_prompt) > 1:
                system_message = f"""{st.session_state.sys_prompt} Chat history: {st.session_state['message_list']}"""
                func_llama3 = f"""<|begin_of_text|<|start_header_id|>system<|end_header_id|>
                {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
                {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
                template = func_llama3

        return template

class Audio:
    def melo_gen_message(message=str):
        output_path = 'melo_tts_playback.wav'
        st.session_state['melo_model'].tts_to_file(text=message, speaker_id=st.session_state['speaker_ids']['EN-AU'], output_path=output_path, speed=1.0)
        with open(output_path, "rb") as file:
            data = file.read()
        audio_base64 = base64.b64encode(data).decode('utf-8')
        audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'
        st.markdown(audio_tag, unsafe_allow_html=True)
        os.remove('melo_tts_playback.wav')

    def voice_to_text():
        speech_model_path = './speech models'
        rec_user_voice = sounddevice.rec(int(st.session_state['user_audio_length']) * 44100, samplerate=44100, channels=2)
        sounddevice.wait()
        write_wav(filename='user_output.wav', rate=44100, data=rec_user_voice)
        st.session_state['speech_tt_model'] = Model(models_dir=speech_model_path, n_threads=10)
        user_voice_data = st.session_state['speech_tt_model'].transcribe('user_output.wav', speed_up=True)
        os.remove(f"user_output.wav")
        text_data = []
        for voice in user_voice_data:        
            text_data.append(voice.text)
        combined_text = ' '.join(text_data)
        return combined_text

class SidebarConfig:
    def memory_display():
        GPUs = GPU.getGPUs()
        gpu = GPUs[0]
        mem_total = 100 / gpu.memoryTotal
        mem_used = 100 / int(gpu.memoryUsed)
        total_ = mem_total / mem_used
        if  total_ < 0.5:
            st.progress((100 / gpu.memoryTotal) / (100 / int(gpu.memoryUsed)), "gpu memory: :green[{0:.0f}/{1:.0f}gb]".format(gpu.memoryUsed, gpu.memoryTotal))
        if  total_ > 0.5:
            if total_ < 0.75:
                st.progress((100 / gpu.memoryTotal) / (100 / int(gpu.memoryUsed)), "gpu memory: :orange[{0:.0f}/{1:.0f}gb]".format(gpu.memoryUsed, gpu.memoryTotal))
        if  total_ > 0.75:
            st.progress((100 / gpu.memoryTotal) / (100 / int(gpu.memoryUsed)), "gpu memory: :red[{0:.0f}/{1:.0f}gb]".format(gpu.memoryUsed, gpu.memoryTotal))  
        memory_usage = psutil.virtual_memory()
        if memory_usage.percent < 50.0:
            st.progress((memory_usage.percent / 100), f"system memory: :green[{memory_usage.percent}%]")
        if memory_usage.percent > 50.0:
            if memory_usage.percent < 70.0:
                st.progress((memory_usage.percent / 100), f"system memory: :orange[{memory_usage.percent}%]")
        if memory_usage.percent > 70.0:
            st.progress((memory_usage.percent / 100), f"system memory: :red[{memory_usage.percent}%]")

    def display_buttons():
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
        col1, col2 = st.columns([1, 1])
        with col1:
            st.button(label=':orange[load/refresh]')
        with col2:
            if st.button(label=':orange[clear context]'):
                st.session_state['message_list'] = []
                st.session_state.messages = []
                st.session_state['model_output_tokens'] = 0

    @st.experimental_fragment
    def app_exit_button():
        exit_app = st.button(":red[restart app]", help='after clicking, close tab, visit llmon-py again')
        if exit_app:
            clear_vram()
            try:
                os.remove('.google-cookie')
            except: pass
            keyboard.press_and_release('ctrl+w')
            os.system(r"C:/Users/User/desktop/llmonpy.bat ")
            pid = os.getpid()
            p = psutil.Process(pid)
            p.terminate()
            
    st.experimental_fragment
    def select_model():
        st.session_state['model_select'] = st.selectbox(label=':orange[model]', options=st.session_state.model_list, label_visibility='collapsed')
        if st.session_state.model_picked != st.session_state['model_select']:
            try:
                del st.session_state['chat_model']
                st.session_state.model_picked = None
                st.rerun()
            except: pass
        st.session_state.model_loaded = True
        st.session_state.model_picked = st.session_state['model_select']
        if st.session_state['model_select'] == 'mistral-7b-instruct.gguf':
            st.session_state['template_select'] = st.session_state.chat_templates[1]
        if st.session_state['model_select'] == 'llama-3-8b-instruct.gguf':
            st.session_state['template_select'] = st.session_state.chat_templates[0]

    @st.experimental_fragment
    def advanced_settings():
        if st.checkbox(label=':orange[advanced settings]'):
            st.session_state['mute_melo'] = st.checkbox(':orange[mute text-to-speech]', value=st.session_state['mute_melo'])
            st.caption(body="function models")
            st.session_state.enable_moondream = st.toggle(label=':orange[moondream]', value=st.session_state.enable_moondream)
            st.session_state.enable_sdxl_turbo = st.toggle(label=':orange[sdxl turbo]', value=st.session_state.enable_sdxl_turbo)
            st.caption(body="custom prompt")
            st.session_state.sys_prompt = st.text_area(label='custom prompt', value="", label_visibility='collapsed')
            st.caption(body="model parameters")
            st.session_state['model_temperature'] = st.text_input(label=':orange[temperature]', value=st.session_state['model_temperature'])
            st.session_state['model_top_p'] = st.text_input(label=':orange[top p]', value=st.session_state['model_top_p'] )
            st.session_state['model_top_k'] = st.text_input(label=':orange[top k]', value=st.session_state['model_top_k'])
            st.session_state['model_min_p'] = st.text_input(label=':orange[min p]', value=st.session_state['model_min_p'])
            st.session_state['repeat_penalty'] = st.text_input(label=':orange[repeat_penalty]', value=st.session_state['repeat_penalty'])

class Functions:
    def youtube_download(link=str):
        helper = 'youtube'
        query = link + helper
        for j in search(query, tld="co.in", num=1, stop=1, pause=2):
            print(j)
        try:
            os.remove('.google-cookie')
        except: pass
        return j

class SDXLTurbo:
    def init():
        st.toast(body='🍋 :orange[loading sdxl turbo...]')
        st.session_state['image_pipe_turbo'] = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", variant="fp16").to('cpu')
    
    def generate_image(prompt="", steps=1):
        _image = st.session_state['image_pipe_turbo'](prompt=prompt, num_inference_steps=steps, guidance_scale=0.0).images[0]
        _image.save('image_turbo.png')

class Moondream:
    def load_vision_encoder():
        model_id = "vikhyatk/moondream2"
        revision = "2024-05-08"
        st.toast(body='🍋 :orange[loading moondream2...]')
        st.session_state['moondream'] = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision).to(device='cuda', dtype=torch.float16) 
        st.session_state.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def generate_response(prompt=str):
        filename = "ocr_upload_image.png"
        image = Image.open(filename)
        enc_image = st.session_state['moondream'].encode_image(image)
        return st.session_state['moondream'].answer_question(enc_image, prompt, st.session_state.tokenizer)