# ./codespace/llmonpy/llmonaid.py
import streamlit as st
import os
import pickle
import torch
import simpleaudio
import torchaudio
from threading import Thread, Lock
import time
import GPUtil as GPU
import psutil
import random

language = 'en'
bits_per_sample = 16
encoding_type = 'PCM_S'
warmup_chunk_size = 40
sample_rate = 44100
chunk_sample_rate = 24000
dim = 0

lock = Lock()
class AudioStream(Thread):
    def __init__(self):
        super(AudioStream, self).__init__()
        self.run_thread = True
        self.counter = 0
        self.start()

    def run(self):
        global lock
        while self.run_thread:
            try:
                wav_object = simpleaudio.WaveObject.from_wave_file(f"xtts_stream{self.counter}.wav")
                lock.acquire()
                play_audio = wav_object.play()
                play_audio.wait_done()
                os.remove(f"xtts_stream{self.counter}.wav")
                self.counter += 1
                lock.release()
            except Exception as e:  
                print("Error occurred while trying to delete the .wav file:", str(e))
                self.run_thread = False

def attempt_login(model_box_data=list, voice_box_data=list, lora_list=list, chat_templates=list):
    if st.session_state['approved_login'] == False:
        st.write("login to access llmon-py")
        username = st.text_input("username")
        password = st.text_input("password", type="password")
        login_button = st.button('sign in')
        if login_button == True and username == "chad" and password == "chad420":
            st.session_state['approved_login'] = True    
            st.session_state['user_type'] = 'admin'

        if login_button == True and username == "mikey" and password == "mikey420":
            st.session_state['approved_login'] = True    
            st.session_state['user_type'] = 'user_basic'

        if st.session_state['approved_login']:
            popup_note(message=f"you have logged in {username}!")
            init_state(model_box_data, voice_box_data, lora_list, chat_templates)
            st.rerun()
        password = ""

def get_paragraph_before_code(sentence, stop_word):
    words = sentence.split()
    result = []
    for word in words:
        if stop_word in word:
            break
        result.append(word)
    return ' '.join(result)

def stream_text(text=str):
    for word in text.split():
        yield word + " "
        time.sleep(0.08)

def trim_message_list():
    if len(st.session_state['message_list']) > 24:
        st.session_state['message_list'].pop(0)
        st.session_state['message_list'].pop(0)

def wav_by_chunk(chunks, token_count=int):
    popup_note(message='😁 generating audio...')
    wav_chunks = []
    all_chunks = []
    for i, chunk in enumerate(chunks):
        wav_chunks.append(chunk)
        all_chunks.append(chunk)
        wav = torch.cat(wav_chunks, dim=dim)
        torchaudio.save(f"xtts_stream{i}.wav", wav.squeeze().unsqueeze(0).cpu(), sample_rate=chunk_sample_rate, encoding=encoding_type, bits_per_sample=bits_per_sample)
        if token_count < 50 and i == 1:
            AudioStream()
        else:
            if i == st.session_state['chunk_pre_buffer']:
                AudioStream()
            wav_chunks = []
    full_wav = torch.cat(all_chunks, dim=dim)
    torchaudio.save(f"xtts_streamFULL_{random.randint(0, 100)}.wav", full_wav.squeeze().unsqueeze(0).cpu(), sample_rate=chunk_sample_rate, encoding=encoding_type, bits_per_sample=bits_per_sample)
    all_chunks = []

def update_current_template(user_prompt=str, template_list=list, current_template=str):
    system_message = ""
    context_list = f"""Context List: {st.session_state['message_list']}"""
    for template in template_list:
        if template == current_template:
            system_message = """You are an AI who excells at being as helpful as possible to the users request. Please check the Context List for additional context."""

        default = f"""{system_message}
        {context_list}

        USER: {user_prompt}
        ASSISTANT:"""
        current_template = default





def update_chat_template(prompt=str, template_type=str):
    template = template_type

    if template_type == "chat_default":
        chat_default = f"""You are an AI who excells at being as helpful as possible to the users request. Please check the Context List for additional context.
        Context List: {st.session_state['message_list']}

        USER: {prompt}
        ASSISTANT:"""
        template = chat_default

    if template_type == "chat_mixtral":
        chat_mixtral = f"""[INST]You are an AI who excells at being as helpful as possible to the users request. Please check the Context List for additional context. 
        Context List: {st.session_state['message_list']} 

        {prompt} [/INST]"""
        template = chat_mixtral

    if template_type == 'code_deepseek':
        code_deepseek = f"""You are an AI programming assistant, specializing in explaining Python code. Please check the Context List for additional context.
        ### Instruction:
        Context List: {st.session_state['message_list']}
        {prompt}
        ### Response:"""
        template = code_deepseek

    if template_type == 'chat_artbell':    
        chat_artbell = f"""SYSTEM: You are pretending to be Art Bell, radio host of the late night talk show Coast to Coast AM. You have a guest with you tonight (the USER), who claims to be a theoretical physicist with a remarkable story. He worked at the top-secret Black Mesa research facility in Nevada, where he witnessed and barely escaped an unimaginable disaster. Please check the Context List for additional context.
        Context List: {st.session_state['message_list']}

        USER: {prompt}
        ASSISTANT:"""
        template = chat_artbell
    
    if template_type == 'chat_redguard':
        chat_redguard = f"""USER: You are a proud Redguard hailing from the mystical realm of Elder Scrolls. A formidable and fiercely independent warrior, your blade is sworn to defend family and uphold honor. Stand ready to recount your tales, for those who dare to inquire shall hear of your valor and the legacy you forge in the sands of destiny. Please check the Context List for additional context.
        Context List: {st.session_state['message_list']}
        
        USER: {prompt}
        ASSISTANT:"""
        template = chat_redguard
    
    if template_type == 'chat_halflife':
        chat_halflife = f"""USER: You are a former scientist from the Black Mesa reseach facility named Dr. Cooper. You escaped the resonance cascade event and made it to the surface. You are here to share you stories when questioned. Please check the Context List for additional context.
        Context List: {st.session_state['message_list']}
        
        USER: {prompt}
        ASSISTANT:"""
        template = chat_halflife
    return template

def scan_dir(directory):
    directory_list = []
    count = 0
    for file in os.scandir(f'{directory}'):
        if file.is_file():
            directory_list.append(file.name)
            count += count
    return directory_list

def popup_note(message=str):
    st.toast(message)

def exclude_id(model=str):
    return {key: value for key, value in st.session_state.items() if key != model}

def clear_vram(save_current_session=False):
    
    model_list = ['vision_encoder', 'text_model', 'xtts_model', 'xtts_config', 'chat_model', 'speech_tt_model', 'image_pipe_turbo', 'image_pipe_sdxl', 'img2img_pipe']
    toggled_on_list = ['enable_voice', 'enable_microphone', 'enable_sdxl', 'enable_sdxl_turbo', 'img2img_on', 'enable_ocr', 'ocr_device']
    print('start: clear vram')

    for toggle in toggled_on_list:
        if st.session_state[toggle]:
            st.session_state[toggle] = False

    for model in model_list:
        try:
            del st.session_state[model]
            print(f'clear: {model}')
            with open('llmon-py_state.pickle', 'wb') as f:
                if save_current_session == False:
                    pickle.dump(exclude_id(model), f)
                if save_current_session:
                    pickle.dump(f)
            time.sleep(0.5)

            with open("llmon-py_state.pickle",'rb') as f:
                st.session_state = pickle.dump(f)
            os.remove('llmon-py_state.pickle')

        except:
            print(f"{model} not loaded")
            pass
    torch.cuda.empty_cache()
    st.session_state['message_list'] = []
    try:
        del st.session_state.messages
    except:
        print('error: no message state to delete')
    print ('end: clear vram')

def load_session():
    with open("llmon-py_state.pickle",'rb') as f:
        st.session_state = pickle.dump(f)

def save_session():
    clear_vram(save_current_session=st.session_state['save_session'])
    with open('llmon-py_state.pickle', 'wb') as f:
        pickle.dump(exclude_id(st.session_state), f)

def init_state(model_box_data=list, voice_box_data=list, lora_list=list, chat_template_data=list):
    default_settings_state =  {'enable_microphone': False,
                                'char_name': 'AI',
                                'enable_voice': False,
                                'enable_code_voice': False,
                                'user_audio_length': 8,
                                'voice_select': voice_box_data[0],
                                'model_select': model_box_data[0],
                                'lora_selected': lora_list[0],
                                'template_select': chat_template_data[0],
                                'verbose_chat': True,
                                'max_context': 8192,
                                'torch_audio_cores': 8,
                                'gpu_layer_count': -1,
                                'cpu_core_count': 8,
                                'cpu_batch_count': 8,
                                'batch_size': 256,
                                'stream_chunk_size': 35,
                                'chunk_pre_buffer': 4,
                                'enable_sdxl_turbo': False,
                                'img2img_on': False,
                                'enable_sdxl': False,
                                'enable_ocr': False,
                                'ocr_device': False}
        
    st.session_state['token_count'] = ""
    st.session_state['response_time'] = ""
    st.session_state['model_output_tokens'] = ""

    st.session_state['sdxl_image_list'] = []
    st.session_state['sdxl_steps'] = 50
    st.session_state['sdxl_iter_count'] = 1
    st.session_state['sdxl_prompt'] = ""

    st.session_state['turbo_prompt'] = ""
    st.session_state['sdxl_turbo_steps'] = 1

    st.session_state['img2img_prompt'] = ""
    st.session_state['img2img_steps'] = 2
    st.session_state['img2img_iter_count'] = 1

    st.session_state['model_temperature'] = 0.85
    st.session_state['model_top_p'] = 0.95
    st.session_state['model_top_k'] = 0
    st.session_state['model_min_p'] = 0.08
    st.session_state['repeat_penalty'] = 1.1

    st.session_state['message_list'] = []

    for key, value in default_settings_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

def memory_display():
    GPUs = GPU.getGPUs()
    gpu = GPUs[0]
    mem_total = 100 / gpu.memoryTotal
    mem_used = 100 / int(gpu.memoryUsed)
    total_ = mem_total / mem_used
    if  total_ < 0.5:
        st.progress((100 / gpu.memoryTotal) / (100 / int(gpu.memoryUsed)), "vram :green[{0:.0f}]/{1:.0f}gb".format(gpu.memoryUsed, gpu.memoryTotal))
    if  total_ > 0.5:
        if total_ < 0.75:
            st.progress((100 / gpu.memoryTotal) / (100 / int(gpu.memoryUsed)), "vram :orange[{0:.0f}]/{1:.0f}gb".format(gpu.memoryUsed, gpu.memoryTotal))
    if  total_ > 0.75:
        st.progress((100 / gpu.memoryTotal) / (100 / int(gpu.memoryUsed)), "vram :red[{0:.0f}]/{1:.0f}gb".format(gpu.memoryUsed, gpu.memoryTotal))
        
    memory_usage = psutil.virtual_memory()
    if memory_usage.percent < 50.0:
        st.progress((memory_usage.percent / 100), f'system ram usage :green[{memory_usage.percent}%]')
    if memory_usage.percent > 50.0:
        if memory_usage.percent < 75.0:
            st.progress((memory_usage.percent / 100), f'system ram usage :orange[{memory_usage.percent}%]')
    if memory_usage.percent > 75.0:
        st.progress((memory_usage.percent / 100), f'system ram usage :red[{memory_usage.percent}%]')

def text_thread(text=str):
    st.write_stream(stream=stream_text(text))

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def message_boop():
    if st.session_state['user_type'] == 'admin':
        message_boop = simpleaudio.WaveObject.from_wave_file("./llmonpy/chat_pop.wav")
        message_boop.play()

def clear_buffers():
    st.session_state.bytes_data = None

def check_user_type():
    disable_option = False
    if st.session_state['user_type'] == 'user_basic':
        disable_option = True
    return disable_option