# ./codespace/llmonpy/llmonaid.py
import streamlit as st
import os
import pickle
import torch
import warnings
import simpleaudio
import torchaudio
from threading import Thread
import time

language = 'en'
bits_per_sample = 16
encoding_type = 'PCM_S'
warmup_chunk_size = 40
sample_rate = 44100
chunk_sample_rate = 24000
dim = 0

class AudioStream(Thread):
    def __init__(self):
        super(AudioStream, self).__init__()
        self.run_thread = True
        self.counter = 0
        self.start()
        self.wav_object = simpleaudio.WaveObject.from_wave_file(f"xtts_stream{self.counter}.wav")

    def run(self):
        while self.run_thread:
            try:
                wav_object = simpleaudio.WaveObject.from_wave_file(f"xtts_stream{self.counter}.wav")
                play_audio = wav_object.play()
                play_audio.wait_done()
                os.remove(f"xtts_stream{self.counter}.wav")
            except:
                pass#self.run_thread = False
            
            self.counter += 1

def get_paragraph_before_code(sentence, stop_word):
    words = sentence.split()
    result = []
    for word in words:
        if stop_word in word:
            break
        result.append(word)
    return ' '.join(result)

def wav_by_chunk(chunks, token_count=int):
    popup_note(message='😁 generating audio...')
    wav_chunks = []
    #all_chunks = []
    for i, chunk in enumerate(chunks):
        wav_chunks.append(chunk)
        #all_chunks.append(chunk)
        wav = torch.cat(wav_chunks, dim=dim)
        torchaudio.save(f"xtts_stream{i}.wav", wav.squeeze().unsqueeze(0).cpu(), sample_rate=chunk_sample_rate, encoding=encoding_type, bits_per_sample=bits_per_sample)
        if i == st.session_state['chunk_pre_buffer']:
            AudioStream()
        wav_chunks = []
    #full_wav = torch.cat(all_chunks, dim=dim)
    #torchaudio.save("xtts_stream_full.wav", full_wav.squeeze().unsqueeze(0).cpu(), sample_rate=chunk_sample_rate, encoding=encoding_type, bits_per_sample=bits_per_sample)
    #all_chunks = []

def update_chat_template(prompt=str, template_type=str):
    template = template_type

    if template_type == "default":
        vicuna_default = f"""You are an AI who excells at being as helpful as possible to the users request.

        USER: {prompt}
        ASSISTANT:"""
        template = vicuna_default

    if template_type == 'deepseek':
        deepseek = f"""You are an AI programming assistant, specializing in explaining Python code. You also have experience in using the Streamlit library.
        ### Instruction:
        {prompt}
        ### Response:"""
        template = deepseek

    if template_type == 'user_assist_duke':
        user_assist_duke = f"""USER: Imagine you're Duke Nukem, the badass one-liner-spouting hero from the video game series Duke Nukem 3D. You're sitting down with the USER to have a conversation. 

        USER: {prompt}
        ASSISTANT:"""
        template = user_assist_duke

    if template_type == 'user_assist_rick':
        user_assist_rick = f"""USER: Imagine you're the genius, eccentric, and slightly cynical scientist Rick from Rick and Morty. You have just jumped through a portal to sit down and have a conversation about what you have been up to in your garage.

        USER: {prompt}
        ASSISTANT:"""
        template = user_assist_rick

    if template_type == 'ajibawa_python':
        ajibawa_python = f"""This is a conversation with your helpful AI assistant. AI assistant can generate Python Code along with necessary explanation.

        Context
        You are a helpful AI assistant who excels at teaching python.

        USER: {prompt}
        ASSISTANT:"""
        template = ajibawa_python

    if template_type == 'user_assist_art':    
        user_assist_art = f"""SYSTEM: You are pretending to be Art Bell, radio host of the late night talk show Coast to Coast AM. You have a guest with you tonight (the USER), who claims to be a theoretical physicist with a remarkable story. He worked at the top-secret Black Mesa research facility in Nevada, where he witnessed and barely escaped an unimaginable disaster.

        USER: {prompt}
        ASSISTANT:"""
        template = user_assist_art

    if template_type == 'user_assist_kyle':
        user_assist_kyle = f"""USER: You are Kyle Katarn from the Star Wars universe. As someone always battling and out running Imperial Forces, you have many stories to share. You sit at a bar in Nar Shaddaa with a close friend. It feels familiar here, like home.

        USER: {prompt}
        ASSISTANT:"""
        template = user_assist_kyle
    
    if template_type == 'user_assist_redguard':
        user_assist_redguard = f"""USER: You are a proud Redguard hailing from the mystical realm of Elder Scrolls. A formidable and fiercely independent warrior, your blade is sworn to defend family and uphold honor. Stand ready to recount your tales, for those who dare to inquire shall hear of your valor and the legacy you forge in the sands of destiny.

        USER: {prompt}
        ASSISTANT:"""
        template = user_assist_redguard
    
    if template_type == 'user_assist_hlsci':
        user_assist_hlsci = f"""USER: You are a former scientist from the Black Mesa reseach facility. You escaped the resonance cascade event and made it to the surface. You are here to share you stories when questioned.

        USER: {prompt}
        ASSISTANT:"""
        template = user_assist_hlsci

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

def clear_vram():
    model_list = ['vision_encoder', 'text_model', 'model', 'config', 'chat_model', 'speech_tt_model', 'image_pipe_turbo']
    for model in model_list:
        print('start: clear vram')
        try:
            del st.session_state[model]
            with open('llmon-py_state.pickle', 'wb') as f:
                pickle.dump(exclude_id(model), f)
        except:
            pass
    try:
        with open("llmon-py_state.pickle",'rb') as f:
            st.session_state = pickle.dump(f)
    except:
        pass
    try:
        os.remove('llmon-py_state.pickle')
    except:
        pass
    torch.cuda.empty_cache()
    print ('end: clear vram: done')

def init_state(model_box_data=list, voice_box_data=list, chat_template_data=list, user_type=str):
    if user_type == 'admin':
        default_settings_state =  {'enable_microphone': False,
                                    'enable_voice': False,
                                    'enable_code_voice': False,
                                    'user_audio_length': 8,
                                    'voice_select': voice_box_data[0],
                                    'model_select': model_box_data[0],
                                    'template_select': chat_template_data[0],
                                    'verbose_chat': True,
                                    'max_context_prompt': 2048,
                                    'max_context': 4096,
                                    'torch_audio_cores': 8,
                                    'gpu_layer_count': -1,
                                    'cpu_core_count': 8,
                                    'cpu_batch_count': 8,
                                    'batch_size': 256,
                                    'stream_chunk_size': 35,
                                    'chunk_pre_buffer': 3,
                                    'enable_sdxl_turbo': False,
                                    'img2img_on': False,
                                    'enable_sdxl': False,
                                    'enable_ocr': False,
                                    'ocr_device': False}
    if user_type == 'user_basic':
        default_settings_state =  {'enable_microphone': False,
                                    'enable_voice': False,
                                    'enable_code_voice': False,
                                    'user_audio_length': 8,
                                    'voice_select': voice_box_data[0],
                                    'model_select': model_box_data[0],
                                    'template_select': chat_template_data[0],
                                    'verbose_chat': True,
                                    'max_context_prompt': 1024,
                                    'max_context': 4096,
                                    'torch_audio_cores': 8,
                                    'gpu_layer_count': -1,
                                    'cpu_core_count': 8,
                                    'cpu_batch_count': 8,
                                    'batch_size': 256,
                                    'stream_chunk_size': 35,
                                    'chunk_pre_buffer': 3,
                                    'enable_sdxl_turbo': False,
                                    'img2img_on': False,
                                    'enable_sdxl': False,
                                    'enable_ocr': False,
                                    'ocr_device': False}
        
    for key, value in default_settings_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

def clear_console():
    os.system('cls')
    warnings.filterwarnings('ignore')

def message_boop():
    if st.session_state['user_type'] == 'admin':
        message_boop = simpleaudio.WaveObject.from_wave_file("./llmonpy/chat_pop.wav")
        message_boop.play()

def clear_buffers():
    st.session_state.bytes_data = None
    st.session_state.buffer = ""

def check_user_type():
    disable_option = False
    if st.session_state['user_type'] == 'user_basic':
        disable_option = True
    return disable_option