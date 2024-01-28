# ./codespace/llmonpy/llmonaid.py
import streamlit as st
import os
import pickle
import torch
import warnings
import simpleaudio

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
    model_list = ['vision_encoder', 'text_model', 'model', 'config', 'chat_model', 'speech_tt_model']
    for model in model_list:
        try:
            del st.session_state[model]
            with open('llmon-py_state.pickle', 'wb') as f:
                pickle.dump(exclude_id(model), f)
        except:
            pass
    torch.cuda.empty_cache()
    try:
        with open("llmon-py_state.pickle",'rb') as f:
            st.session_state = pickle.dump(f)
    except:
        pass
    os.remove('llmon-py_state.pickle')

def init_default_state(model_box_data=list, voice_box_data=list, chat_template_data=list):
    default_settings_state = {'enable_message_beep': True,
                              'enable_microphone': False,
                              'enable_voice': False,
                              'tts_cpu': False,
                              'enable_code_voice': False,
                              'user_audio_length': 8,
                              'voice_select': voice_box_data[0],
                              'model_select': model_box_data[0],
                              'template_select': chat_template_data[0],
                              'verbose_chat': False,
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
                              'turbo_cpu': False,
                              'enable_sdxl': False,
                              'sdxl_cpu': False,
                              'ocr_device': False}

    for key, value in default_settings_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

def clear_console():
    os.system('cls')
    warnings.filterwarnings('ignore')

def message_boop():
    message_boop = simpleaudio.WaveObject.from_wave_file("./llmonpy/chat_pop.wav")
    message_boop.play()

def clear_buffers():
    del st.session_state.bytes_data
    st.session_state.buffer = ""