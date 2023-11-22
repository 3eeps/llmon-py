import os
import time
import warnings
import streamlit as st
from threading import Thread

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts 
from llama_cpp import Llama
from pywhispercpp.model import Model

import torchaudio
import simpleaudio
import sounddevice 
from scipy.io.wavfile import write

def scan_dir(scan_type):
    if scan_type == 'models':
        model_directory = os.scandir('models')
        model_list = []
        model_count = 0
        for model_file in model_directory:
            if model_file.is_file():
                model_list.append(model_file.name)
            model_count = model_count + 1
        model_directory.close()
        return model_list
    
    elif scan_type == 'voices':
        voice_directory = os.scandir('voices')
        voice_list = []
        voice_count = 0
        for voice_file in voice_directory:
            if voice_file.is_file():
                voice_list.append(voice_file.name)
            voice_count = voice_count + 1
        voice_directory.close()
        return voice_list

if 'settings' not in st.session_state:

    llmon_config = st.form('setup')

    model_box_data = scan_dir('models')
    model_select = st.selectbox('model file', [model_box_data[0], model_box_data[1], model_box_data[2], model_box_data[3], model_box_data[4], model_box_data[5], model_box_data[6]])

    character_name = st.selectbox('character name', ['Dr. Rosenburg', 'Cortana', 'Kyle Katarn', 'Art Bell', 'Bot', 'Assistant', 'AI', 'Model'])
    voice_box_data = scan_dir('voices')
    voice_select = st.selectbox('voice file', [voice_box_data[0], voice_box_data[1], voice_box_data[2], voice_box_data[3], voice_box_data[4], voice_box_data[5], voice_box_data[6], voice_box_data[7], voice_box_data[8]])

    template_select = st.selectbox('chat template', ['ajibawa_python', 'instruction', 'user_assist_art', 'user_assist_kyle', 'user_assist_hlsci', 'vicuna'])
    enable_code_voice = st.selectbox('coding mode', ['no', 'yes'])

    submit_config_button = llmon_config.form_submit_button('save')

    st.session_state.model_select = model_select
    st.session_state.char_name = character_name
    st.session_state.voice_select = voice_select
    st.session_state.template_select = template_select
    st.session_state.enable_code_voice = enable_code_voice
    st.session_state.config_set = True
