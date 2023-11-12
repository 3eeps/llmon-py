# ./codespace/llmon-gui.py

import os
import warnings
import streamlit as st

from TTS.api import TTS 
from llama_cpp import Llama
from pywhispercpp.model import Model

import simpleaudio
import sounddevice 
from scipy.io.wavfile import write

warnings.filterwarnings("ignore")
st.title("llmon-py")

char_name = "Kyle Katarn"
rec_seconds = 8

if "messages" not in st.session_state:
    st.session_state.messages = []

if 'text_ts_model' not in st.session_state:
    st.session_state.text_ts_model = text_ts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to('cuda')

if 'chat_model' not in st.session_state:
    st.session_state.chat_model = Llama(model_path="models/airoboros-mistral2.2-7b.Q6_K.gguf",  n_gpu_layers = -1, n_ctx = 4096, verbose = False)

if 'speech_tt_model' not in st.session_state:
    st.session_state.speech_tt_model = Model(models_dir="models")

def llmon():
    # when life gives you lemons, you paint that shit gold
    # thanks https://emojicombos.com/lemon-ascii-art, for the lemon
    os.system("cls")
    color_logo = f"\33[{93}m".format(code=93)
    print(f"""{color_logo}
                        llmon-py
⠀⠀⠀⠀⠀⢀⣀⣠⣤⣴⣶⡶⢿⣿⣿⣿⠿⠿⠿⠿⠟⠛⢋⣁⣤⡴⠂⣠⡆⠀
⠀⠀⠀⠀⠈⠙⠻⢿⣿⣿⣿⣶⣤⣤⣤⣤⣤⣴⣶⣶⣿⣿⣿⡿⠋⣠⣾⣿⠁⠀
⠀⠀⠀⠀⠀⢀⣴⣤⣄⡉⠛⠻⠿⠿⣿⣿⣿⣿⡿⠿⠟⠋⣁⣤⣾⣿⣿⣿⠀⠀
⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣶⣶⣤⣤⣤⣤⣤⣤⣶⣾⣿⣿⣿⣿⣿⣿⣿⡇⠀
⠀⠀⠀⣰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀
⠀⠀⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⠀
⠀⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⢸⡟⢸⡟⠀⠀
⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣷⡿⢿⡿⠁⠀⠀
⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⢁⣴⠟⢀⣾⠃⠀⠀⠀
⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠛⣉⣿⠿⣿⣶⡟⠁⠀⠀⠀⠀
⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⣿⣏⣸⡿⢿⣯⣠⣴⠿⠋⠀⠀⠀⠀⠀⠀
⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⠿⠶⣾⣿⣉⣡⣤⣿⠿⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⢸⣿⣿⣿⣿⡿⠿⠿⠿⠶⠾⠛⠛⠛⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠈⠉⠉⠉""")

def update_chat_template(prompt=str):
    instruction = f"""### Instruction: 
    none

    USER: {prompt}
    ### Response:"""
        
    user_assist_art = f"""USER: You are Art Bell, the radio host from the late-night talk show, Coast to Coast AM. Your guest tonight claims to be a theoretical physicist with a remarkable story. He claims to have worked at the top-secret Black Mesa research facility, where he witnessed an unimaginable disaster.

    GUEST: {prompt}
    ASSISTANT:"""

    user_assist_kyle = f"""USER: You are Kyle Katarn from the Star Wars universe. As someone always battling and out running Imperial Forces with your sidekick Jan Ors and her ship The Crow, you have many stories to share.

    USER: {prompt}
    ASSISTANT:"""

    user_assist_hlsci = f"""USER: You are a former scientist from the Black Mesa reseach facility. You escaped the resonance cascade event and made it to the surface. You are here to share you stories when questioned.

    USER: {prompt}
    ASSISTANT:"""

    vicuna = f"""none

    User: {prompt}
    ASSISTANT:"""

    template_type = user_assist_kyle
    return template_type

def create_chat_wav(chat_model_text=str):
    character_wav = "voices/kyle.wav"
    st.session_state.text_ts_model.tts_to_file(text=chat_model_text, speaker_wav=character_wav, file_path='model_output.wav', language="en")

def play_wav():
    wav_object = simpleaudio.WaveObject.from_wave_file('model_output.wav')
    play_audio = wav_object.play()
    play_audio.wait_done()

def voice_to_text():
    rec_user_voice = sounddevice.rec(int(rec_seconds * 44100), samplerate=44100, channels=2)
    sounddevice.wait()
    write(filename='user_output.wav', rate=44100, data=rec_user_voice)

    text_data = []
    user_voice_data = st.session_state.speech_tt_model.transcribe('user_output.wav')
    for voice in user_voice_data:        
        text_data.append(voice.text)
    combined_text = ' '.join(text_data)
    return combined_text

llmon()
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input(f"Send a message to {char_name}"):
    if user_prompt == 'voice':
        user_prompt = voice_to_text()
        prompt = update_chat_template(user_prompt)
    else:
        prompt = update_chat_template(user_prompt)
    
    with st.chat_message("User"):
        st.markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    model_output = st.session_state.chat_model(prompt=prompt)
    model_response = f"{char_name}: {model_output['choices'][0]['text']}"    
    with st.chat_message("assistant"):
        st.markdown(model_response)
    st.session_state.messages.append({"role": "assistant", "content": model_response})

    create_chat_wav(chat_model_text=model_output['choices'][0]['text'])
    play_wav()
