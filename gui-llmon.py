# ./codespace/llmon-gui.py

import os
import time
import warnings
import streamlit as st
from threading import Thread
from nltk.tokenize import sent_tokenize

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts 
from llama_cpp import Llama
from pywhispercpp.model import Model

import torchaudio
import simpleaudio
import sounddevice 
from scipy.io.wavfile import write

st.title("llmon-py - python-code-13b.Q6_K.gguf")
warnings.filterwarnings("ignore")

code_model = True
char_name = "Cortana"
rec_seconds = 8

if "messages" not in st.session_state:
    st.session_state.messages = []

if 'config' not in st.session_state:
    st.session_state.config = XttsConfig()
    st.session_state.config.load_json("./xtts_config/config.json")
    st.session_state.model = Xtts.init_from_config(st.session_state.config)
    st.session_state.model.load_checkpoint(st.session_state.config, checkpoint_dir="./xtts_config", use_deepspeed=False)
    st.session_state.model.cuda()

gpt_cond_latent, speaker_embedding = st.session_state.model.get_conditioning_latents(audio_path=["./voices/cortana.wav"])

if 'chat_model' not in st.session_state:
    st.session_state.chat_model = Llama(model_path="models/python-code-13b.Q6_K.gguf",  n_batch=1024, n_gpu_layers = 32, n_ctx = 4096, verbose = False)

if 'speech_tt_model' not in st.session_state:
    st.session_state.speech_tt_model = Model(models_dir="models")

class AudioThread(Thread):
    def __init__(self):
        super(AudioThread, self).__init__()
        self.stop_thread = False
        self.start()

    def run(self):
        while not self.stop_thread:
            wav_object = simpleaudio.WaveObject.from_wave_file('model_output.wav')
            play_audio = wav_object.play()
            play_audio.wait_done()
            self.stop_thread = True

class AudioStream(Thread):
    def __init__(self):
        super(AudioStream, self).__init__()
        self.stopped = False
        self.count = 0
        self.start()

    def run(self):
        time.sleep(2.0)
        while not self.stopped:
            try:
                wav_object = simpleaudio.WaveObject.from_wave_file(f"xtts_stream{self.count}.wav")
                play_audio = wav_object.play()
                play_audio.wait_done()
                self.count = self.count + 1 
            except:
                self.stopped = True

def llmon():
    # when life gives you lemons, you paint that shit gold
    # https://emojicombos.com/lemon-ascii-art
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

    ajibawa_python = f"""This is a conversation with your helpful AI assistant. AI assistant can generate Python Code along with necessary explanation.

    Context
    You are a helpful AI assistant who excels at teaching python.

    USER: {prompt}
    ASSISTANT:"""
    instruction = f"""### Instruction: 
    none

    USER: {prompt}
    ### Response:"""
        
    user_assist_art = f"""USER: You are Art Bell, the radio host from the late-night talk show, Coast to Coast AM. Your guest tonight claims to be a theoretical physicist with a remarkable story. He claims to have worked at the top-secret Black Mesa research facility, where he witnessed an unimaginable disaster.

    GUEST: {prompt}
    ASSISTANT:"""

    user_assist_kyle = f"""USER: You are Kyle Katarn from the Star Wars universe. As someone always battling and out running Imperial Forces, you have many stories to share. You sit at a bar in Nar Shaddaa with a close friend. It feels familiar here, like home.

    USER: {prompt}
    ASSISTANT:"""

    user_assist_hlsci = f"""USER: You are a former scientist from the Black Mesa reseach facility. You escaped the resonance cascade event and made it to the surface. You are here to share you stories when questioned.

    USER: {prompt}
    ASSISTANT:"""

    vicuna = f"""none

    User: {prompt}
    ASSISTANT:"""

    template_type = ajibawa_python
    return template_type

def wav_by_chunk(chunks):
    wav_chunks = []
    stream_full = []
    for i, chunk in enumerate(chunks):
        wav_chunks.append(chunk)
        stream_full.append(chunk)
        wav = torch.cat(wav_chunks, dim=0)
        torchaudio.save(f"xtts_stream{i}.wav", wav.squeeze().unsqueeze(0).cpu(), sample_rate=24000, encoding="PCM_S", bits_per_sample=16)
        #0 is fastest to hearing first chunk
        if i == 0:
            play_chunks = AudioStream()
        wav_chunks = []

    wav = torch.cat(stream_full, dim=0)
    torchaudio.save("xtts_stream_full.wav", wav.squeeze().unsqueeze(0).cpu(), sample_rate=24000, encoding="PCM_S", bits_per_sample=16)

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
    st.session_state.messages.append({"role": "user", "content": f'User: {user_prompt}'})

    model_output = st.session_state.chat_model(prompt=prompt)
    model_response = f"{char_name}: {model_output['choices'][0]['text']}"
    print(model_output)    
    with st.chat_message("assistant"):
        st.markdown(model_response)
    st.session_state.messages.append({"role": "assistant", "content": model_response})
        
    if code_model:
        sentence_list = []
        sentence_list = sent_tokenize(model_output['choices'][0]['text'])
        print(f'sent to tts: {sentence_list[0]}')
        first_sentence_only = st.session_state.model.inference_stream(sentence_list[0], "en", gpt_cond_latent, speaker_embedding)
        wav_by_chunk(first_sentence_only)
    else:
        chunk_inference = st.session_state.model.inference_stream(model_output['choices'][0]['text'], "en", gpt_cond_latent, speaker_embedding, stream_chunk_size=80)
        wav_by_chunk(chunk_inference)
