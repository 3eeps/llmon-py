# ./codespace/llmon-gui.py

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

st.title("🍋llmon-py")
warnings.filterwarnings("ignore")

code_model = st.session_state.enable_code_voice
chat_verbose = False
chat_max_context = 4096
chat_threads = 12
chat_batch_threads = 12
torch_threads = 8
sample_rate = 44100
chunk_sample_rate = 24000
channels = 2
max_prompt_context = 1536
bits_per_sample = 16
encoding_type = 'PCM_S'
model_path = f"./models/{st.session_state.model_select}"
voice_path = f"./voices/{st.session_state.voice_select}"
speech_model_path = 'models'

gpu_layers = -1
char_name = st.session_state.char_name
rec_seconds = 8
torch.set_num_threads(torch_threads)

if "messages" not in st.session_state:
    st.session_state.messages = []

if 'config' not in st.session_state:
    st.session_state.config = XttsConfig()
    st.session_state.config.load_json("./xtts_config/config.json")
    st.session_state.model = Xtts.init_from_config(st.session_state.config)
    st.session_state.model.load_checkpoint(st.session_state.config, checkpoint_dir="./xtts_config")
    st.session_state.model.cuda()

gpt_cond_latent, speaker_embedding = st.session_state.model.get_conditioning_latents(audio_path=[f"{voice_path}"])

if 'chat_model' not in st.session_state:
    st.session_state.chat_model = Llama(model_path=model_path, n_threads=chat_threads, n_threads_batch=chat_batch_threads, n_gpu_layers = gpu_layers, n_ctx = chat_max_context, verbose = chat_verbose)

if 'speech_tt_model' not in st.session_state:
    st.session_state.speech_tt_model = Model(models_dir=speech_model_path)

class AudioThread(Thread):
    def __init__(self):
        super(AudioThread, self).__init__()
        self.stop_thread = False
        self.start()

    def run(self):
        while not self.stop_thread:
            wav_object = simpleaudio.WaveObject.from_wave_file('input.wav')
            play_audio = wav_object.play()
            play_audio.wait_done()
            self.stop_thread = True

class AudioStream(Thread):
    def __init__(self):
        super(AudioStream, self).__init__()
        self.stop_thread = False
        self.iter = 0
        self.start()

    def run(self):
        time.sleep(1.0)
        while not self.stop_thread:
            try:
                wav_object = simpleaudio.WaveObject.from_wave_file(f"xtts_stream{self.iter}.wav")
                play_audio = wav_object.play()
                play_audio.wait_done()
                os.remove(f"xtts_stream{self.iter}.wav")
                self.iter = self.iter + 1 
            except:
                self.stop_thread = True

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

    deepseek = f"""You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
    ### Instruction:
    {prompt}
    ### Response:"""

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

    user_assist_redguard = f"""USER: Embark on the epic journey as a proud Redguard hailing from the mystical realm of Elder Scrolls. A formidable and fiercely independent warrior, your blade is sworn to defend family and uphold honor. Stand ready to recount your tales, for those who dare to inquire shall hear of your valor and the legacy you forge in the sands of destiny.

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
    stream_chunks = []
    for i, chunk in enumerate(chunks):
        wav_chunks.append(chunk)
        stream_chunks.append(chunk)
        wav = torch.cat(wav_chunks, dim=0)
        torchaudio.save(f"xtts_stream{i}.wav", wav.squeeze().unsqueeze(0).cpu(), sample_rate=chunk_sample_rate, encoding=encoding_type, bits_per_sample=bits_per_sample)
        if i == 1:
            AudioStream()
        wav_chunks = []
    stream_wav = torch.cat(stream_chunks, dim=0)
    torchaudio.save("xtts_stream_full.wav", stream_wav.squeeze().unsqueeze(0).cpu(), sample_rate=sample_rate, encoding=encoding_type, bits_per_sample=bits_per_sample)

def voice_to_text():
    rec_user_voice = sounddevice.rec(int(rec_seconds * sample_rate), samplerate=sample_rate, channels=channels)
    sounddevice.wait()
    write(filename='user_output.wav', rate=sample_rate, data=rec_user_voice)

    text_data = []
    user_voice_data = st.session_state.speech_tt_model.transcribe('user_output.wav')
    for voice in user_voice_data:        
        text_data.append(voice.text)
    combined_text = ' '.join(text_data)
    return combined_text

def extract_words_before_set(sentence, target_set):
    words = sentence.split()
    result = []
    for word in words:
        if target_set in word:
            break
        result.append(word)

    return ' '.join(result)

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

    model_output = st.session_state.chat_model(prompt=prompt, max_tokens=max_prompt_context)
    model_response = f"{char_name}: {model_output['choices'][0]['text']}"
    print(model_output)    
    with st.chat_message("assistant"):
        st.markdown(model_response)
    st.session_state.messages.append({"role": "assistant", "content": model_response})
        
    if code_model == 'yes':
        first_paragraph = extract_words_before_set(sentence=model_output['choices'][0]['text'], target_set='```')
        send_only_paragraph = st.session_state.model.inference_stream(first_paragraph, "en", gpt_cond_latent, speaker_embedding, stream_chunk_size=40)
        wav_by_chunk(send_only_paragraph)
    else:
        chunk_inference = st.session_state.model.inference_stream(model_output['choices'][0]['text'], "en", gpt_cond_latent, speaker_embedding, stream_chunk_size=60)
        wav_by_chunk(chunk_inference)
