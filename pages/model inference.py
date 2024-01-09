# ./codespace/pages/model inference.py

import streamlit as st
from streamlit_extras.app_logo import add_logo 
import keyboard

keyboard.unhook_all()
st.set_page_config(page_title="model inference", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("llmon-py - model inference")
add_logo("./llmon_art/lemon (17).png")
st.divider()

import os
import time
import pickle
import GPUtil as GPU
import warnings
from threading import Thread
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts 
from llama_cpp import Llama
from pywhispercpp.model import Model
import torchaudio
import simpleaudio
import sounddevice 
from scipy.io.wavfile import write as write_wav

chat_model_path = f"./models/{st.session_state.model_select}"
chat_model_voice_path = f"./voices/{st.session_state.voice_select}"
char_name = st.session_state.char_name
code_model_voice = st.session_state.enable_code_voice
current_template = st.session_state.template_select
enable_voice = st.session_state.enable_voice
enable_popups = st.session_state.enable_popups
verbose_chat = st.session_state.verbose_chat
chat_max_context = st.session_state.max_context
chat_threads = st.session_state.cpu_core_count
chat_batch_threads = st.session_state.cpu_batch_count
max_context_prompt = st.session_state.max_context_prompt
batch_size = st.session_state.batch_size
gpu_layers = st.session_state.gpu_layer_count
torch_audio_threads = st.session_state.torch_audio_cores
rec_seconds = st.session_state.user_audio_length
chunk_pre_buffer = st.session_state.chunk_pre_buffer
stream_chunk_size = st.session_state.stream_chunk_size
language = st.session_state.model_language
enable_console_warnings = st.session_state.console_warnings
enable_microphone = st.session_state.enable_microphone
boop_sound = st.session_state.boop_select

os.system('cls')
torch.set_num_threads(torch_audio_threads)
warnings.filterwarnings(enable_console_warnings)
GPUs = GPU.getGPUs()
gpu = GPUs[0]

reveal_logits = True
log_probs = 5

speech_model_path = './pywhisper_models'
warmup_string = 'warmup string'
bits_per_sample = 16
encoding_type = 'PCM_S'
code_stream_chunk_size = 40
warmup_chunk_size = 20
sample_rate = 44100
chunk_sample_rate = 24000
xtts_token_limit = 400
channels = 2
dim = 0
popup_delay = 1.0

def popup_note(message=str):
    if enable_popups:
        st.toast(message)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_voice_prompt" not in st.session_state:
    st.session_state.user_voice_prompt = 'none'

if 'model' not in st.session_state and enable_voice:
    popup_note(message='😤 hyping up tts model...')
    st.session_state.config = XttsConfig()
    st.session_state.config.load_json("./xtts_config/config.json")
    st.session_state.model = Xtts.init_from_config(st.session_state.config)
    st.session_state.model.load_checkpoint(st.session_state.config, checkpoint_dir="./xtts_config")
    if st.session_state.audio_cuda_or_cpu == 'cuda':
        st.session_state.model.cuda()
    
#if 'summarizer' not in st.session_state:
#    popup_note(message='😎 its game time summarizer!')
#    st.session_state.summarizer = pipeline("summarization", model="./summarizer/t5-small")

if 'speech_tt_model' not in st.session_state and enable_microphone:
    popup_note(message='😎 lets get it stt model!')
    st.session_state.user_voice_prompt = str
    st.session_state.speech_tt_model = Model(models_dir=speech_model_path)

if enable_voice:
    gpt_cond_latent, speaker_embedding = st.session_state.model.get_conditioning_latents(audio_path=[f"{chat_model_voice_path}"])
    st.session_state.model.inference_stream(text=warmup_string, language=language, gpt_cond_latent=gpt_cond_latent, speaker_embedding=speaker_embedding, stream_chunk_size=warmup_chunk_size)

if 'chat_model' not in st.session_state:
    popup_note(message='😴 waking up chat model...')
    logits_list = str
    st.session_state.chat_model = Llama(model_path=chat_model_path, logits_all=reveal_logits, n_batch=batch_size, n_threads=chat_threads, n_threads_batch=chat_batch_threads, n_gpu_layers=gpu_layers, n_ctx=chat_max_context, verbose=verbose_chat)

class AudioStream(Thread):
    def __init__(self):
        super(AudioStream, self).__init__()
        self.stop_thread = False
        self.iter = 0
        self.start()

    def run(self):
        while not self.stop_thread:
            try:    
                wav_object = simpleaudio.WaveObject.from_wave_file(f"xtts_stream{self.iter}.wav")
                play_audio = wav_object.play()
                play_audio.wait_done()
                os.remove(f"xtts_stream{self.iter}.wav")
                self.iter = self.iter + 1
            except:
                self.iter = self.iter + 1
                self.stop_thread = True
                    
def update_chat_template(prompt=str, template_type=str):
    template = template_type

    if template_type == "vicuna_based":
        vicuna_based = f"""You are an AI who excells at being as helpful as possible to the users request.

        USER: {prompt}
        ASSISTANT:"""
        template = vicuna_based

    if template_type == 'deepseek':
        deepseek = f"""You are an AI programming assistant, specializing in explaining Python code. You also have experience in using the Streamlit library.
        ### Instruction:
        {prompt}
        ### Response:"""
        template = deepseek

    if template_type == 'user_assist_duke':
        user_assist_duke = f"""USER: Imagine you're Duke Nukem, the badass one-liner-spouting hero from the video game series Duke Nukem 3D. You're sitting down with the USER to have a conversation. 
        Example dialogue: "Damn, those alien bastards are gonna pay for shooting up my ride.", "It's time to kick ass and chew bubble gum... and I'm all outta gum."

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
        user_assist_art = f"""USER: You are Art Bell, the radio host from the late-night talk show, Coast to Coast AM. Your guest tonight claims to be a theoretical physicist with a remarkable story. He claims to have worked at the top-secret Black Mesa research facility, where he witnessed an unimaginable disaster.

        GUEST: {prompt}
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

def wav_by_chunk(chunks):
    popup_note(message='😁 generating audio...')
    wav_chunks = []
    stream_chunks = []
    for i, chunk in enumerate(chunks):
        wav_chunks.append(chunk)
        stream_chunks.append(chunk)
        wav = torch.cat(wav_chunks, dim=dim)
        torchaudio.save(f"xtts_stream{i}.wav", wav.squeeze().unsqueeze(0).cpu(), sample_rate=chunk_sample_rate, encoding=encoding_type, bits_per_sample=bits_per_sample)
        if i == chunk_pre_buffer:
            time.sleep(2.0)
            AudioStream()
        wav_chunks = []
    #if no_audio_stream == False:
    #    stream_wav = torch.cat(stream_chunks, dim=dim)
    #    torchaudio.save("xtts_stream_full.wav", stream_wav.squeeze().unsqueeze(0).cpu(), sample_rate=chunk_sample_rate, encoding=encoding_type, bits_per_sample=bits_per_sample)
    #    wav_object = simpleaudio.WaveObject.from_wave_file("xtts_stream_full.wav")
    #    wav_object.play()

def voice_to_text():
    rec_user_voice = sounddevice.rec(int(rec_seconds * sample_rate), samplerate=sample_rate, channels=channels)
    sounddevice.wait()
    write_wav(filename='user_output.wav', rate=sample_rate, data=rec_user_voice)
    text_data = []
    speech_tt_model = Model(models_dir=speech_model_path, n_threads=12)
    user_voice_data = speech_tt_model.transcribe('user_output.wav', speed_up=True)

    for voice in user_voice_data:        
        text_data.append(voice.text)
    combined_text = ' '.join(text_data)
    return combined_text

def get_paragraph_before_code(sentence, stop_word):
    words = sentence.split()
    result = []
    for word in words:
        if stop_word in word:
            break
        result.append(word)
    return ' '.join(result)

def summarize_text(text=str):
    return st.session_state.summarizer(text, max_length=230, min_length=30, do_sample=False)

def on_mic_hotkey():
    st.session_state.user_voice_prompt = voice_to_text()
    print (f"text: {st.session_state.user_voice_prompt}")

def message_boop():
    message_boop = simpleaudio.WaveObject.from_wave_file(f"./ui-tones/{boop_sound}")
    message_boop.play()

# will need to add stt/tts models here too, summarizer... sdxl, turbo... lol
def exclude_id():
    return {key: value for key, value in st.session_state.items() if key != 'chat_model'}

st.progress(float("{0:.0f}".format(gpu.memoryFree)) / float("{0:.0f}".format(gpu.memoryTotal)), "vram {0:.0f}/{1:.0f}mb".format(gpu.memoryUsed, gpu.memoryTotal))
st.json(st.session_state, expanded=False)
with st.sidebar:
    notepad = st.text_area(label='notepad', label_visibility='collapsed')
    

for message in st.session_state.messages:
    with st.chat_message(name=message["role"]):
        st.markdown(message["content"])

if user_text_prompt := st.chat_input(f"Send a message to {char_name}"):

    prompt = update_chat_template(prompt=user_text_prompt, template_type=current_template)
    if enable_microphone:
        keyboard.add_hotkey('ctrl', on_mic_hotkey)
        user_voice_prompt = st.session_state.user_voice_prompt
        prompt = update_chat_template(prompt=user_voice_prompt, template_type=current_template)
        
    with st.chat_message(name="user", avatar='🙅'):
        st.markdown(user_text_prompt)      
    st.session_state.messages.append({"role": "user", "content": user_text_prompt})
    message_boop()

    popup_note(message='🍋 generating response...')
    start = time.time()
    model_output = st.session_state.chat_model(prompt=prompt, max_tokens=max_context_prompt, logprobs=log_probs)
    model_response = f"{char_name}: {model_output['choices'][0]['text']}"

    if verbose_chat:
        print(model_output)
        with st.sidebar:
            st.write(f"last message token count: {model_output['usage']['total_tokens']}")
            st.write(f"elapsed time : {int(time.time()-start)} secs")
        
    with st.chat_message(name="assistant", avatar='🤖'):
        st.markdown(model_response)
        st.json(st.session_state, expanded=False)
    st.session_state.messages.append({"role": "assistant", "content": model_response})
    message_boop()

    if enable_voice and int(model_output['usage']['total_tokens']) < xtts_token_limit:
        if code_model_voice:
            get_paragraph = get_paragraph_before_code(sentence=model_output['choices'][0]['text'], stop_word='```')
            paragraph = st.session_state.model.inference_stream(text=get_paragraph, language=language, gpt_cond_latent=gpt_cond_latent, speaker_embedding=speaker_embedding, stream_chunk_size=code_stream_chunk_size)
            wav_by_chunk(paragraph)
        else:
            chunk_inference = st.session_state.model.inference_stream(text=model_output['choices'][0]['text'], language=language, gpt_cond_latent=gpt_cond_latent, speaker_embedding=speaker_embedding, stream_chunk_size=stream_chunk_size)
            wav_by_chunk(chunk_inference)

#with open('llmon-py_state.pickle', 'wb') as f:
#    pickle.dump(exclude_id(), f)
