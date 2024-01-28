# ./codespace/pages/model inference.py

import streamlit as st
from streamlit_extras.app_logo import add_logo 

st.set_page_config(page_title="model inference", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("model inference")
if st.session_state['enable_voice']:
    st.markdown(f"with :orange[***{st.session_state['model_select']}***]   :green[+ xttsv2]!")
else:
    st.markdown(f"with :orange[***{st.session_state['model_select']}***]")
add_logo("./llmon_art/pie.png", height=130)

import os
import time
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

os.system('cls')
torch.set_num_threads(st.session_state['torch_audio_cores'])
warnings.filterwarnings('ignore')

chat_model_path = f"./models/{st.session_state['model_select']}"
chat_model_voice_path = f"./voices/{st.session_state['voice_select']}"
reveal_logits = True
log_probs = 5
message_wav = 'chat_pop.wav'
language = 'en'
stt_threads = 10
speech_model_path = './pywhisper_models'
warmup_string = 'warmup string'
chat_warmup_prompt = 'hello!'
bits_per_sample = 16
encoding_type = 'PCM_S'
code_stream_chunk_size = 40
warmup_chunk_size = 40
sample_rate = 44100
chunk_sample_rate = 24000
channels = 2
dim = 0
popup_delay = 1.0

def popup_note(message=str):
    if st.session_state['enable_popups']:
        st.toast(message)

if "messages" not in st.session_state:
    st.session_state.messages = []

if 'model' not in st.session_state and st.session_state['enable_voice']:# or st.session_state.config == None:
    popup_note(message='😤 hyping up tts model...')
    st.session_state.config = XttsConfig()
    st.session_state.config.load_json("./xtts_config/config.json")
    st.session_state.model = Xtts.init_from_config(st.session_state.config)
    st.session_state.model.load_checkpoint(st.session_state.config, checkpoint_dir="./xtts_config")
    st.session_state.model.cuda()
    
#if 'summarizer' not in st.session_state:
#    popup_note(message='😎 its game time summarizer!')
#    st.session_state.summarizer = pipeline("summarization", model="./summarizer/t5-small")

if 'speech_tt_model' not in st.session_state and st.session_state['enable_microphone']:
    popup_note(message='😎 lets get it stt model!')
    st.session_state.user_voice_prompt = None
    st.session_state.speech_tt_model = Model(models_dir=speech_model_path)

if st.session_state['enable_voice']:
    gpt_cond_latent, speaker_embedding = st.session_state.model.get_conditioning_latents(audio_path=[f"{chat_model_voice_path}"])
    warmup_tts = st.session_state.model.inference_stream(text=warmup_string, language=language, gpt_cond_latent=gpt_cond_latent, speaker_embedding=speaker_embedding, stream_chunk_size=warmup_chunk_size)

if 'chat_model' not in st.session_state: # or st.session_state['chat_model'] == None:
    popup_note(message='😴 waking up chat model...')
    logits_list = str
    st.session_state.chat_model = Llama(model_path=chat_model_path, logits_all=reveal_logits, n_batch=st.session_state['batch_size'], n_threads=st.session_state['cpu_core_count'], n_threads_batch=st.session_state['cpu_batch_count'], n_gpu_layers=st.session_state['gpu_layer_count'], n_ctx=st.session_state['max_context'], verbose=st.session_state['verbose_chat'])
    warmup_chat = st.session_state.chat_model(prompt=chat_warmup_prompt)

class AudioStream(Thread):
    def __init__(self):
        super(AudioStream, self).__init__()
        self.run_thread = True
        self.counter = 0
        self.start()

    def run(self):
        try_again = False
        while self.run_thread:
            try:    
                wav_object = simpleaudio.WaveObject.from_wave_file(f"xtts_stream{self.counter}.wav")
                play_audio = wav_object.play()
                play_audio.wait_done()
                os.remove(f"xtts_stream{self.counter}.wav")
                self.counter += 1
            except:
                try_again = True
                if try_again:
                    pass
                else:
                    self.run_thread = False
                    
def update_chat_template(prompt=str, template_type=str):
    template = template_type

    if template_type == "alpaca_noro":
        alpaca_noro = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {prompt}

        ### Response:"""
        template_type = alpaca_noro

    if template_type == "none":
        none_type = f"""{prompt}"""
        template_type = none_type

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

def wav_by_chunk(chunks, token_count=int):
    popup_note(message='😁 generating audio...')
    wav_chunks = []
    all_chunks = []
    chunk_buffer = st.session_state['chunk_pre_buffer']
    for i, chunk in enumerate(chunks):
        wav_chunks.append(chunk)
        all_chunks.append(chunk)
        wav = torch.cat(wav_chunks, dim=dim)
        torchaudio.save(f"xtts_stream{i}.wav", wav.squeeze().unsqueeze(0).cpu(), sample_rate=chunk_sample_rate, encoding=encoding_type, bits_per_sample=bits_per_sample)
        if token_count < 50:
            chunk_buffer = 1
        if i == chunk_buffer and wav_chunks:
            AudioStream()
        wav_chunks = []
    full_wav = torch.cat(all_chunks, dim=dim)
    torchaudio.save("xtts_stream_full.wav", full_wav.squeeze().unsqueeze(0).cpu(), sample_rate=chunk_sample_rate, encoding=encoding_type, bits_per_sample=bits_per_sample)
    all_chunks = []

def voice_to_text():
    rec_user_voice = sounddevice.rec(st.session_state['user_audio_length'] * sample_rate, samplerate=sample_rate, channels=channels)
    sounddevice.wait()
    write_wav(filename='user_output.wav', rate=sample_rate, data=rec_user_voice)
    speech_tt_model = Model(models_dir=speech_model_path, n_threads=stt_threads)
    user_voice_data = speech_tt_model.transcribe('user_output.wav', speed_up=True)
    os.remove(f"user_output.wav")

    text_data = []
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

def message_boop():
    message_boop = simpleaudio.WaveObject.from_wave_file(f"./ui-tones/{message_wav}")
    message_boop.play()

with st.sidebar:
    notepad = st.text_area(label='notepad', label_visibility='collapsed')
    if st.session_state['enable_microphone']:
        st.markdown(f":red[*microphone enabled*]")

for message in st.session_state.messages:
    with st.chat_message(name=message["role"]):
        st.markdown(message["content"])

input_message = str
if st.session_state['enable_microphone']:
    input_message = f"Type a message to {st.session_state['char_name']}, or use microphone by typing 'q'"
else:
    input_message = f"Send a message to {st.session_state['char_name']}"

if user_text_prompt := st.chat_input(input_message):
    user_prompt = user_text_prompt
    if user_text_prompt == 'q' and st.session_state['enable_microphone']:
        user_prompt = voice_to_text()

    final_prompt = update_chat_template(prompt=user_prompt, template_type=st.session_state['template_select'])

    with st.chat_message(name="user", avatar='🙅'):
        st.markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    message_boop()

    popup_note(message='🍋 generating response...')
    llm_start = time.time()
    model_output = st.session_state.chat_model(prompt=final_prompt, max_tokens=st.session_state['max_context_prompt'], logprobs=log_probs)
    model_response = f"{st.session_state['char_name']}: {model_output['choices'][0]['text']}"

    if st.session_state['verbose_chat']:
        print(model_output)
        with st.sidebar:
            st.write(f"last message token count: {model_output['usage']['total_tokens']}")
            st.write(f"elapsed time : {int(time.time()-llm_start)} secs")
        
    with st.chat_message(name="assistant", avatar='🤖'):
        st.markdown(model_response)
        st.json(st.session_state, expanded=False)
    st.session_state.messages.append({"role": "assistant", "content": model_response})
    message_boop()

    if st.session_state['enable_voice']:
        tts_start = time.time()
        if st.session_state['enable_code_voice']:
            get_paragraph = get_paragraph_before_code(sentence=model_output['choices'][0]['text'], stop_word='```')
            paragraph = st.session_state.model.inference_stream(text=get_paragraph, language=language, gpt_cond_latent=gpt_cond_latent, speaker_embedding=speaker_embedding, stream_chunk_size=code_stream_chunk_size)
            wav_by_chunk(chunks=paragraph, token_count=int(model_output['usage']['total_tokens']))
        else:
            chunk_inference = st.session_state.model.inference_stream(text=model_output['choices'][0]['text'], language=language, gpt_cond_latent=gpt_cond_latent, speaker_embedding=speaker_embedding, stream_chunk_size=st.session_state['stream_chunk_size'], enable_text_splitting=True)
            wav_by_chunk(chunks=chunk_inference, token_count=int(model_output['usage']['total_tokens']))
