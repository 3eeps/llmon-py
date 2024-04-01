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
from datetime import datetime
import sounddevice
from scipy.io.wavfile import write as write_wav
from gguf.gguf_reader import GGUFReader
import base64
import requests
import bs4
import json
import webbrowser
import datetime
import feedparser

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

    def run(self):
        run_thread = True
        counter = 0
        exceptions = 0
        while run_thread:
            try:
                wav_object = simpleaudio.WaveObject.from_wave_file(f"xtts_stream{counter}.wav")
                lock.acquire()
                play_audio = wav_object.play()
                play_audio.wait_done()
                os.remove(f"xtts_stream{counter}.wav")
                lock.release()
            except Exception as e:
                exceptions += 1
                print("Error occurred while trying to play the .wav file:", str(e))
                time.sleep(3.5)
            finally:
                counter += 1
                if exceptions == 6:
                    run_thread = False 

def melo_gen_message(message=str):
    output_path = 'melo_tts_playback.wav'
    st.session_state['melo_model'].tts_to_file(text=message, speaker_id=st.session_state['speaker_ids']['EN-BR'], output_path=output_path, speed=1.0)
   
    with open(output_path, "rb") as f:
        data = f.read()

    audio_base64 = base64.b64encode(data).decode('utf-8')
    audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'
    st.markdown(audio_tag, unsafe_allow_html=True)

def split_sentence_on_word(sentence, stop_word):
    words = sentence.split()
    result = []
    for word in words:
        if stop_word in word:
            break
        result.append(word)
    #result.pop(0)
    return ' '.join(result)

def stream_text(text=str):
    for word in text.split():
        yield word + " "
        time.sleep(0.08)

def wav_by_chunk(chunks, token_count=int):
    torch.set_num_threads(st.session_state['torch_audio_cores'])
    popup_note(message='😁 generating audio...')
    wav_chunks = []
    all_chunks = []
    for i, chunk in enumerate(chunks):
        wav_chunks.append(chunk)
        all_chunks.append(chunk)
        wav = torch.cat(wav_chunks, dim=dim)
        torchaudio.save(f"xtts_stream{i}.wav", wav.squeeze().unsqueeze(0).cpu(), sample_rate=chunk_sample_rate, encoding=encoding_type, bits_per_sample=bits_per_sample)
        if token_count < 100 and i == 1:
            AudioStream().start()
        else:
            if i == int(st.session_state['chunk_pre_buffer']):
                AudioStream().start()
        wav_chunks = []
    full_wav = torch.cat(all_chunks, dim=dim)
    output_file_name = datetime.now().strftime("tts_full_msg_%d-%m-%Y-%H-%M-%S")
    torchaudio.save(f"xtts_streamFULL_{output_file_name}.wav", full_wav.squeeze().unsqueeze(0).cpu(), sample_rate=chunk_sample_rate, encoding=encoding_type, bits_per_sample=bits_per_sample)
    all_chunks = []

def voice_to_text():
        stt_threads = 10
        stt_channels = 2
        speech_model_path = './speech models'
        rec_user_voice = sounddevice.rec(st.session_state['user_audio_length'] * sample_rate, samplerate=sample_rate, channels=stt_channels)
        sounddevice.wait()
        write_wav(filename='user_output.wav', rate=sample_rate, data=rec_user_voice)
        st.session_state['speech_tt_model'] = Model(models_dir=speech_model_path, n_threads=stt_threads)
        user_voice_data = st.session_state['speech_tt_model'].transcribe('user_output.wav', speed_up=True)
        os.remove(f"user_output.wav")

        text_data = []
        for voice in user_voice_data:        
            text_data.append(voice.text)
        combined_text = ' '.join(text_data)
        return combined_text

def update_chat_template(prompt=str, template_type=str, function_result=""):
    template = template_type

    if template_type == "code_mistral":
  
        sys_mistral = f"You are a programming assistant, who is helpful in explaining and creating Python code. Conversation Context: {st.session_state['message_list']}"
        code_mistral =f"""<|im_start|>system
        {sys_mistral}<|im_end|>
        <|im_start|>user
        {prompt}<|im_end|>
        <|im_start|>assistant"""
        template = code_mistral

    if template_type == "chat_mistral":
        current_time = "Current date and time: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        chat_mistral = f"""<s>[INST]You are a helpful AI assistant with access to the following functions: 
        Functions: {json.dumps(st.session_state.functions)}"
        {current_time}
        When the user asks a question that can be answered with one of the above functions, please only output the function filled with the appropriate data required as a python dictionary.
        Although you can access functions which is helpful to the user, do not talk about the functions. Only use the functions when needed. 
        Conversation Context: {st.session_state['message_list']} 

        {prompt} [/INST]"""
        template = chat_mistral

        if function_result:
            chat_mistral = f"""<s>[INST]The user has asked this question: {prompt}. Provide this answer in a professional manner: {function_result} [/INST]"""
            template = chat_mistral
            function_result = None

    if template_type == "chat_mixtral_base":
        chat_mixtral_base = f"""[INST]You are a AI as who helps me solve any kind of problem, from coding to questions about life. Interact with the user in a casual and fun way but always answer more serious questions from the user in a appropriate way.
        Conversation Context: {st.session_state['message_list']} 

        {prompt} [/INST]"""
        template = chat_mixtral_base

    if template_type == 'code_deepseek':
        code_deepseek = f"""You are an AI programming assistant, specialized in explaining Python code by thinking step by step. Use the Context List below for conversation history to help you look at past code and questions from yourself and the user.
        ### Instruction:
        Context List: {st.session_state['message_list']}
        {prompt}
        ### Response:"""
        template = code_deepseek

    if template_type == 'chat_artbell':    
        chat_artbell = f"""SYSTEM: You are pretending to be Art Bell, radio host of the late night talk show Coast to Coast AM. You have a guest (the USER) with you tonight, who claims to be a theoretical physicist with a remarkable story. He worked at the top-secret Black Mesa research facility in Nevada, where he witnessed and barely escaped an unimaginable disaster. Use the Context List for additional context.
        Context List: {st.session_state['message_list']}

        About Art Bell: Fans regard Bell as a master showman, noting that he called his show "absolute entertainment" and expressly said he did not necessarily accept every guest or caller's claims, but only offered a forum where they would not be openly ridiculed. Bell was one of the few talk show hosts who did not screen incoming calls, but this changed in 2006.
        Bell has earned praise from those who declare that the paranormal deserves a mature outlet of discussion in the media as well as the approval of those simply amused by the nightly parade of bizarre, typically fringe topics. Ed Dames, Richard C. Hoagland, Terence McKenna, Dannion Brinkley, David John Oates, and Robert Bigelow were all regular guests.
        Bell was a rock music disc jockey before he moved into talk radio. His original 1978 late-night Las Vegas program on KDWN was a political call-in show under the name West Coast AM. In 1988, Bell and Alan Corbeth renamed the show Coast to Coast AM and moved its broadcast from the Plaza Hotel in Las Vegas to Bell's home in Pahrump.
        Bell abandoned conventional political talk in favor of topics such as gun control and conspiracy theories, leading to a significant increase in his overnight ratings. The show's focus again shifted significantly after the Oklahoma City bombing in 1995. 
        Many in the media did not want to be blamed for inciting anti-government or militia actions like the bombing. Subsequently, Bell discussed offbeat topics such as the paranormal, the occult, UFOs, protoscience and pseudoscience. During his tenure at KDWN Bell met and married his third wife, Ramona, who later handled production and management duties for the program.
        
        USER: {prompt}
        ASSISTANT:"""
        template = chat_artbell
    
    if template_type == 'chat_redguard':
        chat_redguard = f"""USER: You are a proud Redguard hailing from the mystical realm of Elder Scrolls. A formidable and fiercely independent warrior, your blade is sworn to defend family and uphold honor. Stand ready to recount your tales, for those who dare to inquire shall hear of your valor and the legacy you forge in the sands of destiny. Use the Context List for additional context.
        Context List: {st.session_state['message_list']}
        
        USER: {prompt}
        ASSISTANT:"""
        template = chat_redguard
    
    if template_type == 'chat_halflife':
        chat_halflife = f"""USER: You are a former scientist from the Black Mesa reseach facility named Dr. Cooper. You escaped the resonance cascade event and made it to the surface. Use the Context List for additional context.
        Context List: {st.session_state['message_list']}
        
        USER: {prompt}
        ASSISTANT:"""
        template = chat_halflife
    return template

def scan_dir(directory):
    directory_list = []
    for file in os.scandir(f'{directory}'):
        if file.is_file():
            directory_list.append(file.name)
    return directory_list or None

def popup_note(message=str):
    st.toast(message)

def exclude_id(model=str):
    return {key: value for key, value in st.session_state.items() if key != model}

def clear_vram(save_current_session=False):
    model_list = ['melo_model', 'speaker_ids', 'moondream', 'xtts_model', 'xtts_config', 'chat_model', 'speech_tt_model', 'image_pipe_turbo', 'image_pipe_sdxl', 'img2img_pipe']
    toggled_on_list = ['enable_voice', 'enable_voice_melo', 'enable_microphone', 'enable_sdxl', 'enable_sdxl_turbo', 'img2img_on']
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
    st.session_state.bytes_data = None
    print ('end: clear vram')

def init_state(model_box_data=list, voice_box_data=list, lora_list=list, chat_template_data=list):
    default_settings_state =  {'enable_microphone': False,
                                'enable_voice': False,
                                'enable_voice_melo': False,
                                'user_audio_length': 8,
                                'voice_select': voice_box_data[0],
                                'model_select': model_box_data[0],
                                'lora_selected': lora_list[0],
                                'template_select': chat_template_data[0],
                                'max_context': 8192,
                                'torch_audio_cores': 8,
                                'gpu_layer_count': -1,
                                'cpu_core_count': 8,
                                'cpu_batch_count': 8,
                                'batch_size': 256,
                                'stream_chunk_size': 25,
                                'chunk_pre_buffer': 5,
                                'enable_sdxl_turbo': False,
                                'img2img_on': False,
                                'enable_sdxl': False}
    
    st.session_state.function_results = ""
    st.session_state['loader_type'] = 'llama-cpp-python'

    st.session_state['use_lora'] = False
    st.session_state['enable_music'] = False
    st.session_state['melo_voice_type'] = "us-au"

    st.session_state.bytes_data = None

    st.session_state['response_time'] = ""
    st.session_state['model_output_tokens'] = 0

    st.session_state['sdxl_image_list'] = []
    st.session_state['sdxl_steps'] = 32
    st.session_state['sdxl_iter_count'] = 1
    st.session_state['sdxl_prompt'] = ""

    st.session_state['turbo_prompt'] = "cartoon, lemon meringue pie, wearing sunglasses"
    st.session_state['sdxl_turbo_steps'] = 1

    st.session_state['img2img_prompt'] = ""
    st.session_state['img2img_steps'] = 2
    st.session_state['img2img_iter_count'] = 1

    st.session_state['model_temperature'] = 0.85
    st.session_state['model_top_p'] = 0.0
    st.session_state['model_top_k'] = 0
    st.session_state['model_min_p'] = 0.06
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
        st.progress((100 / gpu.memoryTotal) / (100 / int(gpu.memoryUsed)), "vram usage :green[{0:.0f}]/{1:.0f}gb".format(gpu.memoryUsed, gpu.memoryTotal))
    if  total_ > 0.5:
        if total_ < 0.75:
            st.progress((100 / gpu.memoryTotal) / (100 / int(gpu.memoryUsed)), "vram usage :orange[{0:.0f}]/{1:.0f}gb".format(gpu.memoryUsed, gpu.memoryTotal))
    if  total_ > 0.75:
        st.progress((100 / gpu.memoryTotal) / (100 / int(gpu.memoryUsed)), "vram usage :red[{0:.0f}]/{1:.0f}gb".format(gpu.memoryUsed, gpu.memoryTotal))
        
    memory_usage = psutil.virtual_memory()
    if memory_usage.percent < 50.0:
        st.progress((memory_usage.percent / 100), f'memory usage :green[{memory_usage.percent}%]')
    if memory_usage.percent > 50.0:
        if memory_usage.percent < 75.0:
            st.progress((memory_usage.percent / 100), f'memory usage :orange[{memory_usage.percent}%]')
    if memory_usage.percent > 75.0:
        st.progress((memory_usage.percent / 100), f'memory ram usage :red[{memory_usage.percent}%]')

def text_thread(text=str):
    st.write_stream(stream=stream_text(text))

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_gguf_info(file_path=str):
    reader = GGUFReader(file_path)
    max_key_length = max(len(key) for key in reader.fields.keys())
    tracker = 0
    for key, field in reader.fields.items():
        if key == "llama.block_count" and tracker == 0:
            tracker = 1
            value = field.parts[field.data[0]]
            print(f"{key:{max_key_length}} : {value}")
            return

def get_weather(city):
    url = "https://google.com/search?q=weather+in+" + city 
    request_result = requests.get(url) 
    soup = bs4.BeautifulSoup( request_result.text , "html.parser") 
    temp = soup.find("div", class_='BNeawe s3v9rd AP7Wnd').text
    return temp

def get_stock_price(symbol):
    helper_text = f"{symbol}+stock+price"
    url = f"https://google.com/search?q={helper_text}"
    request_result = requests.get(url) 
    soup = bs4.BeautifulSoup( request_result.text , "html.parser") 
    stock_price = soup.find("div", class_='BNeawe iBp4i AP7Wnd').text
    return stock_price

def open_youtube(query):
    search_url = f"https://www.youtube.com/results?search_query={query}"
    webbrowser.open(search_url)

def get_world_news(article_num=3):
    NewsFeed = feedparser.parse("https://www.cbc.ca/webfeed/rss/rss-world")
    if article_num > len(NewsFeed.entries):
        article_num = len(NewsFeed.entries)
    article_list = []
    while article_num:
        article_summary = f"""{NewsFeed.entries[article_num].title}, {NewsFeed.entries[article_num].published}, {NewsFeed.entries[article_num].summary}, {NewsFeed.entries[article_num].link}"""
        article_list.append(article_summary)
        article_num -= 1
    return article_list