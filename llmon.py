# ./codespace/llmonpy.py
import streamlit as st
import os
import torch
import sounddevice
import GPUtil as GPU
import psutil
from scipy.io.wavfile import write as write_wav
import base64
import requests
import bs4
import json
import webbrowser
import feedparser
from googlesearch import search
from pywhispercpp.model import Model
from diffusers import AutoPipelineForText2Image
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

def melo_gen_message(message=str):
    #EN-Default, EN-US, EN-BR, EN_INDIA EN-AU
    output_path = 'melo_tts_playback.wav'
    st.session_state['melo_model'].tts_to_file(text=message, speaker_id=st.session_state['speaker_ids']['EN-AU'], output_path=output_path, speed=1.0)
   
    with open(output_path, "rb") as file:
        data = file.read()

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

def voice_to_text():
        speech_model_path = './speech models'
        rec_user_voice = sounddevice.rec(int(st.session_state['user_audio_length']) * 44100, samplerate=44100, channels=2)
        sounddevice.wait()
        write_wav(filename='user_output.wav', rate=44100, data=rec_user_voice)
        st.session_state['speech_tt_model'] = Model(models_dir=speech_model_path, n_threads=10)
        user_voice_data = st.session_state['speech_tt_model'].transcribe('user_output.wav', speed_up=True)
        os.remove(f"user_output.wav")

        text_data = []
        for voice in user_voice_data:        
            text_data.append(voice.text)
        combined_text = ' '.join(text_data)
        return combined_text

def update_chat_template(prompt="", template_type="", function_result=""):
    template = ""
    if template_type == "code_mistral":
        sys_mistral = f"You are a programming assistant, who is helpful at explaining and creating Python code. Conversation history: {st.session_state['message_list']}"

        code_mistral =f"""<|im_start|>system
        {sys_mistral}<|im_end|>
        <|im_start|>user
        {prompt}<|im_end|>
        <|im_start|>assistant"""
        template = code_mistral

    if template_type == "func_mistral":
        func_mistral = f"""<s>[INST]You are a function calling AI model. You are provided with the following functions: 
        Functions: {json.dumps(st.session_state.functions)}"
        When the user asks a question that can be answered with one of the above functions, only output the function filled with the appropriate data required as a python dictionary.
        Do not describe the function, only reply with the function and the data as a python dictionary.
        {prompt} [/INST]"""
        template = func_mistral

        normal_reply = False
        if function_result == "func_reply":
            normal_reply = True
            func_mistral = f"""<s>[INST]You are an helpful AI assistant, answer any request the user may have. Chat history: {st.session_state['message_list']}

            {prompt} [/INST]"""
            template = func_mistral

        if function_result and normal_reply == False:
            func_mistral = f"""<s>[INST]The user has asked this question: {prompt}. Provide this answer: {function_result}. [/INST]"""
            template = func_mistral

    if template_type == "chat_mixtral":
        chat_mixtral = f"""[INST]You are an helpful AI assistant, answer any request the user may have. Chat history: {st.session_state['message_list']}
        Conversation history: {st.session_state['message_list']}

        {prompt} [/INST]"""
        template = chat_mixtral

    if template_type == "tiny_dolphin":
        tiny_sys = f"""You are a helpful AI assistant. Chat history: {st.session_state['message_list']}"""
        tiny_dolphin = f"""<|im_start|>system
        {tiny_sys}<|im_end|>
        <|im_start|>user
        {prompt}<|im_end|>
        <|im_start|>assistant"""
        template = tiny_dolphin

    if template_type == 'code_deepseek':
        code_deepseek = f"""You are an AI programming assistant, specialized in explaining Python code by thinking step by step. Use the 'Context History' below for conversation history to help you look at past code and questions from yourself and the user.
        ### Instruction:
        Context History: {st.session_state['message_list']}
        {prompt}
        ### Response:"""
        template = code_deepseek

    if template_type == 'chat_llama3':
        system_message = f"""You are an helpful AI assistant, answer any request the user may have. Chat history: {st.session_state['message_list']}"""
        chat_llama3 = f"""<|begin_of_text|<|start_header_id|>system<|end_header_id|>
        {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
        {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        template = chat_llama3

    if template_type == "instruct_phi":
        instruct_phi = f"""<|user|>
        {prompt}<|end|>
        <|assistant|>"""
        template = instruct_phi

    return template

def scan_dir(directory):
    directory_list = []
    for file in os.scandir(f'{directory}'):
        if file.is_file():
            directory_list.append(file.name)
    return directory_list or None

def clear_vram():
    model_list = ['melo_model', 'speaker_ids', 'xtts_model', 'gpt_cond_latent', 'xtts_config', 'chat_model']
    toggled_on_list = ['enable_xtts', 'enable_melo']

    for toggle in toggled_on_list:
        if st.session_state[toggle]:
            st.session_state[toggle] = False

    for model in model_list:
        try:
            del st.session_state[model]
            print(f'unloaded {model}')
        except: pass

    st.session_state['message_list'] = []
    st.session_state.messages = []
    st.session_state.bytes_data = None
    st.session_state.model_loader = False
    st.session_state.bite_llmon = False
    st.session_state.lock_input = False
    st.session_state['model_output_tokens'] = 0
    torch.cuda.empty_cache()
    print ('cleared vram')

def init_state():
    default_settings_state = {  'enable_xtts': False,
                                'enable_melo': False,
                                'user_audio_length': 8,
                                'max_context': 4096,
                                'torch_audio_cores': 8,
                                'cpu_core_count': 8,
                                'cpu_batch_count': 8,
                                'batch_size': 256,
                                'gpt_cond_latent': None,
                                'speaker_embedding': None,
                                'loader_type': 'llama-cpp-python',
                                'model_output_tokens': 0,
                                'model_temperature': 0.85,
                                'model_top_p': 0.0,
                                'model_top_k': 0,
                                'model_min_p': 0.06,
                                'repeat_penalty': 1.1,
                                'message_list': []}

    with open("functions.json", "r") as file:
        st.session_state.functions = json.load(file)

    st.session_state.model_loader = False
    st.session_state.bite_llmon = False
    st.session_state.model_list = scan_dir('./models')
    st.session_state.chat_templates = ['code_mistral', 'code_deepseek', 'chat_llama3', 'func_mistral', 'chat_mixtral' , 'instruct_phi', 'tiny_dolphin']
    st.session_state.video_link = None
    st.session_state.first_watch = False
    st.session_state.function_results = ""
    st.session_state.bytes_data = None
    st.session_state['app_state_init'] = True
    st.session_state.lock_input = False
    st.session_state.model_label = 'visible'

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
        st.progress((100 / gpu.memoryTotal) / (100 / int(gpu.memoryUsed)), "gpu memory: :green[{0:.0f}/{1:.0f}gb]".format(gpu.memoryUsed, gpu.memoryTotal))
    if  total_ > 0.5:
        if total_ < 0.75:
            st.progress((100 / gpu.memoryTotal) / (100 / int(gpu.memoryUsed)), "gpu memory: :orange[{0:.0f}/{1:.0f}gb]".format(gpu.memoryUsed, gpu.memoryTotal))
    if  total_ > 0.75:
        st.progress((100 / gpu.memoryTotal) / (100 / int(gpu.memoryUsed)), "gpu memory: :red[{0:.0f}/{1:.0f}gb]".format(gpu.memoryUsed, gpu.memoryTotal))
        
    memory_usage = psutil.virtual_memory()
    if memory_usage.percent < 50.0:
        st.progress((memory_usage.percent / 100), f"system memory: :green[{memory_usage.percent}%]")
    if memory_usage.percent > 50.0:
        if memory_usage.percent < 70.0:
            st.progress((memory_usage.percent / 100), f"system memory: :orange[{memory_usage.percent}%]")
    if memory_usage.percent > 70.0:
        st.progress((memory_usage.percent / 100), f"system memory: :red[{memory_usage.percent}%]")

class FunctionCall:
    def get_weather(city):
        url = "https://google.com/search?q=current+weather+forecast+" + city 
        request_result = requests.get(url) 
        soup = bs4.BeautifulSoup( request_result.text , "html.parser") 
        temp = soup.find("div", class_='BNeawe s3v9rd AP7Wnd').text
        return temp

    def youtube_download(link=str):
        #youtube_url = link
        #response = requests.request("GET", youtube_url)
        #soup = bs4.BeautifulSoup(response.text, "html.parser")
        #body = soup.find_all("body")[0]
        #scripts = body.find_all("script")
        #result = json.loads(scripts[0].string[30:-1])
        #print(result['streamingData']['formats'][0]['url'])
        #return result['streamingData']['formats'][0]['url']

        helper = 'youtube'
        # to search
        query = link + helper
 
        for j in search(query, tld="co.in", num=1, stop=1, pause=2):
            print(j)
        return j
    
    def open_youtube(query):
        search_url = f"https://www.youtube.com/results?search_query={query}"
        webbrowser.open(search_url)
    
    def get_news(article_num=3):
        NewsFeed = feedparser.parse("https://www.cbc.ca/webfeed/rss/rss-world")
        if article_num > len(NewsFeed.entries):
            article_num = len(NewsFeed.entries)
        article_list = []
        while article_num:
            article_summary = f"""{NewsFeed.entries[article_num].title}, {NewsFeed.entries[article_num].published}, {NewsFeed.entries[article_num].summary}, {NewsFeed.entries[article_num].link}"""
            article_list.append(article_summary)
            article_num -= 1
        return article_list
    
    def get_stock_price(symbol):
        helper_text = f"{symbol}+stock+price"
        url = f"https://google.com/search?q={helper_text}"
        request_result = requests.get(url) 
        soup = bs4.BeautifulSoup( request_result.text , "html.parser") 
        stock_price = soup.find("div", class_='BNeawe iBp4i AP7Wnd').text
        return stock_price

class SDXLTurbo:
    def init():
        st.session_state['image_pipe_turbo'] = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", variant="fp16").to('cpu')
    
    def generate_image(prompt="", steps=1):
        _image = st.session_state['image_pipe_turbo'](prompt=prompt, num_inference_steps=steps, guidance_scale=0.0).images[0]
        _image.save('image_turbo.png')

class Moondream:
    def load_vision_encoder():
        model_id = "moondream2"
        revision = "2024-04-02"
        st.session_state['moondream'] = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision)
        st.session_state.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def generate_response(prompt=str):
        filename = "ocr_upload_image.png"
        image = Image.open(filename)
        enc_image = st.session_state['moondream'].encode_image(image)
        return st.session_state['moondream'].answer_question(enc_image, prompt, st.session_state.tokenizer)
