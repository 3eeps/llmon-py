import streamlit as st
import os
import sounddevice
import psutil
import json
import base64
from time import sleep
import keyboard
from scipy.io.wavfile import write as write_wav
from googlesearch import search
from transformers import AutoTokenizer
import torch
import torchaudio
from diffusers import StableDiffusion3Pipeline
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

def model_inference(prompt=""):
    if st.session_state.function_calling:
        st.session_state['model_temperature'] = 0.85
        st.session_state['model_min_p'] = 0.06
        st.session_state['model_top_p'] = 0.0
        st.session_state['model_top_k'] = 0
        st.session_state['repeat_penalty'] = 1.1

    if st.session_state['model_select'] == 'DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf':
        st.session_state['model_temperature'] = 0.1
        st.session_state['model_min_p'] = 0.06
        st.session_state['model_top_p'] = 0.0
        st.session_state['model_top_k'] = 0
        st.session_state['repeat_penalty'] = 1.1

    model_output = st.session_state["chat_model"](prompt=prompt,
        max_tokens=st.session_state['max_context'] - st.session_state.token_count, 
        repeat_penalty=float(st.session_state['repeat_penalty']), 
        top_k=int(st.session_state['model_top_k']),
        top_p=float(st.session_state['model_top_p']),
        min_p=float(st.session_state['model_min_p']),
        temperature=float(st.session_state['model_temperature']))
    return model_output['choices'][0]['text'], model_output['usage']['total_tokens']

def clear_vram():
    try:
        del st.session_state['sd3_medium']
    except: pass
    del st.session_state['chat_model']

def stream_text(text="", delay=0.03):
    for word in text.split(" "):
        yield word + " "
        sleep(delay)

def init_state():
    default_settings_state = {
        'max_context': 8192,
        'gpu_layer_count': -1,
        'cpu_core_count': 8,
        'cpu_batch_count': 8,
        'batch_size': 256,
        'model_temperature': 0.85,
        'model_top_p': 0.0,
        'model_top_k': 0,
        'model_min_p': 0.05,
        'repeat_penalty': 1.1,
        'message_list': [],
        'init_app': False,
        'model_select': None}
    
    st.session_state.model_list = ['Meta-Llama-3-8B-Instruct.Q6_K.gguf', 'DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf']
    st.session_state.show_generated_image = False
    st.session_state.show_uploaded_image = False
    st.session_state.bytes_data = ""
    st.session_state.sdxl_base64 = ""
    st.session_state.model_param_settings = False
    st.session_state.token_count = 0
    st.session_state.start_app = False
    st.session_state.function_calling = False
    st.session_state.function_results = ""
    st.session_state.custom_template = ""
    st.session_state.video_link = ""
    st.session_state.user_chat = False
    st.session_state.new_voice_reply = ""

    with open("functions.json", "r") as file:
        st.session_state.functions = json.load(file)

    for key, value in default_settings_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

def hide_deploy_button():
    st.markdown(
        r"""
        <style>
        .stDeployButton {
                visibility: hidden;
            }
        </style>
        """, unsafe_allow_html=True)

def load_conversation():
    with open('messsage_list.json', 'r') as file:
        st.session_state['message_list'] = json.load(file)
    with open('streamlit_list.json', 'r') as file:
        st.session_state.messages = json.load(file)

def save_conversation(message_list, streamlit_message_list):
    with open('messsage_list.json', 'w') as file:
        json.dump(message_list, file)
    with open('streamlit_list.json', 'w') as file:
        json.dump(streamlit_message_list, file)

def unload_model():
    sleep(0.5)
    del st.session_state['chat_model']

def sidebar(): 
    col1, col2, col3 = st.columns([3,1,1])
    with col1:
        st.title('🍋 llmon-py')
    with col2:
        if st.button(label="📂", help='load previous conversation'):
            load_conversation()
    with col3:
        if st.button(label='✨', help='start a new chat'):
            save_conversation(message_list=st.session_state['message_list'], streamlit_message_list=st.session_state.messages)
            st.session_state['message_list'] = []
            st.session_state.messages = []
            st.session_state.bytes_data = None
            st.session_state.sdxl_base64 = None
            st.session_state.function_calling = False
            st.session_state.custom_template = ""
            st.session_state.token_count = 0
            st.session_state.video_link = None

    loaded_model_title = ""
    if st.session_state['model_select'] == 'DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf':
        loaded_model_title = "deepseek-coder-v2"
    if st.session_state['model_select'] == 'Meta-Llama-3-8B-Instruct.Q6_K.gguf':
        loaded_model_title = "meta-llama-3"
    st.caption(f"running :green[{loaded_model_title}]")
    st.caption("")
    
    #st.session_state['model_select'] = st.selectbox(label='model list', options=st.session_state.model_list, label_visibility='hidden', help='load your model of choice', on_change=unload_model)
    uploaded_file = st.file_uploader(label='file uploader', label_visibility='collapsed', type=['png', 'jpeg'], disabled=True)
    if uploaded_file:
        st.session_state.bytes_data = uploaded_file.getvalue()
        with open("ocr_upload_image.png", 'wb') as file:
            file.write(st.session_state.bytes_data)

        with open("ocr_upload_image.png", "rb") as f:
            st.session_state.bytes_data = base64.b64encode(f.read()).decode()
        os.remove('ocr_upload_image.png')

    st.caption(body="custom model template")
    st.session_state.custom_template = st.text_area(label='custom prompt', value="", label_visibility='collapsed')
    disable_function_call = False
    if st.session_state['model_select'] == 'DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf':
        disable_function_call = True
    st.session_state.function_calling = st.checkbox(':orange[enable function calling] :green[(beta)]', value=st.session_state.function_calling, help='currently allows user to :green[generate images and find youtube videos]. may not produce desired output.', disabled=disable_function_call)
    st.session_state.model_param_settings = st.checkbox(':orange[model parameters]', value=st.session_state.model_param_settings, help='tweak model parameters to steer llama-3s output.')
    if st.session_state.model_param_settings:
        st.caption(f"ctx:{st.session_state.token_count}/{st.session_state['max_context']}")
        temp_help = "determinines whether the output is more random and creative or more predictable. :green[a higher temperature will result in lower probability], i.e more creative outputs."
        top_p_help = "controls the diversity of the generated text by only considering tokens with the highest probability mass. :green[top_p = 0.1: only tokens within the top 10% probability are considered. 0.9: considers tokens within the top 90% probability]."
        top_k_help = "limits the model's output to the top-k most probable tokens at each step. This can help reduce incoherent or nonsensical output by restricting the model's vocabulary. :green[a top-K of 1 means the next selected token is the most probable among all tokens in the model's vocabulary]."
        min_p_help = "different from top k or top p, sets a minimum percentage requirement to consider tokens relative to the largest token probability. :green[for example, min p = 0.1 is equivalent to only considering tokens at least 1/10th the top token probability]."
        rep_help = "helps the model generate more diverse content instead of repeating previous phrases. Repetition is prevented by applying a high penalty to phrases or words that tend to be repeated. :green[a higher penalty generally results in more diverse outputs, whilst a lower value might lead to more repetition]."
        st.session_state['model_temperature'] = st.text_input(label=':orange[temperature]', value=st.session_state['model_temperature'], help=temp_help)
        st.session_state['model_top_p'] = st.text_input(label=':orange[top p]', value=st.session_state['model_top_p'], help=top_p_help)
        st.session_state['model_top_k'] = st.text_input(label=':orange[top k]', value=st.session_state['model_top_k'], help=top_k_help)
        st.session_state['model_min_p'] = st.text_input(label=':orange[min p]', value=st.session_state['model_min_p'], help=min_p_help)
        st.session_state['repeat_penalty'] = st.text_input(label=':orange[repetition penalty]', value=st.session_state['repeat_penalty'], help=rep_help)
    
    bottom_col1, bottom_col2, bottom_col3 = st.columns([1,1,1])
    with bottom_col2:
        if st.button(":orange[shutdown]", help='shut down app on server side'):
            shut_down_app()

def shut_down_app():
    save_conversation(message_list=st.session_state['message_list'], streamlit_message_list=st.session_state.messages)
    if st.session_state.start_app:
        clear_vram()
    try:
        os.remove('image.txt')
    except: pass
    try:
        os.remove('.google-cookie')
    except: pass
    keyboard.press_and_release('ctrl+w')
    llmon_process_id = os.getpid()
    process = psutil.Process(llmon_process_id)
    process.terminate()

class ChatTemplate:
    def chat_template(prompt="", function_result=""):
        system_message = ""
        template = ""
        if st.session_state['model_select'] == 'Meta-Llama-3-8B-Instruct.Q6_K.gguf':
            system_message = f"""You are an helpful AI assistant, answer any request the user may have. Share postive or negative considerations when appropriate, but keep them concise. Conversation history: {st.session_state['message_list']}"""
            template = f"""<|begin_of_text|<|start_header_id|>system<|end_header_id|>{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        if st.session_state['model_select'] == 'DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf':
            system_message = f"""You are an helpful AI coding assistant, answer any request the user may have. Chat history: {st.session_state['message_list']}"""
            template = f"""<｜begin▁of▁sentence｜>{system_message}

            User: {prompt}

            Assistant: <｜end▁of▁sentence｜>Assistant:"""

        #if st.session_state.function_calling:
         #   system_message = f"""As an AI model with function calling support, you are provided with the following functions: {json.dumps(st.session_state.functions)}
         #   When the user asks a question that can be answered with a function, do not describe the function or wrap the function in "||", output the function filled with the appropriate data required as a Python dictionary."""
          #  system_message_plus_example = """Example message: 'Hello there!' Your reply: '{'function_name': 'user_chat', 'parameters': {'user_message': 'Hello there!'}}'"""

        if st.session_state.function_calling:
            system_message = f"""As an AI model with function calling support, you are provided with the following functions: {json.dumps(st.session_state.functions)}
            When the user asks a question that can be answered with a function, please do not describe the function. Only output the function filled with the appropriate data required as a Python style dictionary."""
            system_message_plus_example = """Example: User: 'play brother ali take me home' Your reply: '{'function_name': 'video_player', 'parameters': {'youtube_query': 'play brother ali take me home'}}'"""
            template = f"""<|begin_of_text|<|start_header_id|>system<|end_header_id|>{system_message + system_message_plus_example}<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            if len(function_result) > 1:
                system_message = f"""Using this data that is up to date, reply to the user using it: {function_result}. The question the user asked was:"""
                template = f"""<|begin_of_text|<|start_header_id|>system<|end_header_id|>{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        if len(st.session_state.custom_template) > 1:
            system_message = f"""{st.session_state.custom_template} Chat history: {st.session_state['message_list']}"""
            template = f"""<|begin_of_text|<|start_header_id|>system<|end_header_id|>{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""                 

        return template

class Audio:
    def voice_to_text():
        rec_user_voice = sounddevice.rec(int(st.session_state['user_audio_length']) * 44100, samplerate=44100, channels=2)
        sounddevice.wait()
        write_wav(filename='user_output.wav', rate=44100, data=rec_user_voice)
        user_voice_data = st.session_state['speech_tt_model'].transcribe('user_output.wav', speed_up=True)
        os.remove(f"user_output.wav")
        text_data = []
        for voice in user_voice_data:        
            text_data.append(voice.text)
        combined_text = ' '.join(text_data)
        return combined_text

class Functions:
    def find_youtube_link(user_query):
        search_helper = 'youtube'
        search_query = user_query + search_helper
        for youtube_link in search(query=search_query, tld="co.in", num=1, stop=1, pause=2):
            print(youtube_link)
            return youtube_link
        return 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    
    def change_llm_voice(voice_description):
        #load models
        device = "cuda:0"
        model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device, dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

        config = XttsConfig()
        config.load_json("./xtts_model/config.json")
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir="./xtts_model/")
        model.cuda()

        style_prompt = "It took me quite a long time to develop a voice and now that I have it I am not going to be silent."


        #### easy way!!!! make style voice here... use the voice with built in tts, even if its slow
        #### trick!!! when using tts... ask the llm to keep anwers very brief. less then a paragraph. should help inference time!



        #run parlor with user voice description
        input_ids = tokenizer(voice_description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(style_prompt, return_tensors="pt").input_ids.to(device)
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids).to(torch.float32)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write("new_voice_style.wav", audio_arr, model.config.sampling_rate)

        # send new voice style to xtts to say 'how is it?' trick :P
        language = 'en'
        #st.session_state.new_voice_reply = model_inference(prompt=f"The user has changed your text to speech voice! Here is the voice description from the user: {voice_description}. Say something fun to show off your new voice!")
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["new_voice_style.wav"])
        tts_chunks = model.inference_stream(
            prompt,
            language,
            gpt_cond_latent,
            speaker_embedding)
        
        chunk_list = []
        for i, chunk in enumerate(tts_chunks):
            chunk_list.append(chunk)
            print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
        wav_chunk = torch.cat(chunk_list, dim=0)
        torchaudio.save("new_llm_final_voice.wav", wav_chunk.squeeze().unsqueeze(0).cpu(), 24000)
        return st.session_state.new_voice_reply
        
class SD3Medium:
    def init():
        st.session_state['sd3_medium'] = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16,
            text_encoder_3=None, 
            tokenizer_3=None
        ).to("cuda")

    def generate_image(prompt=""):
        image = st.session_state['sd3_medium'](
            prompt,
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=7.0
        ).images[0]
        image.save('image_sd3.png')

        with open("image_sd3.png", "rb") as f:
            st.session_state.sdxl_base64 = base64.b64encode(f.read()).decode()
        os.remove('image_sd3.png')