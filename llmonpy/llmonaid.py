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
             
def melo_audio():
    #wav_object = simpleaudio.WaveObject.from_wave_file('melo_tts.wav')
    #play_audio = wav_object.play()
    #play_audio.wait_done()
    pass

def melo_gen_message(message=str, token_count=int):
    print(token_count)
    output_path = 'melo_tts_playback.wav'
    #if token_count < 1024:
    st.session_state['melo_model'].tts_to_file(text=message, speaker_id=st.session_state['speaker_ids']['EN-AU'], output_path=output_path, speed=1.0)
        #st.audio(data=output_path)
        
    with open(output_path, "rb") as f:
        data = f.read()

    audio_base64 = base64.b64encode(data).decode('utf-8')
    audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'
    st.markdown(audio_tag, unsafe_allow_html=True)
    os.remove(output_path)

def melo_audio_splitter():
    pass

def attempt_login(model_box_data=list, voice_box_data=list, lora_list=list, chat_templates=list):
    if st.session_state['approved_login'] == False:
        st.write("login to access llmon-py")
        username = st.text_input("username")
        password = st.text_input("password", type="password")
        login_button = st.button('sign in')
        if login_button == True and username == "chad" and password == "chad420":
            st.session_state['approved_login'] = True    
            st.session_state['user_type'] = 'admin'

        if login_button == True and username == "user" and password == "userpass":
            st.session_state['approved_login'] = True    
            st.session_state['user_type'] = 'user_basic'

        if login_button == True and username == "" and password == "":
            st.session_state['approved_login'] = True    
            st.session_state['user_type'] = 'admin'

        if st.session_state['approved_login']:
            init_state(model_box_data, voice_box_data, lora_list, chat_templates)
            st.rerun()

def split_sentence_on_word(sentence, stop_word):
    words = sentence.split()
    result = []
    for word in words:
        if stop_word in word:
            break
        result.append(word)
    return ' '.join(result)

def stream_text(text=str):
    for word in text.split():
        yield word + " "
        time.sleep(0.08)

def trim_message_list():
    if (st.session_state['model_output_tokens'] + 384) > st.session_state['max_context']:
        try:
            last_message = len(st.session_state['message_list'])
            st.session_state['message_list'].pop(last_message)
            last_message = len(st.session_state['message_list'])
            st.session_state['message_list'].pop(last_message)
        except:
            print("could not remove msgs from message_list")

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

def update_current_template(user_prompt=str, template_list=list, current_template=str):
    system_message = ""
    context_list = f"""Context List: {st.session_state['message_list']}"""
    for template in template_list:
        if template == current_template:
            system_message = """You are an AI who excells at being as helpful as possible to the users request. Please check the Context List for additional context."""

        default = f"""{system_message}
        {context_list}

        USER: {user_prompt}
        ASSISTANT:"""
        current_template = default




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

def update_chat_template(prompt=str, template_type=str):
    template = template_type

    if template_type == "chat_mistralx":
        sys_mistral = f"""You are an AI who excels at being as helpful as possible to the users request. Conversation Context: {st.session_state['message_list']}"""
        chat_mistral =f"""<|im_start|>system
        {sys_mistral}<|im_end|>
        <|im_start|>user
        {prompt}<|im_end|>
        <|im_start|>assistant"""
        template = chat_mistral

    if template_type == "chat_mistral":
        chat_mistral = f"""<s>[INST]You are an AI who excels at being as helpful as possible to the users request. 
        Conversation Context: {st.session_state['message_list']} 

        {prompt} [/INST]"""
        template = chat_mistral

    if template_type == "chat_mixtral_base":
        chat_mixtral_base = f"""[INST]You are my AI best friend and assistant who helps me solve any kind of problem, from coding to questions about life. Interact with the user in a casual and fun way but always answer more serious questions from the user in a appropriate way.
        Conversation Context: {st.session_state['message_list']} 

        {prompt} [/INST]"""
        template = chat_mixtral_base

    if template_type == 'code_deepseek':
        code_deepseek = f"""You are an AI programming assistant, specialized in explaining Python code by thinking step by step. Use the Context List below for conversation history.
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

def clear_vram(save_current_session=False):
    
    model_list = ['melo_model', 'speaker_ids', 'moondream', 'xtts_model', 'xtts_config', 'chat_model', 'speech_tt_model', 'image_pipe_turbo', 'image_pipe_sdxl', 'img2img_pipe']
    toggled_on_list = ['enable_voice', 'enable_voice_melo', 'enable_microphone', 'enable_sdxl', 'enable_sdxl_turbo', 'img2img_on', 'enable_vision']
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

def load_session():
    with open("llmon-py_state.pickle",'rb') as f:
        st.session_state = pickle.dump(f)

def save_session():
    clear_vram(save_current_session=st.session_state['save_session'])
    with open('llmon-py_state.pickle', 'wb') as f:
        pickle.dump(exclude_id(st.session_state), f)

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
                                'enable_sdxl': False,
                                'enable_vision': False, 
                                'enable_vision_deepseek': False}
    
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
    st.session_state['model_top_p'] = 0.75
    st.session_state['model_top_k'] = 120
    st.session_state['model_min_p'] = 0.08
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

def message_boop():
    if st.session_state['user_type'] == 'admin':
        message_boop = simpleaudio.WaveObject.from_wave_file("./llmonpy/chat_pop.wav")
        message_boop.play()

def check_user_type():
    disable_option = False
    if st.session_state['user_type'] == 'user_basic':
        disable_option = True
    return disable_option

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