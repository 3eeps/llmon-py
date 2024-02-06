# ./codespace/pages/model inference.py
import streamlit as st
from streamlit_extras.app_logo import add_logo 

st.set_page_config(page_title="model inference", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("🍋model inference")

if st.session_state['approved_login'] == True:
    if st.session_state['enable_voice']:
        st.markdown(f"with :orange[**{st.session_state['model_select']}**] :green[+ xttsv2]!")
    else:
        st.markdown(f"with :orange[**{st.session_state['model_select']}**]")
    add_logo("./llmonpy/pie.png", height=130)

    import os
    import time
    import torch
    import warnings
    import GPUtil as GPU
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts 
    from llama_cpp import Llama
    from pywhispercpp.model import Model
    import sounddevice
    from scipy.io.wavfile import write as write_wav
    from llmonpy import llmonaid
    import psutil

    warnings.filterwarnings('ignore')
    torch.set_num_threads(st.session_state['torch_audio_cores'])
    chat_model_path = f"./models/{st.session_state['model_select']}"
    chat_model_voice_path = f"./voices/{st.session_state['voice_select']}"
    language = 'en'
    stt_threads = 10
    speech_model_path = './speech models'
    warmup_string = 'hello there'
    warmup_chunk_size = 40
    sample_rate = 44100
    channels = 2

    # try stderr for console output

    with st.sidebar:
        st.session_state['notepad'] = st.text_area(label='notepad', label_visibility='collapsed', value=st.session_state['notepad'])
        if st.session_state['enable_microphone']:
            st.markdown(f":red[*microphone enabled*]")
        st.write('model parameters:')
        st.session_state['model_temperature'] = st.text_input(label='temperature', value=0.8)
        st.session_state['model_top_p'] = st.text_input(label='top p', value=0.95)
        st.session_state['model_top_k'] = st.text_input(label='top k', value=40)
        st.session_state['repeat_penalty'] = st.text_input(label='repeat_penalty', value=1.1)

    def voice_to_text():
        rec_user_voice = sounddevice.rec(st.session_state['user_audio_length'] * sample_rate, samplerate=sample_rate, channels=channels)
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

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if 'xtts_model' not in st.session_state and st.session_state['enable_voice']:
        llmonaid.popup_note(message='😤 hyping up tts model...')
        st.session_state['xtts_config'] = XttsConfig()
        st.session_state['xtts_config'].load_json("./speech models/xtts/config.json")
        st.session_state['xtts_model'] = Xtts.init_from_config(st.session_state['config'])
        st.session_state['xtts_model'].load_checkpoint(st.session_state['config'], checkpoint_dir="./speech models/xtts")
        st.session_state['xtts_model'].cuda()

    if 'speech_tt_model' not in st.session_state and st.session_state['enable_microphone']:
        llmonaid.popup_note(message='😎 lets get it stt model!')
        st.session_state.user_voice_prompt = None
        st.session_state['speech_tt_model'] = Model(models_dir=speech_model_path)

    if st.session_state['enable_voice']:
        gpt_cond_latent, speaker_embedding = st.session_state.model.get_conditioning_latents(audio_path=[f"{chat_model_voice_path}"])
        warmup_tts = st.session_state['xtts_model'].inference_stream(text=warmup_string,
                                                                language=language,
                                                                gpt_cond_latent=gpt_cond_latent,
                                                                speaker_embedding=speaker_embedding,
                                                                stream_chunk_size=warmup_chunk_size)

    if "chat_model" not in st.session_state:
        llmonaid.popup_note(message='😴 waking up chat model...')
        st.session_state[f"chat_model"] = Llama(model_path=chat_model_path,
                                                n_batch=st.session_state['batch_size'],
                                                n_threads=st.session_state['cpu_core_count'],
                                                n_threads_batch=st.session_state['cpu_batch_count'],
                                                n_gpu_layers=int(st.session_state['gpu_layer_count']),
                                                n_ctx=st.session_state['max_context'],
                                                verbose=st.session_state['verbose_chat'])
        warmup_chat = st.session_state[f"chat_model"](prompt=warmup_string)

    GPUs = GPU.getGPUs()
    gpu = GPUs[0]
    mem_total = 100 / gpu.memoryTotal
    mem_used = 100 / int(gpu.memoryUsed)
    total_ = mem_total / mem_used
    if  total_> 85.0:
        st.progress((100 / gpu.memoryTotal) / (100 / int(gpu.memoryUsed)), "vram :red[{0:.0f}]/{1:.0f}gb".format(gpu.memoryUsed, gpu.memoryTotal))
    else:
        st.progress((100 / gpu.memoryTotal) / (100 / int(gpu.memoryUsed)), "vram :green[{0:.0f}]/{1:.0f}gb".format(gpu.memoryUsed, gpu.memoryTotal))

    memory_usage = psutil.virtual_memory()
    if memory_usage.percent > 85.0:
        st.progress((memory_usage.percent / 100), f'system memory usage: :red{memory_usage.percent}%]')
    else:
        st.progress((memory_usage.percent / 100), f'system memory usage: :green[{memory_usage.percent}%]')

    for message in st.session_state.messages:
        with st.chat_message(name=message["role"]):
            st.markdown(message["content"])

    input_message = str
    if st.session_state['enable_microphone']:
        input_message = f"Type a message to {st.session_state['char_name']}, or use the microphone by typing 'q'"
    else:
        input_message = f"Send a message to {st.session_state['char_name']}"

    if user_text_prompt:= st.chat_input(input_message):
        #too
        #
        #
        #
        #
        #
        #
        #
        llmonaid.save_session()

        user_prompt = user_text_prompt
        if user_text_prompt == "q" and st.session_state['enable_microphone']:
            user_prompt = voice_to_text()
        final_prompt = llmonaid.update_chat_template(prompt=user_prompt, template_type=st.session_state['template_select'])

        with st.chat_message(name="user", avatar='🙅'):
            st.markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        llmonaid.message_boop()

        llmonaid.popup_note(message='🍋 generating response...')
        llm_start = time.time()

        model_output = st.session_state["chat_model"](prompt=final_prompt,
                                                       max_tokens=st.session_state['max_context'], 
                                                       top_k=int(st.session_state['model_top_k']),
                                                       top_p=float(st.session_state['model_top_p']),
                                                       temperature=float(st.session_state['model_temperature']),
                                                       repeat_penalty=float(st.session_state['repeat_penalty']))
        model_response = f"{st.session_state['char_name']}: {model_output['choices'][0]['text']}"

        if st.session_state['verbose_chat']:
            with st.sidebar:
                st.write(f"last message token count: {model_output['usage']['total_tokens']}")
                st.write(f"elapsed time : {int(time.time()-llm_start)} secs")
        
        with st.chat_message(name="assistant", avatar='🤖'):
            st.write_stream(llmonaid.stream_text(text=model_response))
        st.session_state.messages.append({"role": "assistant", "content": model_response})
        llmonaid.message_boop()

        if st.session_state['enable_voice']:
            tts_start = time.time()
            if st.session_state['enable_code_voice']:
                get_paragraph = llmonaid.get_paragraph_before_code(sentence=model_output['choices'][0]['text'], stop_word='```')
                paragraph = st.session_state['xtts_model'].inference_stream(text=get_paragraph,
                                                                       language=language,
                                                                       gpt_cond_latent=gpt_cond_latent,
                                                                       speaker_embedding=speaker_embedding,
                                                                       stream_chunk_size=st.session_state['stream_chunk_size'])
                llmonaid.wav_by_chunk(chunks=paragraph, token_count=int(model_output['usage']['total_tokens']))
            else:
                chunk_inference = st.session_state['xtts_model'].inference_stream(text=model_output['choices'][0]['text'],
                                                                              language=language,
                                                                              gpt_cond_latent=gpt_cond_latent,
                                                                              speaker_embedding=speaker_embedding,
                                                                              stream_chunk_size=st.session_state['stream_chunk_size'],
                                                                              enable_text_splitting=True)
                llmonaid.wav_by_chunk(chunks=chunk_inference, token_count=int(model_output['usage']['total_tokens']))
else:
    st.image('./llmonpy/pie.png', caption='please login to continue')