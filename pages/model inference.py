# ./codespace/pages/model inference.py
import streamlit as st

st.set_page_config(page_title="model inference", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("🍋model inference")

import os
import time
import torch
import warnings
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts 
from llama_cpp import Llama
from pywhispercpp.model import Model
import sounddevice
from scipy.io.wavfile import write as write_wav
from llmonpy import llmonaid
from datetime import datetime
from cleantext import clean

if st.session_state['approved_login'] == True:
    if st.session_state['enable_voice']:
        st.markdown(f"with :orange[**{st.session_state['model_select']}**] :green[+ xttsv2]!")
    else:
        st.markdown(f"with :orange[**{st.session_state['model_select']}**]")

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
    param_name_list = ['model_temperature', 'model_top_p', 'model_top_k', 'model_min_p', 'repeat_penalty']
    param_name_disc_list = ['model temperature', 'model top p', 'model top k', 'model min p', 'repeat penalty']

    with st.sidebar:
        if st.session_state['enable_voice']:
            st.markdown(f"with :orange[**{st.session_state['model_select']}**] :green[+ xttsv2]!")
        else:
            st.markdown(f"with :orange[**{st.session_state['model_select']}**]")

        if st.button(label='clear model context', help="resets the current models conversation context. important when your `token count` may exceed the set 'max_context' value"):
            st.session_state['message_list'] = []
            print('context list empty')

        if st.button(label='download conversation'):
            output_file_name = datetime.now().strftime("%d-%m-%Y-%H-%M")
            file = open(f'session_messages_{output_file_name}.log', 'a')
            for msg in st.session_state['message_list']:
                line = clean(msg, no_emoji=True)
                file.writelines(f"{line}\n")
                file.writelines("\n")
            file.close()

        if st.session_state['enable_microphone']:
            st.markdown(f":red[*microphone enabled*]")

        st.session_state['model_temperature'] = st.text_input(label='temperature', value=st.session_state['model_temperature'])
        st.session_state['model_top_p'] = st.text_input(label='top p', value=st.session_state['model_top_p'] )
        st.session_state['model_top_k'] = st.text_input(label='top k', value=st.session_state['model_top_k'])
        st.session_state['model_min_p'] = st.text_input(label='min p', value=st.session_state['model_min_p'])
        st.session_state['repeat_penalty'] = st.text_input(label='repeat_penalty', value=st.session_state['repeat_penalty'])

        model_parameters = f"{st.session_state['model_temperature']}\n{st.session_state['model_top_p']}\n{st.session_state['model_top_k']}\n{st.session_state['model_min_p']}\n{st.session_state['repeat_penalty']}"
        st.download_button(label='save parameters', data=model_parameters, file_name='saved_params.txt')

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
        st.session_state['xtts_model'] = Xtts.init_from_config(st.session_state['xtts_config'])
        st.session_state['xtts_model'].load_checkpoint(st.session_state['xtts_config'], checkpoint_dir="./speech models/xtts")
        st.session_state['xtts_model'].cuda()

    if st.session_state['enable_voice']:
        gpt_cond_latent, speaker_embedding = st.session_state['xtts_model'].get_conditioning_latents(audio_path=[f"{chat_model_voice_path}"])
        warmup_tts = st.session_state['xtts_model'].inference_stream(text=warmup_string,
                                                                    language=language,
                                                                    gpt_cond_latent=gpt_cond_latent,
                                                                    speaker_embedding=speaker_embedding,
                                                                    stream_chunk_size=warmup_chunk_size)

    if 'speech_tt_model' not in st.session_state and st.session_state['enable_microphone']:
        llmonaid.popup_note(message='😎 lets get it stt model!')
        st.session_state.user_voice_prompt = None
        st.session_state['speech_tt_model'] = Model(models_dir=speech_model_path)

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

    llmonaid.memory_display()

    for message in st.session_state.messages:
        with st.chat_message(name=message["role"]):
            st.markdown(message["content"])

    input_message = str
    if st.session_state['enable_microphone']:
        input_message = f"Type a message to {st.session_state['char_name']}, or use the microphone by typing 'q'"
    else:
        input_message = f"Send a message to {st.session_state['char_name']}"
    
    user_name = 'user'
    model_name = st.session_state['char_name']
    if user_text_prompt:= st.chat_input(input_message):
        user_prompt = user_text_prompt
        if user_text_prompt == "q" and st.session_state['enable_microphone']:
            user_prompt = voice_to_text()

        st.session_state['message_list'].append(f"""{user_name}: {user_prompt}""")
        final_prompt = llmonaid.update_chat_template(prompt=user_prompt, template_type=st.session_state['template_select'])

        with st.chat_message(name="user", avatar='🙅'):
            st.markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        llmonaid.message_boop()

        llmonaid.popup_note(message='🍋 generating response...')
        llm_start = time.time()
        model_output = st.session_state["chat_model"](prompt=final_prompt,
                                                        repeat_penalty=float(st.session_state['repeat_penalty']),
                                                        max_tokens=st.session_state['max_context'], 
                                                        top_k=int(st.session_state['model_top_k']),
                                                        top_p=float(st.session_state['model_top_p']),
                                                        min_p=float(st.session_state['model_min_p']),
                                                        temperature=float(st.session_state['model_temperature']))
        model_response = f"{st.session_state['char_name']}: {model_output['choices'][0]['text']}"
        st.session_state['model_output_tokens'] = model_output['usage']['total_tokens']
        st.session_state['token_count'] = (f"token count: {model_output['usage']['total_tokens']}")
        st.session_state['response_time'] = (f"inferenced in: {int(time.time()-llm_start)} secs")

        st.session_state['message_list'].append(f"""You: {model_output['choices'][0]['text']}""")
        if model_output['usage']['total_tokens'] > (int(st.session_state['max_context'] - 512)):
            st.session_state['message_list'] = []

        with st.chat_message(name="assistant", avatar='🤖'):
            st.markdown(model_response)
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

    st.markdown("""
    <style>
    .big-font {
        font-size:10px !important;
        color:orange
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(f"""<p class="big-font">{st.session_state['token_count']} {st.session_state['response_time']}</p>""", unsafe_allow_html=True)

    print(st.session_state['model_output_tokens'])
else:
    st.image('./llmonpy/pie.png', caption='please login to continue')