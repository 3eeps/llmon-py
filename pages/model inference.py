# ./codespace/pages/model inference.py
import streamlit as st

st.set_page_config(page_title="model inference", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("🍋model inference")

import time
from llmonpy import llmonaid

if st.session_state['approved_login'] == True:
    chat_model_path = f"./models/{st.session_state['model_select']}"
    chat_model_voice_path = f"./voices/{st.session_state['voice_select']}"
    language = 'EN'
    warmup_string = 'hello there'
    speech_model_path = './speech models'
    warmup_chunk_size = 40
    sample_rate = 44100
    param_name_list = ['model_temperature', 'model_top_p', 'model_top_k', 'model_min_p', 'repeat_penalty']
    param_name_disc_list = ['model temperature', 'model top p', 'model top k', 'model min p', 'repeat penalty']

    model_select_list  = ['chat', 'cloned tts', 'vision', 'sdxl']
    # sdxl 1.0, turbo, and img2img can be 'types' that are checkbox selectable

    with st.sidebar:

        # drop down menu to 'open/display' settings for the models you have enabled (or just a list of them all)
        #
        ##
        #
        #
        #


        llmonaid.memory_display()
        if st.button(label='refresh page'):
            pass

        model_selected_settings = st.selectbox(label='model setting menu', options=model_select_list)
        if model_selected_settings == 'chat':

            # rename label based on selection
            st.checkbox('load model', value=False)
            if st.checkbox:
                pass
                #load  {model_select_list} model
            elif st.checkbox==False:
                pass
                #unload {model_select_list} model (this model from vram only)

            if st.session_state['enable_microphone']:
                st.markdown(f":red[*microphone enabled*]")

            if st.session_state['enable_voice'] or st.session_state['enable_voice_melo']:
                st.markdown(f":orange[**{st.session_state['model_select']}**] :green[+ tts]")
            else:
                st.markdown(f":orange[**{st.session_state['model_select']}**]")

            st.markdown(f":violet[context limit:] :orange[{st.session_state['model_output_tokens']}/{st.session_state['max_context']}]")
            st.markdown(f":violet[{st.session_state['response_time']}]")

            if st.button(label='clear context'):
                st.session_state['message_list'] = []
                del st.session_state.messages
                st.session_state['model_output_tokens'] = 0
                print('context cleared')

            st.session_state['model_temperature'] = st.text_input(label=':orange[temperature]', value=st.session_state['model_temperature'])
            st.session_state['model_top_p'] = st.text_input(label=':orange[top p]', value=st.session_state['model_top_p'] )
            st.session_state['model_top_k'] = st.text_input(label=':orange[top k]', value=st.session_state['model_top_k'])
            st.session_state['model_min_p'] = st.text_input(label=':orange[min p]', value=st.session_state['model_min_p'])
            st.session_state['repeat_penalty'] = st.text_input(label=':orange[repeat_penalty]', value=st.session_state['repeat_penalty'])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if 'melo_model' not in st.session_state and st.session_state['enable_voice_melo']:
        from melo_tts.api import TTS
        llmonaid.popup_note(message='😤 ahh yeah melo!')
        st.session_state['melo_model'] = TTS(language=language, device='cuda')
        st.session_state['speaker_ids'] = st.session_state['melo_model'].hps.data.spk2id

    if 'xtts_model' not in st.session_state and st.session_state['enable_voice']:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
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
        from pywhispercpp.model import Model
        llmonaid.popup_note(message='😎 lets grab the mic!')
        st.session_state.user_voice_prompt = None
        st.session_state['speech_tt_model'] = Model(models_dir=speech_model_path, n_threads=10)

    if "chat_model" not in st.session_state:
        if st.session_state['loader_type'] == 'llama-cpp-python':
            from llama_cpp import Llama
            llmonaid.popup_note(message='😴 hello chat model!')
            st.session_state["chat_model"] = Llama(model_path=chat_model_path,
                                                    n_batch=st.session_state['batch_size'],
                                                    n_threads=st.session_state['cpu_core_count'],
                                                    n_threads_batch=st.session_state['cpu_batch_count'],
                                                    n_gpu_layers=int(st.session_state['gpu_layer_count']),
                                                    n_ctx=st.session_state['max_context'],
                                                    verbose=True)
        if st.session_state['loader_type'] == 'exllamav2':
            from exllamav2 import *
            from exllamav2.generator import *
            import sys
            st.session_state.config = ExLlamaV2Config()
            st.session_state.config.model_dir = "C:\codespace\models\dolphin-2.6-mistral-7b-dpo-laser-6.0bpw-h6-exl2"
            st.session_state.config.prepare()

            st.session_state["chat_model"] = ExLlamaV2(st.session_state.config)
            st.session_state.cache = ExLlamaV2Cache(st.session_state["chat_model"], lazy=True)
            st.session_state["chat_model"].load_autosplit(st.session_state.cache)

            st.session_state.tokenizer = ExLlamaV2Tokenizer(st.session_state.config)
            st.session_state.generator = ExLlamaV2StreamingGenerator(st.session_state["chat_model"], st.session_state.cache, st.session_state.tokenizer)
            st.session_state.generator.set_stop_conditions([st.session_state.tokenizer.eos_token_id])
            st.session_state.gen_settings = ExLlamaV2Sampler.Settings()

    for message in st.session_state.messages:
        with st.chat_message(name=message["role"]):
            st.markdown(message["content"])
    
    llmonaid.trim_message_list()
    if user_text_prompt:= st.chat_input():
        user_prompt = user_text_prompt

        # yoooo just @ any model you need and have it do everything in the main chat window!!




        # do this for real later, just get the first word
        only_use_vision_model = False
        if user_text_prompt == "@vision" and st.session_state['enable_vision'] and st.session_state.byte_data:
            from llmonpy import reconition
            llmonaid.popup_note(message='🌒 describe me to the moon...')
            reconition.load_vision_encoder()
            only_use_vision_model = True
            pass
            



        if user_text_prompt == "q" and st.session_state['enable_microphone']:
            user_prompt = llmonaid.voice_to_text()

        st.session_state['message_list'].append(f"""user: {user_prompt}""")
        final_prompt = llmonaid.update_chat_template(prompt=user_prompt, template_type=st.session_state['template_select'])

        with st.chat_message(name="user"):
            st.markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        llmonaid.popup_note(message='🍋 generating response...')
        llm_start = time.time()


        # integrate moondream/ image etc!!!

        # checkbox to unload/load stuff!!!! aka check: fast more mem    uncheck: slow, less mem

        # store images in an numpy array of somesorts to keep messages in the proper order/location in message history when user scrolls up

        if user_text_prompt == "@vision" and st.session_state['enable_vision'] and st.session_state.byte_data and only_use_vision_model:
            # remove "@vision" from the prompt!!!

            model_response = reconition.generate_response(prompt=user_text_prompt, image_data=st.session_state.byte_data)
            only_use_vision_model = False

        model_output = st.session_state["chat_model"](prompt=final_prompt,
                                                        repeat_penalty=float(st.session_state['repeat_penalty']),
                                                        max_tokens=st.session_state['max_context'], 
                                                        top_k=int(st.session_state['model_top_k']),
                                                        top_p=float(st.session_state['model_top_p']),
                                                        min_p=float(st.session_state['model_min_p']),
                                                        temperature=float(st.session_state['model_temperature']))
        model_response = model_output['choices'][0]['text']
        st.session_state['model_output_tokens'] = model_output['usage']['total_tokens']
        st.session_state['response_time'] = (f":violet[last inference:] {int(time.time()-llm_start)} secs")

        st.session_state['message_list'].append(f"""You: {model_output['choices'][0]['text']}""")

        if st.session_state['enable_voice_melo']:
            llmonaid.melo_gen_message(message=model_response, token_count=model_output['usage']['completion_tokens'])

        with st.chat_message(name="assistant"):
            st.markdown(model_response)
        st.session_state.messages.append({"role": "assistant", "content": model_response})
    
        if st.session_state['enable_voice']:
            chunk_inference = st.session_state['xtts_model'].inference_stream(text=model_output['choices'][0]['text'],
                                                                            language=language,
                                                                            gpt_cond_latent=gpt_cond_latent,
                                                                            speaker_embedding=speaker_embedding,
                                                                            stream_chunk_size=int(st.session_state['stream_chunk_size']),
                                                                            enable_text_splitting=True)
            llmonaid.wav_by_chunk(chunks=chunk_inference, token_count=int(model_output['usage']['completion_tokens']))
else:
    st.image('./llmonpy/pie.png', caption='please login to continue')