import streamlit as st
import numpy as np
import librosa
import sounddevice as sd
import queue
import threading
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

def load_xtts():
    print("Loading xtts model...")
    st.session_state['xtts_config'] = XttsConfig()
    st.session_state['xtts_config'].load_json("./speech models/xtts/config.json")
    st.session_state['xtts_model'] = Xtts.init_from_config(st.session_state['xtts_config'])
    st.session_state['xtts_model'].load_checkpoint(st.session_state['xtts_config'], checkpoint_dir="./speech models/xtts", use_deepspeed=False)
    st.session_state['xtts_model'].cuda()
    st.session_state['gpt_cond_latent'], st.session_state['speaker_embedding'] = st.session_state['xtts_model'].get_conditioning_latents(audio_path=["./voices/redguard.wav"])
    print ('loading xtts: done')
    
q = queue.Queue()
event = threading.Event()

def buffer_audio_data(audio_data, target_size=2048):
    """Buffer and yield audio data chunks of a specific size."""
    buffer = np.array([], dtype=np.float32)
    for chunk in audio_data:
        if chunk.ndim != 1:
            chunk = chunk.reshape(-1)
        buffer = np.concatenate((buffer, chunk))
        while len(buffer) >= target_size:
            yield buffer[:target_size]
            buffer = buffer[target_size:]
    if len(buffer) > 0:
        yield buffer

def callback(outdata, frames, time, status):
    if status.output_underflow:
        print('Output underflow: increase buffer size?')
        raise sd.CallbackAbort
    try:
        data = q.get_nowait()
    except queue.Empty:
        print('Buffer is empty: increase buffer size?')
        raise sd.CallbackAbort
    reshaped_data = data.reshape(-1, 1)
    if len(reshaped_data) < len(outdata):
        outdata[:len(reshaped_data)] = reshaped_data
        outdata[len(reshaped_data):] = 0
    else: 
        outdata[:] = reshaped_data

def generate_speech(text, language="en"):
    chunks = st.session_state['xtts_model'].inference_stream(text, language, st.session_state['gpt_cond_latent'], st.session_state['speaker_embedding'], enable_text_splitting=True)
    for chunk in chunks:
        chunk = chunk.cpu().numpy().astype(np.float32)
        chunk = librosa.resample(chunk, orig_sr=24000, target_sr=48000)
        chunk = chunk.flatten()
        for buffered_chunk in buffer_audio_data([chunk]):
            q.put(buffered_chunk, block=True)

def play_back_speech(prompt=str):
    try:
        stream = sd.OutputStream(samplerate=48000, blocksize=2048, channels=1, callback=callback, finished_callback=event.set)
        with stream:
            for _ in range(256):
                dummy = np.zeros((2048, ), dtype=np.float32)
                q.put_nowait(dummy)
            generate_speech(prompt)
            event.wait()
    except KeyboardInterrupt:
        print('Interrupted by user')