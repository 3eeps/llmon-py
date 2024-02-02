# ./codespace/llmon/reconition.py
import streamlit as st
import os
from moondream import VisionEncoder, TextModel
from PIL import Image
from threading import Thread
from transformers import TextIteratorStreamer

image_ocr_model_path = "c:\codespace\moondream\moondream1"

def load_vision_encoder(enable_cpu=bool):
    st.session_state['vision_encoder'] = VisionEncoder(image_ocr_model_path, run_on=enable_cpu)
    st.session_state['text_model'] = TextModel(image_ocr_model_path, run_on=enable_cpu)

def generate_response(image_data, prompt=str):
    filename = "ocr_upload_image.png"
    with open(filename, 'wb') as file:
        file.write(image_data)
    image = Image.open("ocr_upload_image.png")

    image_embeds = st.session_state['vision_encoder'](image)
    streamer = TextIteratorStreamer(st.session_state['text_model'].tokenizer, skip_special_tokens=True)
    generation_kwargs = dict(image_embeds=image_embeds, question=prompt, streamer=streamer)
    thread = Thread(target=st.session_state['text_model'].answer_question, kwargs=generation_kwargs)
    thread.start()
    os.remove("ocr_upload_image.png")
    return streamer