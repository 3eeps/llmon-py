# ./codespace/llmonpy/reconition.py
import streamlit as st
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_vision_encoder():
    model_id = "moondream2"
    revision = "2024-03-05"
    st.session_state['moondream'] = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision)
    st.session_state.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

def generate_response(image_data, prompt=str):
    filename = "ocr_upload_image.png"
    with open(filename, 'wb') as file:
        file.write(image_data)

    image = Image.open(filename)
    enc_image = st.session_state['moondream'].encode_image(image)
    return st.session_state['moondream'].answer_question(enc_image, prompt, st.session_state.tokenizer)