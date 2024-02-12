# ./codespace/llmon/generation.py
import streamlit as st
import time
import torch
from PIL import Image
from diffusers import DiffusionPipeline, AutoPipelineForText2Image, AutoPipelineForImage2Image
from datetime import datetime
from io import BytesIO

def load_sdxl(lora_path="loras", lora_name=str):
    st.session_state['image_pipe_sdxl'] = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to(0)
    st.session_state['image_pipe_sdxl'].load_lora_weights(pretrained_model_name_or_path_or_dict=lora_path, weight_name=lora_name)
    st.session_state['image_pipe_sdxl'].enable_vae_slicing()
    st.session_state['image_pipe_sdxl'].enable_model_cpu_offload()

def load_sdxl_turbo():
    st.session_state['image_pipe_turbo'] = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to('cuda')

def load_turbo_img2img(): 
    st.session_state['img2img_pipe'] = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to('cuda')

def create_image_sdxl(prompt=str, steps=50, iterations=1):
    while iterations:
        image = st.session_state['image_pipe_sdxl'](prompt=prompt, num_inference_steps=steps).images[0]
        output_file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        image.save(f"{output_file_name}-{iterations}.png")
        st.session_state['sdxl_image_list'].append(f"{output_file_name}-{iterations}.png")
        iterations -= 1
    return st.session_state['sdxl_image_list']

def create_image_turbo(prompt=str, steps=1):
    image = st.session_state['image_pipe_turbo'](prompt=prompt, num_inference_steps=steps, guidance_scale=0.0).images[0]
    image.save('image_turbo.png')
    time.sleep(0.2)
    st.image('image_turbo.png')

def create_image_img2img(prompt=str, steps=2):
    st.session_state['img2img_pipe'] = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
    init_image = Image.open(BytesIO(st.session_state.bytes_data))
    final_image = init_image.resize((512, 512))
    image = st.session_state['img2img_pipe'](prompt, image=final_image, num_inference_steps=steps, strength=0.5, guidance_scale=0.0).images[0]
    output_file_name = datetime.now().strftime("img2img-%d-%m-%Y-%H-%M-%S")
    image.save(f"{output_file_name}.png")
    send_image = f"{output_file_name}.png"
    return send_image