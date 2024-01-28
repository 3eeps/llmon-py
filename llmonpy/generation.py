# ./codespace/llmon/generation.py
import streamlit as st
import time
import torch
from diffusers import DiffusionPipeline, AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image
from datetime import datetime

def load_sdxl(default_device='cuda', lora_path="./loras", lora_name=str):
    device = default_device
    if st.session_state['sdxl_cpu']:
        device = 'cpu'
    st.session_state.image_pipe_sdxl = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to(device)
    st.session_state.image_pipe_sdxl.load_lora_weights(pretrained_model_name_or_path_or_dict=lora_path, weight_name=lora_name)
    st.session_state.image_pipe_sdxl.enable_vae_slicing()
    st.session_state.image_pipe_sdxl.enable_model_cpu_offload()

def load_sdxl_turbo(default_device='cuda'):
    device = default_device
    if st.session_state['turbo_cpu']:
        device = 'cpu'
    st.session_state.image_pipe_turbo = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to(device)

def load_turbo_img2img(default_device='cuda'):
    device = default_device
    if st.session_state['turbo_cpu']:
        device = 'cpu'   
    st.session_state['img2img_pipe'] = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to(device)

def create_image_sdxl(prompt=str, steps=50, iterations=1):
    image_list = []
    while iterations:     
        image = st.session_state.image_pipe_sdxl(prompt=prompt, num_inference_steps=steps).images[0]
        output_file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        image.save(f"{output_file_name}-{iterations}.png")
        image_list.append(f"{output_file_name}-{iterations}.png")
        iterations -= 1
        if len(image_list) > 32:
            image_list = []
    return image_list

def create_image_turbo(prompt=str, steps=1):
    image = st.session_state.image_pipe_turbo(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0).images[0]
    image.save('image_turbo.png')
    time.sleep(0.2)
    st.image('image_turbo.png')

def create_image_img2img(from_image, prompt=str, steps=1, iterations=1):
    image_list = []
    while iterations:
        init_image = load_image(f"./images/{from_image}").resize((512, 512))
        image = st.session_state['img2img_pipe'](prompt=prompt, image=init_image, num_inference_steps=steps, strength=0.5, guidance_scale=0.0).images[0]
        output_file_name = datetime.now().strftime("img2img-%d-%m-%Y-%H-%M-%S")
        image.save(f"img2img-{output_file_name}.png")
        image_list.append(f"img2img-{output_file_name}.png")
        iterations -= 1
        if len(image_list) > 32:
            image_list = []
    return image_list
