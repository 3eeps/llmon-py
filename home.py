# ./codespace/llmon-gui.py

import os
import time
import warnings
import streamlit as st
from threading import Thread

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts 
from llama_cpp import Llama
from pywhispercpp.model import Model

import torchaudio
import simpleaudio
import sounddevice 
from scipy.io.wavfile import write

st.title("🍋llmon-py")
warnings.filterwarnings("ignore")