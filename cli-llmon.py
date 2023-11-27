# ./codespace/llmon.py

import os
import warnings

import torch
from TTS.api import TTS 
from llama_cpp import Llama
from pywhispercpp.model import Model

import readchar

import simpleaudio
import sounddevice 
from scipy.io.wavfile import write

import GPUtil
from threading import Thread
import time

warnings.filterwarnings("ignore")

def display_logo():
    os.system("cls")
    color_logo = f"\33[{93}m".format(code=93)
    print(f"""{color_logo}
                        llmon-py
⠀⠀⠀⠀⠀⢀⣀⣠⣤⣴⣶⡶⢿⣿⣿⣿⠿⠿⠿⠿⠟⠛⢋⣁⣤⡴⠂⣠⡆⠀
⠀⠀⠀⠀⠈⠙⠻⢿⣿⣿⣿⣶⣤⣤⣤⣤⣤⣴⣶⣶⣿⣿⣿⡿⠋⣠⣾⣿⠁⠀
⠀⠀⠀⠀⠀⢀⣴⣤⣄⡉⠛⠻⠿⠿⣿⣿⣿⣿⡿⠿⠟⠋⣁⣤⣾⣿⣿⣿⠀⠀
⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣶⣶⣤⣤⣤⣤⣤⣤⣶⣾⣿⣿⣿⣿⣿⣿⣿⡇⠀
⠀⠀⠀⣰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀
⠀⠀⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⠀
⠀⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⢸⡟⢸⡟⠀⠀
⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣷⡿⢿⡿⠁⠀⠀
⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⢁⣴⠟⢀⣾⠃⠀⠀⠀
⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠛⣉⣿⠿⣿⣶⡟⠁⠀⠀⠀⠀
⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⣿⣏⣸⡿⢿⣯⣠⣴⠿⠋⠀⠀⠀⠀⠀⠀
⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⠿⠶⣾⣿⣉⣡⣤⣿⠿⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⢸⣿⣿⣿⣿⡿⠿⠿⠿⠶⠾⠛⠛⠛⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠈⠉⠉⠉""")

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)
            os.system("cls")

    def stop(self):
        self.stopped = True

class Client():
    def __init__(self):
        self.gpu_layers = 20
        self.model_ctx = 4096
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.verbose_output = False

        self.rec_sample_freq = 44100
        self.rec_seconds = 10
        self.rec_channels = 2
        self.user_file_count = 0
        self.model_file_count = 0
        self.model_list_count = 1
        self.clone_voice_file = './voices/artbell.wav'
        self.log_file = './chat_history.log'

        self.new_session = True
        self.chat_template = str                                                                                                                                                                                                                            
        self.model_path = "./models"
        self.model_list = []
        self.path_list = os.scandir(self.model_path)

        self.green = 32
        self.yellow = 93
        self.red = 91
        self.grey  = 90
        self.white = 37

        self.text_ts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device=self.device)
        self.speech_tt_model = Model('./stt/ggml-tiny.bin')

    def log_chat(self, user_message=str, model_message=str):
        write_type = 'w'
        if self.new_session == False:
            write_type = 'a'
        self.new_session = False

        output_line = [f'USER: {user_message}', f'ASSISTANT: {model_message}']
        with open(self.log_file, f'{write_type}') as output_file:
            for message in output_line:
                output_file.write(message)
                output_file.write('\n')
        output_file.close()

    def color(self, color_code):
        return f"\33[{color_code}m".format(code=color_code)

    def create_chat_wav(self, chat_model_text=str):
        self.model_file_count = self.model_file_count + 1
        self.text_ts_model.tts_to_file(text=chat_model_text, speaker_wav=self.clone_voice_file, file_path=f'./model_output-{self.model_file_count}.wav', language="en")

    def play_wav(self):
        wav_filename = f'./model_output-{self.model_file_count}.wav'
        wav_object = simpleaudio.WaveObject.from_wave_file(wav_filename)
        play_audio = wav_object.play()
        play_audio.wait_done()

    def voice_to_text(self):
        text_data = []
        user_voice_data = self.speech_tt_model.transcribe(f'user_output-{self.user_file_count}.wav')
        for voice in user_voice_data:        
            text_data.append(voice.text)
        combined_text = ' '.join(text_data)
        print(combined_text)
        return combined_text
    
    def record_user(self):
        self.user_file_count = self.user_file_count + 1
        rec_user_voice = sounddevice.rec(int(self.rec_seconds * self.rec_sample_freq), samplerate=self.rec_sample_freq, channels=self.rec_channels)
        sounddevice.wait()
        write(filename=f'user_output-{self.user_file_count}.wav', rate=self.rec_sample_freq, data=rec_user_voice)
                                 
    def update_chat_template(self, prompt=str):

        instruction = f"""### Instruction: 
        You are Art Bell, the radio host from Coast to Coast AM. Your guest tonight claims they are a former scientist from the Black Mesa research facility.

        GUEST: {prompt}
        ### Response:"""
        
        user_assist = f"""USER: You are Art Bell, the radio host from the late-night talk show, Coast to Coast AM. Your guest tonight claims to be a theoretical physicist with a remarkable story. He claims to have worked at the top-secret Black Mesa research facility, where he witnessed an unimaginable disaster.

        GUEST: {prompt}
        ASSISTANT:"""

        vicuna = f"""You are a virtual assistant with expertise in extracting information from job offers. Your primary task is to respond to user-submitted job offers by extracting key details such as the job title, location, required experience, level of education, type of contract, field, required skills, and salary. You should provide your responses in JSON format, and all responses must be in French.

        User: {prompt}
        ASSISTANT:"""

        template_type = user_assist
        return template_type

    def select_models(self):
        os.system("cls")
        print(self.color(self.yellow) + "llmon-py" + self.color(self.grey) + self.model_path, "\n")
        for model_file in self.path_list:
            if model_file.is_file():
                print(self.color(self.green) + f"{self.model_list_count}.) " + self.color(self.yellow) + f"{model_file.name}")
                self.model_list.append(model_file.name)
            self.model_list_count = self.model_list_count + 1
        self.path_list.close()

        print(self.color(self.grey))
        model_intstr = readchar.readkey()
        model_int = int(model_intstr)
        chat_model_filename = self.model_list[model_int - 1]
        os.system("cls")
        return chat_model_filename

    def start(self):
        running_client = True
        chat_model_loaded = self.select_models()
        print (self.color(self.yellow) + "llmon-py " + self.color(self.grey) + "loaded " + self.color(self.white) + f"{chat_model_loaded}")
        chat_model = Llama(
            model_path=f'./models/{chat_model_loaded}',  
            n_gpu_layers = self.gpu_layers, 
            n_ctx = self.model_ctx, 
            verbose = self.verbose_output)
      
        while running_client:
            user_text_prompt = input(self.color(self.yellow) + "user>>> ")
            if user_text_prompt == '':
                print(f"Recording for {self.rec_seconds}secs:")
                self.record_user()
                user_voice_prompt = self.voice_to_text()
                prompt = self.update_chat_template(user_voice_prompt)
            else:  
                prompt = self.update_chat_template(user_text_prompt)

            #monitor = Monitor(5)
            chat_model_output = chat_model(prompt=prompt) 
            self.log_chat(user_message=user_text_prompt, model_message=chat_model_output['choices'][0]['text'])
            print(self.color(self.green))
            self.create_chat_wav(chat_model_output['choices'][0]['text'])
            #monitor.stop()
            self.play_wav()

if __name__ == "__main__":
    display_logo()
    client = Client()
    client.start()
