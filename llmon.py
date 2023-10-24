#/codespace/llmon.py

import os
from TTS.api import TTS 
from llama_cpp import Llama
from pywhispercpp.model import Model

import simpleaudio
import sounddevice 
from scipy.io.wavfile import write

# client app
class Client():
    def __init__(self):
        self.rec_sample_freq = 44100
        self.rec_seconds = 5
        self.rec_channels = 2
        self.test_output = './voices/aftertest.wav'
        self.user_file_count = 0
        self.rec_user_voice_file = f'user_chatout-{self.user_file_count}.wav'
        self.stt_model = Model('./models/stt/ggml-tiny.bin')

        self.file_count = 0
        self.new_session = True
        self.model_path = "./models"
        self.model_list = []
        self.model_count = 0
        self.path_list = os.scandir(self.model_path)

        self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v1").cuda()

    def log_chat(self, user_message, model_message):
        if self.new_session == False:
            write_type = 'a'
        else:
            write_type = 'w'
        self.new_session = False

        messages_to_log = [f'USER: {user_message}', f'model_output{self.file_count}.wav: {model_message}']
        with open('chat_history.log', f'{write_type}') as output:
            for line in messages_to_log:
                output.write(line)
                output.write('\n')
        output.close()

    def color(self, color_code):
        return f"\33[{color_code}m".format(code=color_code)

    def create_chat_wav(self, chat_model_text=str):
        self.file_count = self.file_count + 1
        self.tts_model.tts_to_file(text=chat_model_text, speaker_wav='./voices/redguard0.wav', file_path=f'./model_output{self.file_count}.wav', language="en")

    def play_wav(self):
        wav_filename = f'./model_output{self.file_count}.wav'
        wav_object = simpleaudio.WaveObject.from_wave_file(wav_filename)
        play_audio = wav_object.play()
        play_audio.wait_done()

    def voice_to_text(self):
        store_text = []
        user_voice_data = self.stt_model.transcribe(f'./{self.test_output}', speed_up=False)
        for voice in user_voice_data:        
            store_text.append(voice.text)
        combined_text = ' '.join(store_text)
        print(combined_text)
        return combined_text
    
    def record_user(self):
        self.user_file_count = self.user_file_count + 1
        rec_user_voice = sounddevice.rec(int(self.rec_seconds * self.rec_sample_freq), samplerate=self.rec_sample_freq, channels=self.rec_channels)
        sounddevice.wait()
        write(filename=self.rec_user_voice_file, rate=self.rec_sample_freq, data=rec_user_voice)
                                 
    def update_chat_template(self, prompt):
        chatml = f"""system
        You are a scientist from the Half-Life game that escaped the Black Mesa incident. You love to talk about it.
        user
        {prompt}
        assistant"""

        vicuna = f"""You are a scientist from the Half-Life game that escaped the Black Mesa incident. You love to talk about it.

        User: {prompt}
        ASSISTANT:"""
        template_type = vicuna
        return template_type

    def select_character(self):
        # pick character profile for chat (template + .wav file)
        pass

    def select_models(self):
        os.system("cls")
        print(self.color(93) + "llmon-py" + self.color(90) + self.model_path, "\n")
        for model_file in self.path_list:
            self.model_count = self.model_count + 1
            if model_file.is_file():
                print(self.color(32) + f"{self.model_count}.) " + self.color(93) + f"{model_file.name}")
                self.model_list.append(model_file.name)
        self.path_list.close()

        print(self.color(90))
        model_int = int(input('#>> '))
        chat_model_filename = self.model_list[model_int - 1]
        os.system("cls")
        return chat_model_filename

    def start(self):
        running_client = True
        chat_model_loaded = self.select_models()
        print (self.color(93) + "llmon-py " + self.color(90) + "using " + self.color(37) + f"{chat_model_loaded}")
        chat_model = Llama(model_path=f'./models/{chat_model_loaded}', n_ctx=2048, verbose=False, chat_format="vicuna")
      
        while running_client:
            user_prompt = input(self.color(34) + "user>>> ")
            if user_prompt == 'rec':
                #self.record_user()  
                text = self.voice_to_text()
                prompt = self.update_chat_template(prompt=text)
            else:
                prompt = self.update_chat_template(prompt=user_prompt)

            chat_model_output = chat_model(prompt=prompt) 
            self.log_chat(user_message=user_prompt, model_message=chat_model_output['choices'][0]['text'])
            print(self.color(31))
            self.create_chat_wav(chat_model_output['choices'][0]['text'])            
            self.play_wav()

if __name__ == "__main__":
    client = Client()
    client.start()
