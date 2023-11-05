# ./codespace/llmon.py

import os
from TTS.api import TTS 
from llama_cpp import Llama
from pywhispercpp.model import Model
#from nltk.tokenize import sent_tokenize
#import rssnews

import simpleaudio
import sounddevice 
from scipy.io.wavfile import write

class Client():
    def __init__(self):
        self.rec_sample_freq = 44100
        self.rec_seconds = 8
        self.rec_channels = 2
        self.user_file_count = 0
        self.rec_user_voice_file = f'user_chatout-{self.user_file_count}.wav'
        self.stt_model = Model('./stt/ggml-tiny.bin')
        self.clone_voice_file = 'redguard0.wav'
        self.log_file = 'chat_history.log'

        self.chat_template = str                                                                                                                                                                                                                            
        self.file_count = 0
        self.new_session = True
        self.model_path = "./models"
        self.model_list = []
        self.model_count = 1
        self.path_list = os.scandir(self.model_path)

        self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v1").cuda()

        self.gpu_layers = 22
        self.model_ctx = 4096
        self.gpu_device = 0
        self.chat_format = "chatml"
        self.verbose_output = False

        self.green = 32
        self.yellow = 93
        self.red = 91
        self.grey  = 90
        self.white = 37

    def log_chat(self, user_message=str, model_message=str):
        write_type = 'w'
        if self.new_session == False:
            write_type = 'a'
        self.new_session = False

        output_line = [f'USER: {user_message}', f'model_output{self.file_count}.wav: {model_message}']
        with open(self.log_file, f'{write_type}') as output_file:
            for message in output_line:
                output_file.write(message)
                output_file.write('\n')
        output_file.close()

    def color(self, color_code):
        return f"\33[{color_code}m".format(code=color_code)

    def create_chat_wav(self, chat_model_text=str):
        self.file_count = self.file_count + 1
        self.tts_model.tts_to_file(text=chat_model_text, speaker_wav=f'./voices/{self.clone_voice_file}', file_path=f'./model_output{self.file_count}.wav', language="en")

    def play_wav(self):
        wav_filename = f'./model_output{self.file_count}.wav'
        wav_object = simpleaudio.WaveObject.from_wave_file(wav_filename)
        play_audio = wav_object.play()
        play_audio.wait_done()

    def voice_to_text(self):
        text_data = []
        user_voice_data = self.stt_model.transcribe(f'./{self.rec_user_voice_file}')
        for voice in user_voice_data:        
            text_data.append(voice.text)
        combined_text = ' '.join(text_data)
        print(combined_text)
        return combined_text
    
    def record_user(self):
        self.user_file_count = self.user_file_count + 1
        rec_user_voice = sounddevice.rec(int(self.rec_seconds * self.rec_sample_freq), samplerate=self.rec_sample_freq, channels=self.rec_channels)
        sounddevice.wait()
        write(filename=self.rec_user_voice_file, rate=self.rec_sample_freq, data=rec_user_voice)
                                 
    def update_chat_template(self, prompt=str):
        
        chatml = f"""<|im_start|>system
        You are a scientist from the videogame Half-life. You are one of few who were able to escape the Resonance Cascade event that happened at the Black Mesa facility. You love to answer questions about your experiences during the event.<|im_end|>
        <|im_start|>user
        {prompt}<|im_end|>
        <|im_start|>assistant"""

        vicuna = f"""You are a scientist from the videogame Half-life. You are one of few who were able to escape the Resonance Cascade event that happened at the Black Mesa facility. You love to answer questions about your experiences during the event.

        User: {prompt}
        ASSISTANT:"""

        template_type = chatml
        return template_type

    def select_models(self):
        os.system("cls")
        print(self.color(self.yellow) + "llmon-py" + self.color(self.grey) + self.model_path, "\n")
        for model_file in self.path_list:
            if model_file.is_file():
                print(self.color(self.green) + f"{self.model_count}.) " + self.color(self.yellow) + f"{model_file.name}")
                self.model_list.append(model_file.name)
            self.model_count = self.model_count + 1
        self.path_list.close()

        print(self.color(self.grey))
        model_int = int(input('#>> '))
        chat_model_filename = self.model_list[model_int - 1]
        os.system("cls")
        return chat_model_filename

    def start(self):
        running_client = True
        chat_model_loaded = self.select_models()
        print (self.color(self.yellow) + "llmon-py " + self.color(self.grey) + "using " + self.color(self.white) + f"{chat_model_loaded}")
        chat_model = Llama(
            model_path=f'./models/{chat_model_loaded}', 
            main_gpu = self.gpu_device, 
            n_gpu_layers = self.gpu_layers, 
            n_ctx = self.model_ctx, 
            verbose = self.verbose_output, 
            chat_format = self.chat_format)
      
        while running_client:
            #feed = rssnews.parse_url(urls=['https://www.cbc.ca/webfeed/rss/rss-canada'])
            user_text_prompt = input(self.color(self.yellow) + "user>>> ")
            if user_text_prompt == '':
                print("Recording for {self.rec_seconds}:")
                self.record_user()
                user_voice_prompt = self.voice_to_text()
                prompt = self.update_chat_template(user_voice_prompt)
            else:  
                #read_test = """Kelly got the drip coffee machine going, then he pulled on a pair of swim trunks and headed topside. He hadn't forgotten to set the anchor light, he was gratified to see. The sky had cleared off, and the air was cool after the thunderstorms of the previous night. He went forward and was surprised to see that one of his anchors had dragged somewhat. Kelly reproached himself for that, even though nothing had actually gone wrong. The water was a flat, oily calm and the breeze gentle. The pink-orange glow of first light decorated the tree-spotted coastline to the east. All in all, it seemed as fine a morning as he could remember. Then he remembered that what had changed had nothing at all to do with the weather."""
                prompt = self.update_chat_template(user_text_prompt)

            chat_model_output = chat_model(prompt=prompt) 
            self.log_chat(user_message=user_text_prompt, model_message=chat_model_output['choices'][0]['text'])
            print(self.color(self.green))
            self.create_chat_wav(chat_model_output['choices'][0]['text'])
            self.play_wav()

if __name__ == "__main__":
    client = Client()
    client.start()
