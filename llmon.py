# /codespace/llmon.py
import os
from TTS.api import TTS
from llama_cpp import Llama
import simpleaudio

class Client():
    def __init__(self):
        self.file_count = 0
        self.model_path = "./models"
        self.model_list = []
        self.model_count = 0
        self.path_list = os.scandir(self.model_path)
        self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v1")
        
    def color(self, color_code):
        return f"\33[{color_code}m".format(code=color_code)

    def create_chat_wav(self, chat_model_text=str):
        self.file_count = self.file_count + 1
        self.tts_model.tts_to_file(text=chat_model_text, speaker_wav='./voices/max_payne.wav', file_path=f'./chat_output{self.file_count}.wav', language="en")

    def play_wav(self):
        wav_filename = f'./chat_output{self.file_count}.wav'
        wav_object = simpleaudio.WaveObject.from_wave_file(wav_filename)
        play_audio = wav_object.play()
        play_audio.wait_done()

    def update_chat_template(self, prompt):
        chatml = f"""<|im_start|>system
        You are a scientist that escaped the Black Mesa incident from the game series Half-Life.<|im_end|>
        <|im_start|>user
        {prompt}<|im_end|>
        <|im_start|>assistant"""

        vicuna = f"""You are a scientist that escaped the Black Mesa incident from the game series Half-Life. You love to talk about what happened.
        
        User: {prompt}
        ASSISTANT:"""
        template_type = vicuna
        return template_type

    def select_models(self):
        os.system("cls")
        print(self.color(93) + "llmon-py" + self.color(90) + self.model_path, "\n")
        for model_file in self.path_list:
            self. model_count = self.model_count + 1
            if model_file.is_file():
                print(self.color(32) + f"{self.model_count}.) " + self.color(93) +  f"{model_file.name}")
                self.model_list.append(model_file.name)
        self.path_list.close()

        print(self.color(90))
        model_int = int(input('#>> '))
        chat_model_filename = self.model_list[model_int - 1]
        os.system("cls")
        return chat_model_filename
    
    def start(self):
        chat_model_loaded = self.select_models()
        print (self.color(93) + "llmon-py " + self.color(90) + "using " + self.color(37) + f"{chat_model_loaded}")
        chat_model = Llama(model_path=f'./models/{chat_model_loaded}', n_ctx=2048, verbose=False, chat_format="vicuna")
        while True:
            user_prompt = input(self.color(34) + "user>>> ")
            prompt = self.update_chat_template(user_prompt)

            chat_model_output = chat_model(prompt=prompt) 
            print(self.color(31))
            self.create_chat_wav(chat_model_output['choices'][0]['text'])            
            self.play_wav()
       
if __name__ == "__main__":   
    client = Client()
    client.start()
