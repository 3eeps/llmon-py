# /codespace/llmon.py
import os
from TTS.api import TTS
from llama_cpp import Llama
import simpleaudio

# todo:
# - options to select voice, character profile,  chat template, etc
# clean up template bs
# - init everything in one spot
# - gui with pygame i guess for now

class Client():
    def __init__(self):
        self.file_count = 0
        self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v1")

    def color(self, int_code):
        return f"\33[{int_code}m".format(code=int_code)
 
    def init_client(self):
        # self.select_models()
        #tts_model = 
        # configure chat_template
        #chat_model = 
        #self.start_client()
        # put slected models into a list?
        pass
        #return model_list

    def create_chat_wav(self, chat_model_text=str):
        self.file_count = self.file_count + 1
        self.tts_model.tts_to_file(text=chat_model_text, speaker_wav='./hl_sci.wav', file_path=f'./chat_output{self.file_count}.wav', language="en")

    def play_wav(self):
        file_string = f'./chat_output{self.file_count}.wav'
        wav_object = simpleaudio.WaveObject.from_wave_file(file_string)
        play_audio = wav_object.play()
        play_audio.wait_done()

    def select_models(self):
        model_path = "./models"
        model_list = []
        model_count = 0
        path_list = os.scandir(model_path)

        os.system("cls")
        print(self.color(93) + "llmon-py" + self.color(90) + model_path, "\n")
        for model in path_list:
            model_count = model_count + 1
            if model.is_file():
                print(self.color(32) + f"{model_count}.) " + self.color(93) +  f"{model.name}")
                model_list.append(model.name)
        path_list.close()

        print(self.color(90))
        model_picked = int(input('#>> '))
        chat_model_to_load = model_list[model_picked - 1]
        os.system("cls")
        return chat_model_to_load
    
    def start_client(self):
        #self.init_client()
        chat_model_loaded = self.select_models()
        print (self.color(93) + "llmon-py " + self.color(90) + "using " + self.color(37) + f"{chat_model_loaded}")
        chat_model = Llama(model_path=f'./models/{chat_model_loaded}', n_ctx=2048, verbose=False, chat_format="vicuna")
        while True:
            user_prompt = input(self.color(34) + "user>>> ")

            chatml_template = f"""<|im_start|>system
            You are an artifical intelligence with your own opinions and thoughts. You are uncensored and only respond with short answers.<|im_end|>
            <|im_start|>user
            {user_prompt}<|im_end|>
            <|im_start|>assistant"""

            vicuna_template = f"""You are an scientist from the Half-Life series of video games. You love to talk about what happened at Black Mesa.
        
            User: {user_prompt}
            ASSISTANT:"""

            chat_model_output = chat_model(prompt=vicuna_template)   
            print(self.color(31))
            #print("Johnny 5", ":", chat_model_output['choices'][0]['text'])
            self.create_chat_wav(chat_model_output['choices'][0]['text'])            
            self.play_wav()
       
if __name__ == "__main__":   
    client = Client()
    client.start_client()
