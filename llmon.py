#/codespace/llmon.py

import os
# inference models
from TTS.api import TTS 
from llama_cpp import Llama
from pywhispercpp.model import Model

# audio imports
import simpleaudio
import sounddevice 
from scipy.io.wavfile import write

# Create a Client class to manage the chat application
class Client():
    def __init__(self):
        # Initialize various properties of the client
        self.rec_sample_freq = 44100
        self.rec_seconds = 5
        self.rec_channels = 2
        self.test_output = './voices/redg_sneaky.wav'
        self.user_file_count = 0
        self.rec_user_voice_file = f'user_chatout-{self.user_file_count}.wav'
        # speech to text model -pywhispercpp
        self.stt_model = Model('./models/stt/ggml-tiny.bin')
        self.file_count = 0
        self.new_session = True
        self.model_path = "./models"
        self.model_list = []
        self.model_count = 0
        self.path_list = os.scandir(self.model_path)
        # text to speech model -xtts/coqui-ai
        self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v1").cuda()

    # Log a user message and a model-generated message to a log file
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

    # Define a method to format text with color codes
    def color(self, color_code):
        return f"\33[{color_code}m".format(code=color_code)

    # Generate an audio file for a chat model's text
    def create_chat_wav(self, chat_model_text=str):
        self.file_count = self.file_count + 1
        self.tts_model.tts_to_file(text=chat_model_text, speaker_wav='./voices/redguard3.wav', file_path=f'./model_output{self.file_count}.wav', language="en")

    # Play a generated audio file
    def play_wav(self):
        wav_filename = f'./model_output{self.file_count}.wav'
        wav_object = simpleaudio.WaveObject.from_wave_file(wav_filename)
        play_audio = wav_object.play()
        play_audio.wait_done()

    def voice_to_text(self):
        user_voice_data = self.stt_model.transcribe(f'./{self.test_output}', speed_up=True)
        for voice in user_voice_data:
            voice_to_text = voice.text
        return voice_to_text

    def record_user(self):
        self.user_file_count = self.user_file_count + 1
        rec_user_voice = sounddevice.rec(int(self.rec_seconds * self.rec_sample_freq), samplerate=self.rec_sample_freq, channels=self.rec_channels)
        sounddevice.wait()
        write(filename=self.rec_user_voice_file, rate=self.rec_sample_freq, data=rec_user_voice)
                                 
    # Generate a chat template with a user prompt
    def update_chat_template(self, prompt):
        chatml = f"""system
        You are a scientist from the Half-Life universe that escaped the Black Mesa incident. You have many incredible stories to share.
        user
        {prompt}
        assistant"""

        vicuna = f"""You are a scientist from the Half-Life universe that escaped the Black Mesa incident. You have many incredible stories to share.
        
        User: {prompt}
        ASSISTANT:"""
        template_type = vicuna
        return template_type

    def select_character(self):
        # pick character profile for chat (template + .wav file)
        pass

    # Select a chat model from the available models
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

    # Start the chat application
    def start(self):
        running_client = True
        chat_model_loaded = self.select_models()
        print (self.color(93) + "llmon-py " + self.color(90) + "using " + self.color(37) + f"{chat_model_loaded}")
        # gguf llm model -llama-cpp-python
        chat_model = Llama(model_path=f'./models/{chat_model_loaded}', n_ctx=2048, verbose=False, chat_format="vicuna")
      
        while running_client:
            user_prompt = input(self.color(34) + "user>>> ")
            if user_prompt == 'rec':
                #self.record_user()
                user_voice_prompt = self.voice_to_text()  
                prompt = self.update_chat_template(user_voice_prompt)
            else:
                prompt = self.update_chat_template(user_prompt)

            chat_model_output = chat_model(prompt=prompt) 
            self.log_chat(user_message=user_prompt, model_message=chat_model_output['choices'][0]['text'])
            print(self.color(31))
            self.create_chat_wav(chat_model_output['choices'][0]['text'])            
            self.play_wav()

# Entry point for the script
if __name__ == "__main__":
    # Create a Client object and start the chat application
    client = Client()
    client.start()
