#/codespace/llmon.py

# Import necessary modules and libraries
import os
from TTS.api import TTS
from llama_cpp import Llama
import simpleaudio

# Create a Client class to manage the chat application
class Client():
    def __init__(self):
        # Initialize various properties of the client
        self.file_count = 0
        self.new_session = True
        self.model_path = "./models"
        self.model_list = []
        self.model_count = 0
        self.path_list = os.scandir(self.model_path)
        self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v1")

    # Log a user message and a model-generated message to a log file
    def log_chat(self, user_message, model_message):
        if self.new_session == False:
            write_type = 'a'
        else:
            write_type = 'w'
        self.new_session = False

        messages_to_log = [f'user: {user_message}', f'chat_output{self.file_count}.wav: {model_message}']
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
        self.tts_model.tts_to_file(text=chat_model_text, speaker_wav='./voices/redguard0.wav', file_path=f'./chat_output{self.file_count}.wav', language="en")

    # Play a generated audio file
    def play_wav(self):
        wav_filename = f'./chat_output{self.file_count}.wav'
        wav_object = simpleaudio.WaveObject.from_wave_file(wav_filename)
        play_audio = wav_object.play()
        play_audio.wait_done()

    # Generate a chat template with a user prompt
    def update_chat_template(self, prompt):
        chatml = f"""system
        You are a scientist from the game Half-Life that escaped the Black Mesa incident. You have many incredible stories to share.
        user
        {prompt}
        assistant"""

        vicuna = f"""You are a scientist from the game Half-Life that escaped the Black Mesa incident. You have many incredible stories to share.
        User: {prompt}
        ASSISTANT:"""
        template_type = vicuna
        return template_type

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
        chat_model = Llama(model_path=f'./models/{chat_model_loaded}', n_ctx=2048, verbose=False, chat_format="vicuna")
      
        while running_client:
            user_prompt = input(self.color(34) + "user>>> ")
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
