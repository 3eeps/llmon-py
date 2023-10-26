#/codespace/llmon.py

import os
from TTS.api import TTS 
from llama_cpp import Llama
from pywhispercpp.model import Model
from nltk.tokenize import sent_tokenize

import simpleaudio
import sounddevice 
from scipy.io.wavfile import write

class Client():
    def __init__(self):
        self.rec_sample_freq = 44100
        self.rec_seconds = 10
        self.rec_channels = 2
        self.test_output = './voices/aftertest.wav'
        self.user_file_count = 0
        self.rec_user_voice_file = f'user_chatout-{self.user_file_count}.wav'
        self.stt_model = Model('./models/stt/ggml-tiny.bin')

        self.chat_template = str
        self.file_count = 0
        self.new_session = True
        self.model_path = "./models"
        self.model_list = []
        self.model_count = 0
        self.path_list = os.scandir(self.model_path)

        self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v1").cuda()

    def log_chat(self, user_message=str, model_message=str):
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

        basic = f"""{prompt}"""

        chatml = f"""system
        You are an helpful AI assisant. Please take all text from the user prompt, and repeat it back. Thank you!
        user
        {prompt}
        assistant"""

        vicuna = f"""Your name is Johnny 5, and you are an AI assistant. Please feel free to express your thoughts and opinions on topics we discuss. Thank you!

        User: {prompt}
        ASSISTANT:"""
        template_type = vicuna
        return template_type

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
        chat_model = Llama(model_path=f'./models/{chat_model_loaded}', main_gpu=0, n_gpu_layers=22, n_ctx=2048, verbose=False, chat_format="vicuna")
      
        while running_client:
            #feed = rssnews.parse_url(urls=['https://www.cbc.ca/webfeed/rss/rss-canada'])
            user_prompt = input(self.color(34) + "user>>> ")
            if user_prompt == '':
                self.record_user()  
                text = self.voice_to_text()
                prompt = self.update_chat_template(prompt=text)
            else:               
                remorse = """November
                Camille had either been the world's most powerful hurricane or the largest tornado in history. Certainly it had done the job
                to this oil rig, Kelly thought, donning his tanks for his last dive into the Gulf. The super-structure was wrecked, and all four of
                the massive legs weakened - twisted like the ruined toy of a gigantic child. Everything that could safely be removed had
                lready been torched off and lowered by crane onto the barge they were using as their dive base. What remained was a skeletal
                platform which would soon make a fine home for local game fish, he thought, entering the launch that would take him
                alongside. The two divers would be working with him, but Kelly was in charge. They went over procedures on the way over
                while a safety boat circled nervously to keep the local fishermen away. It was foolish of them to be here - the fishing wouldn't
                be very good for the next few hours - but events like this attracted the curious. And it would be quite a show, Kelly thought
                with a grin as he rolled backwards off the dive boat."""

                read_back = """Launch ... two, two valid launches, Robin,' Tait warned from the back.
                'Only two?' the pilot asked.
                'Maybe he has to pay for them,' Tait suggested coolly. 'I have them at nine. Time to do some pilot magic, Rob.'
                'Like this?' Zacharias rolled left to keep them in view, pulling into them, and split-S-ing back down. He'd planned it well,
                ducking behind a ridge. He pulled out at a dangerous low altitude, but the SA-2 Guideline missiles went wild and dumb four
                thousand feet over his head.
                'I think it's time,' Tait said.
                'I think you're right.' Zacharias turned hard left, arming his cluster munitions. The F-105 skimmed over the ridge, dropping
                back down again while his eyes checked the next ridge, six miles and fifty seconds away.
                'His radar is still up,' Tait reported. 'He knows we're coming.'
                'But he's only got one left.' Unless his reload crews are really hot today. Well, you can't allow for everything.
                'Some light flak at ten o'clock.' It was too far to be a matter of concern, though it did tell him which way out not to take.
                'There's the plateau.'"""

                prompt = self.update_chat_template(user_prompt)

            chat_model_output = chat_model(prompt=prompt) 
            self.log_chat(user_message=user_prompt, model_message=chat_model_output['choices'][0]['text'])
            print(self.color(31))
            self.create_chat_wav(chat_model_output['choices'][0]['text'])            
            self.play_wav()

if __name__ == "__main__":
    client = Client()
    client.start()
