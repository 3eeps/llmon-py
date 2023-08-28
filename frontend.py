# /codespace/llmon.py
import os
from colorama import Fore
from langchain import LlamaCpp, ConversationChain, PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def load_model():
    model_path = "./models"
    model_list = []
    file_list = os.scandir(model_path)
    print(Fore.YELLOW + "llmon-py" + Fore.LIGHTBLACK_EX + model_path, "\n")
    list_num = 0
    for obj in file_list:
        list_num = list_num + 1
        if obj.is_file():
            print(Fore.GREEN + f"{list_num}.)" + Fore.YELLOW +  f"{obj.name}")
            model_list.append(obj.name)
    file_list.close()

    model_num = int(input(Fore.LIGHTBLACK_EX + ">>> "))
    model_to_load = model_list[model_num - 1]   
    return model_to_load

def main(model=str, ctx_size=int, cpu_cores=int, ai_name=str, user_name=str):
    template = """Your name is {ai_name}, and you are my AI friend, we always have a blast learning and chatting together and speak whats on our mind.

    User: {prompt}
    ASSISTANT:"""
    prompt = PromptTemplate(
    input_variables=["prompt", "ai_name"],
    template=template)

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llama_model = LlamaCpp(model_path=model, n_ctx=ctx_size, n_batch=1024, n_threads=cpu_cores, verbose=False, callback_manager=callback_manager)
    llm = LLMChain(llm=llama_model, verbose=False, prompt=prompt)

    print (Fore.YELLOW + "llmon-py " + Fore.LIGHTBLACK_EX + "topped with " + Fore.GREEN + f"{model}")
    while True:
        user_prompt = input(Fore.BLUE + f"{user_name} >>> ")
        print (Fore.RED)       
        llm.run({"prompt" : user_prompt, "ai_name" : ai_name})
        print("\n")

if __name__ == "__main__":   
    main(model=f"./models/{load_model()}", ctx_size=2048, cpu_cores=12, user_name="fusion", ai_name="Johnny 5")
