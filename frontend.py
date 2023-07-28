# jank local frontend for llm models 

# libraries 
import os
from colorama import Fore
from cpuinfo import get_cpu_info
from langchain import LlamaCpp, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# init user
cpu_core_count = get_cpu_info()['count'] / 2
user_input = ""

# init model
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
conversation_buffer = ConversationBufferMemory()
ctx_size = 2048
ggml_model = LlamaCpp(
    model_path = "./based-30b.ggmlv3.q4_K_M.bin", 
    n_ctx = ctx_size, 
    n_threads = cpu_core_count,
    callback_manager = callback_manager, 
    verbose = False)

os.system("cls")
print (Fore.GREEN + "<model loaded>")
# run model
while user_input != "exit":
    user_input = input(Fore.BLUE + "> ")
    generate = ConversationChain(
        llm = ggml_model,
        memory = conversation_buffer,
        verbose = False)
    # output response
    print (Fore.RED + "")
    generate.predict(input = user_input)
