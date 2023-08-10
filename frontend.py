# /codespace/llm-frontend.py
import logging
from colorama import Fore
from langchain import LlamaCpp, ConversationChain, PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationEntityMemory, CombinedMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def main(model=str, ctx_size=int, cpu_cores=int, gpu_layers=int, ai_name=str, user_name=str):
    with open("chat.log") as file:
        chat_log = file.readlines()

    memory = ConversationBufferMemory(ai_prefix=ai_name)
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    logging.basicConfig(filename="chat.log", filemode="w", level=logging.INFO, format="%(message)s")
    try:
        model = LlamaCpp(model_path=model, n_gpu_layers=gpu_layers, n_ctx=ctx_size, n_threads=cpu_cores, callback_manager=callback_manager, verbose=False)
    except:
        print("cannot load model")    
    llm = ConversationChain(llm=model, memory=memory, verbose=False)  

    print (Fore.YELLOW + "llmon-py")
    user_input = str
    while user_input != "exit":
        user_input = input(Fore.BLUE + "> ")
        logging.info(f"{user_name}: {user_input}")
        print (Fore.RED)      
        try:
            llm_response = llm.predict(input=user_input)
            logging.info(f"{ai_name}: {llm_response}")
        except:
            print(Fore.YELLOW + "token limit reached")
        print("\n")
    
if __name__ == "__main__":
    main("based-30b.ggmlv3.q4_K_M.bin", 2048, 10, 4, "Johnny 5", "User")
