# /codespace/llm-frontend.py
import logging
from datetime import date
from colorama import Fore
from langchain import LlamaCpp, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def main(model=str, ctx_size=int, cores=int, gpu_layers=int, ai_name=str, user_name=str):
    date_log = date.today()
    logging.basicConfig(filename=f"{user_name}-{ai_name}-{date_log}.log", filemode="w", level=logging.INFO, format="%(message)s")
    try:
        model = LlamaCpp(model_path=model, n_gpu_layers=gpu_layers, n_ctx=ctx_size, n_threads=cores, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=False)
    except:
        print("cannot load model")    
    llm = ConversationChain(llm=model, memory=ConversationBufferMemory(ai_prefix=ai_name, verbose=False))  

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
