# /codespace/llmon.py
import logging
from colorama import Fore
from langchain import LlamaCpp, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def load_chat(filename=str):
    with open(filename) as file:
        chat_log = file.readlines()
    return " ".join(chat_log)

def main(model=str, ctx_size=int, cpu_cores=int, gpu_layers=int, ai_name=str, username=str):
    memory = ConversationBufferMemory(ai_prefix=ai_name)
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    logging.basicConfig(filename="chat.log", filemode="a", level=logging.INFO, format="%(message)s")
    model = LlamaCpp(model_path=model, n_gpu_layers=gpu_layers, n_ctx=ctx_size, n_threads=cpu_cores, callback_manager=callback_manager, verbose=False, max_tokens=2048)
    llm = ConversationChain(llm=model, memory=memory, verbose=False)

    print (Fore.YELLOW + "llmon-py")
    user_input = input(Fore.GREEN + "import chat history? ")
    print(Fore.BLUE)
    if user_input == "yes":
        import_chat = True
        imported_chat = load_chat(filename="chat.log")           
    else:
        import_chat = False

    while True:
        if import_chat == False:
            user_input = input(Fore.BLUE + f"{username} > ")
            logging.info(f"{username}: {user_input}")
            print (Fore.RED)      
            llm_response = llm.predict(input=user_input)
            logging.info(f"{ai_name}: {llm_response}")
            print("\n")
        else:
            import_chat = False   
            print (Fore.RED)               
            llm.predict(input=imported_chat)
            print("\n")

if __name__ == "__main__":
    main(model="based-30b.ggmlv3.q4_K_M.bin", ctx_size=2048, cpu_cores=10, gpu_layers=12, ai_name="Johnny 5", username="Fusion")
