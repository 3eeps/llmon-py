from colorama import Fore
from langchain import LlamaCpp, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# TODO - get tokens per message, when model reaches limit, summerize with summary buffer and add back to model memory
# TODO - add support for multiple backends, switch models, options to change settings
# MAYBE - irc style chat with multiple bots in a room? seems fun heh

def run_model(model, mem_buffer):       
    user_input = input(Fore.BLUE + "> ")
    llm = ConversationChain(llm=model, memory=mem_buffer, verbose=False)
    
    print (Fore.RED)
    llm.predict(input = user_input)
    print("\n")

def main():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    mem_buffer = ConversationBufferMemory(ai_prefix = "Johnny 5")
    model = LlamaCpp(model_path="./based-30b.ggmlv3.q4_K_M.bin", n_ctx=2048, n_threads=10, callback_manager=callback_manager, verbose=False)
    
    print (Fore.YELLOW + "llamon-py")
    user_input = None    
    while user_input != "exit":
        run_model(model, mem_buffer)

if __name__ == "__main__":
    main()
