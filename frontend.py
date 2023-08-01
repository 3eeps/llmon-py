from colorama import Fore
from langchain import LlamaCpp, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def run_model(model, mem_buffer):       
    user_input = input(Fore.BLUE + "> ")
    llm = ConversationChain(llm=model, memory=mem_buffer, verbose=False)

    print (Fore.RED)
    llm.predict(input=user_input)
    print("\n")

def main(ctx_size, threads):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    mem_buffer = ConversationBufferMemory(ai_prefix="Johnny 5", human_prefix="3eeps")
    model = LlamaCpp(model_path="./based-30b.ggmlv3.q4_K_M.bin", n_ctx=ctx_size, n_threads=threads, callback_manager=callback_manager, verbose=False)
    
    print (Fore.YELLOW + "llmon-py")
    user_input = None
    while user_input != "exit":
        run_model(model, mem_buffer)

if __name__ == "__main__":
    main(2048, 10)
