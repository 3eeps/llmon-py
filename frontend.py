# /codespace/llm-frontend.py
from colorama import Fore
from langchain import LlamaCpp, ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def main(model, ctx_size, threads, ai_name):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    model = LlamaCpp(model_path=model, n_ctx=ctx_size, n_threads=threads, callback_manager=callback_manager, verbose=False)
    llm = ConversationChain(llm=model, memory=ConversationSummaryBufferMemory(ai_prefix=ai_name, llm=model, max_token_limit=2048, verbose=False))
    
    print (Fore.YELLOW + "llmon-py")
    user_input = None
    while user_input != "exit":
        user_input = input(Fore.BLUE + "> ")
        print (Fore.RED)
        llm.predict(input=user_input)
        print("\n")

if __name__ == "__main__":
    main("based-30b.ggmlv3.q4_K_M.bin", 2048, 10, "Johnny 5")
