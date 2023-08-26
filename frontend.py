# /codespace/llmon.py

from colorama import Fore
from langchain import LlamaCpp, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def main(model=str, ctx_size=int, batch_size=int, cpu_cores=int, ai_name=str, user_name=str):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    memory = ConversationBufferMemory(ai_prefix=ai_name)
    model = LlamaCpp(model_path=model, n_batch=batch_size, n_ctx=ctx_size, n_threads=cpu_cores, verbose=False, callback_manager=callback_manager)
    llm = ConversationChain(llm=model, memory=memory, verbose=False)

    print (Fore.YELLOW + "llmon-py")
    while True:
        user_input = input(Fore.BLUE + f"{user_name} >>> ")
        print (Fore.RED)      
        llm.predict(input=user_input)
        print("\n")

if __name__ == "__main__":
    main(model="./models/based-30b.ggmlv3.q4_K_M.bin", ctx_size=2048, batch_size=1024, cpu_cores=12, ai_name="Johnny 5", user_name="fusion")
