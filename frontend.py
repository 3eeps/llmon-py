# jank local frontend for llm models 

# libraries 
from langchain.memory import ConversationBufferMemory
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain, ConversationChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# model settings
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
conversation_buffer = ConversationBufferMemory()
thread_count = 12
ctx_size = 2048
user_input = ""

# init model
ggml_model = LlamaCpp(
    model_path = "based-30b.ggmlv3.q4_K_M.bin", 
    n_ctx = ctx_size, 
    n_threads = thread_count,
    callback_manager = callback_manager, 
    verbose = False)

print ("chadGPT v0.5")
print (">>>")

while user_input != "exit":
    user_input = input()
    generate = ConversationChain(
        llm = ggml_model,
        memory = conversation_buffer,
        verbose = False)

    generate.predict(input = user_input)
    print ("\n")