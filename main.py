from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama


llm = Ollama(model="neural-chat", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
llm("What is 1+1")