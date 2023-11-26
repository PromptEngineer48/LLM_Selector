from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama

import streamlit as st

def select_best_model(user_input, models_dict):
    llm = Ollama(model="neural-chat") #Selector Model

    # Construct the prompt for the LLM
    prompt = f"Given the user question: '{user_input}', evaluate which of the following models is most suitable: Strictly respond in 1 word only."
    for model, description in models_dict.items():
        prompt += f"\n- {model}: {description}"
    # print('prompt:', prompt)
    
    # Send the prompt to the LLM
    llm_response = llm(prompt)

    # print("llm_response: ", llm_response)

    # Parse the response to find the best model
    # This part depends on how your LLM formats its response. You might need to adjust the parsing logic.
    best_model = parse_llm_response(llm_response, models_dict=models_dict)

    return best_model



def parse_llm_response(response, models_dict):
    # Convert response to lower case for case-insensitive matching
    response_lower = response.lower()

    # Initialize a dictionary to store the occurrence count of each model in the response
    model_occurrences = {model: response_lower.count(model) for model in models_dict}

    # Find the model with the highest occurrence count
    best_model = max(model_occurrences, key=model_occurrences.get)

    # If no model is mentioned or there is a tie, you might need additional logic to handle these cases
    if model_occurrences[best_model] == 0:
        return "neural-chat"  # Or some default model

    return best_model


models_dict = {
    'neural-chat': 'A fine-tuned model based on Mistral with good coverage of domain and language.',
    'mistral': 'The popular model which is able to generate coherent text and perform various natural language processing tasks.',
    'codellama': 'A model that can use generate and discuss code and programme in python, c++, javacript and all the programming languages.',
    'llama2-uncensored': 'Uncensored Llama 2 model. This model can give NSFW replies and can be used to generate anything which the other models are shy about. Questions on hacking, immorality, sexual and all bad contents can be asked to this model',
    'orca-mini': 'A general-purpose model ranging from 3 billion parameters to 70 billion, suitable for entry-level hardware.',
    'llama2-chinese': 'Llama 2 based model fine tuned to improve Chinese dialogue ability.',
    'dolphin2.2-mistral': 'An instruct-tuned model based on Mistral. Version 2.2 is fine-tuned for improved conversation and empathy.',
}

# Layout the UI
st.set_page_config(page_title="Ollama Web UI by @PromptEngineer48", layout="wide")
st.title("Ollama Web UI by @PromptEngineer48")

# Main content area
st.header("How can I help you today?")
st.text_input("Send a message", key="user_input")

# Check for input
if st.session_state.user_input:
    best_model = select_best_model(st.session_state.user_input, models_dict)
    
    st.sidebar.write(f"THE SELECTED MODEL IS : {best_model}")
    # You can then use this model in your LLM call
    
    llm = Ollama(model=best_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    response = llm(st.session_state.user_input)
    st.write(response)
    


### Future 
# Memory
# Host
# More LLMs
# Own LLMs (Finetuning)
