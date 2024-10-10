import requests
import pandas as pd
import pysrt
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate


# get a sentence and return its translation
def translate(sentence):
    # Metis API and URL
    metis_api_key = "tpsg-YkxgxrCSgCmZSxENz1Mcrsy2Ph3lPot"

    # model initialization in Metis
    model_provider = "openai_chat_completion"
    model_name = "gpt-4o"

    # Create a custom LangChain LLM Class that interacts with Metis API
    class MetisLLM(LLM):
        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            return request_to_metis(prompt, metis_api_key, model_provider, model_name)

        @property
        def _identifying_params(self):
            return {"model": model_name}

        @property
        def _llm_type(self):
            return "metis_llm"

    # Instantiate the model
    model = MetisLLM()

    # Define a template for translation prompt
    template = """Use informal language to translate the following English text to Persian:{input_text}."""

    # Create a prompt template
    prompt_template = PromptTemplate(template=template, input_variables=["input_text"])

    # Create a message to get to the model
    messages = prompt_template.format(input_text=sentence)

    # translate the input
    output_sentence = model.invoke(messages)

    return output_sentence


# Function to make a request to the Metis API
def request_to_metis(prompt, api_Key, model_provider, model_name):
    metis_url = 'https://api.metisai.ir/api/v1/chat/{provider}/completions'

    # construct the headers for the request
    headers = {
        "x-api-key": api_Key,
        "Content-Type": "application/json"
    }

    # Payload for the request
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that translates English subtitle of a movie to Persian."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    # Construct the full API endpoint URL with the provider
    url = metis_url.replace('{provider}', model_provider)

    # Make the POST request to Metis API
    response = requests.post(url, headers=headers, json=data)

    # Parse the response JSON and return the result
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:  # Check for errors
        raise Exception(f"Failed to get response from Metis API: {response.text}")


if __name__ == '__main__':
    print(translate("can you tell me what's going on?"))
