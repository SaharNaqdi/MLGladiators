from flask import Flask, request, jsonify
from transformers import pipeline
import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration,GenerationConfig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "persiannlp/mt5-small-parsinlu-translation_en_fa"
tokenizer = T5Tokenizer.from_pretrained(model_name)
translator = T5ForConditionalGeneration.from_pretrained('./t5_translator1')



def translate_text(text):
    translator.eval()
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    outputs = translator.generate(input_ids,max_new_tokens=500)
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_text

print(translate_text("hi"))

df = pd.read_csv('LSTM_translate.csv')
refrence=df['fa_text']
en_text=df['en_text']
output=[translate_text(text) for text in refrence]

data={
    'en_text':en_text
    ,'fa_text':refrence
    ,'T5_translation':output}
new_df = pd.DataFrame(data)
new_df.to_csv('t5_translation.csv', mode='a', header=True, index=True)



