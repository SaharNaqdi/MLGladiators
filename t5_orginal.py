from flask import Flask, request, jsonify
from transformers import pipeline
import torch
import pandas as pd


# Load the translation model
translator = pipeline("translation_en_to_fa", model="persiannlp/mt5-small-parsinlu-translation_en_fa")

df = pd.read_csv('LSTM_translate.csv')
refrence=df['fa_text']
en_text=df['en_text']
output=[translator(text)[0]['translation_text'] for text in refrence]

data={
    'en_text':en_text
    ,'fa_text':refrence
    ,'T5_translation':output}
new_df = pd.DataFrame(data)
new_df.to_csv('t5_orginal_translation.csv', mode='a', header=True, index=True)


