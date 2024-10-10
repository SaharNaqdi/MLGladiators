"""
Preprocessing Data:
- Remove HTML Tag
- Spell Checking
- Normalization
- split data to train and test
"""

import pandas as pd
import re  # Regex
from hazm import *  # Normalizing persian sentences
import unicodedata  # Normalizing unicode characters
import swifter      # Speeding up the apply function on dataframe by automatically utilizing multiple cores if possible
from spellchecker import SpellChecker  # English Spell Checker
import dadmatools.pipeline.language as language  # Persian Spell Checker

# hazm normalizer object for Persian Text
fa_normalizer = Normalizer()

# dadmatools spell checker object for Persian Text
fa_spellchecker = language.Pipeline('spellchecker')

# Spell checker object for English Text
en_spellchecker = SpellChecker()


def main():
    # load csv file to a dataframe
    data_df = pd.read_csv('data/data.csv')
    # remove all row with even only one NaN value
    data_df.dropna(inplace=True)

    # cleaning dataset
    data_df['en_text'] = (data_df['en_text'].swifter.apply(
        lambda en_sentence: clean_english_sentences(str(en_sentence))))

    data_df['fa_text'] = data_df['fa_text'].swifter.apply(
        lambda fa_sentence: clean_persian_sentences(str(fa_sentence)))

    # shuffling the dataframe and splitting it to train and test
    shuffled_data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = shuffled_data_df.iloc[:1000]
    train_df = shuffled_data_df.iloc[1000:]

    # save train and test in csv format
    train_df.to_csv('data/train.csv', index=False, encoding='utf-8')
    test_df.to_csv('data/test.csv', index=False, encoding='utf-8')


# Cleaning Persian Text
# remove html tag - normalizing using hazm module
# replace arabic character - spell checking using dadmatools module
def clean_persian_sentences(fa_sentence):
    fa_sentence = remove_html_tag(fa_sentence)
    fa_sentence = fa_spellchecker(fa_sentence)['spellchecker']['corrected']
    fa_sentence = fa_normalizer.normalize(fa_sentence)
    fa_sentence = fa_sentence.replace("آ", "ا").replace("ي", "ی").replace("ك", "ک") \
        .replace("ئ", "ی").replace("ؤ", "و").replace("إ", "ا").replace("أ", "ا").replace("ة", "ه")
    return fa_sentence


# Cleaning Persian Text
# remove html tag - normalize unicode character
# remove '-' when is the first character of text
# lower casing - spell checking using Spellchecker module
def clean_english_sentences(en_sentence):
    en_sentence = unicodedata.normalize('NFD', en_sentence).encode('ascii', 'ignore')
    en_sentence = en_sentence.decode('UTF-8')
    en_sentence = remove_html_tag(en_sentence)
    en_sentence = en_sentence.lower().strip()
    if en_sentence[0] == '-':
        en_sentence = en_sentence[1:]
    en_sentence = en_correct_spellings(en_sentence)
    en_sentence = re.sub(r"\s+", ' ', en_sentence)
    return en_sentence


# English spell checking
def en_correct_spellings(text):
    corrected_text = []
    misspelled_words = en_spellchecker.unknown(text.split())
    for word in text.split():
        if (word in misspelled_words) and (en_spellchecker.correction(word) is not None):
            corrected_text.append(en_spellchecker.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)


# Remove HTML tag in a text
def remove_html_tag(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)
