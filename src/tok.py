from transformers import AutoTokenizer
from huggingface_hub import ModelCard

import re

import statistics

import pandas as pd
import numpy as np

import json
from pathlib import Path

import datasets
"""
Reads in the list of tokenizers and their assosciated languages
"""
def read_tokenizers(path):
    tokenizers = []
    with open(path, 'r') as f:
        for line in f:
            l = line.split('\t')[0:2]
            tokenizers.append(l)
    return tokenizers

"""
Calculates the ratio of UNK tokens to all tokens in a tokenized text
"""
def unk_number(text, tokenizer):
    tokenized = tokenizer.tokenize(text)
    unk = tokenizer.unk_token
    unk_ratio = tokenized.count(unk)/len(tokenized)
    return unk_ratio

"""
Calculates the average token length of a text, drops the join sign 
from tokenized text, e.g "##"
"""
def avg_token_length(text, tokenizer):
    # Attempt at getting the character that signifies a word split
    if tokenizer.tokenize('electroencephalogram')[1][0] not in "electroencephalogram":
        join_sign = tokenizer.tokenize('electroencephalogram')[1][0]
    elif tokenizer.tokenize('electroencephalogram')[0][0] not in "electroencephalogram":
        join_sign = tokenizer.tokenize('electroencephalogram')[0][0]
    else:
        join_sign = ''

    unk = tokenizer.unk_token

    tokenized = tokenizer.tokenize(text)
    tokenized = [token.replace(join_sign, '') for token in tokenized]
    if unk:
        tokenized = [token.replace(unk, '') for token in tokenized]
    
    avg_length = statistics.fmean([len(t) for t in tokenized])
    
    return avg_length

"""
Calculates the number of token per word for provided text
"""
def toks_per_word(text, tokenizer):
    tokenized = tokenizer.tokenize(text)
    num_words = len(re.findall(r'\w+', text))
    return len(tokenized)/num_words

"""
Tokenier based method for detecting the presence of a language 
in a particular language model.

Inputs
--------
@ tok_path: str         - the path to a tokenizer on Hugging Face or local storage
@ languages: list[str]  - list of languages to examine the presence of in the training data

Returns
--------
@ 
"""
def tokenizer_based(path, langs=None):
    # load in the tokenizer file
    tok_path    = path
    tok_langs   = ModelCard.load(path).data['language'] # Not quite bulletproof
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    # If languages are provided manually and were not available in the model card on huggign face
    if langs:
        tok_langs = langs

    # Open files for writing in data
    dir_path = Path("..") / "data" / "tokenizer_based" / tok_path
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get full language name from the current langauge id
    #configs = datasets.get_dataset_config_names('cis-lmu/glotlid-corpus') # Online
    configs = datasets.get_dataset_config_names("../data/glotlid-corpus") # Offline

    # Convert full language names (referant names) to id names for filtering the datasets later on. Need different ISO formats because of Hugging Face's lack of consistency
    lang_df = pd.read_csv(Path("../data/language_codes.txt"), sep='\t')
    
    languages = [(i, np.squeeze(lang_df.loc[lang_df['Id'] == i[:3]][['Ref_Name', 'Id', 'Part2b', 'Part2t', 'Part1']].values.tolist())) for i in configs]
    
    languages_to_check = [
        (any(elem in tok_langs for elem in a[1]), a)
        for a in languages
    ]

    #languages_to_check = [(True, ('eng_Latn', 1))]
    for present, language in languages_to_check:

        # 2. create variables for calculating token statistics
        n = 0
        unk_ratios, avg_tk_lenghts, tokens_per_word = [], [], []

        #if present:
        try:
            dataset = datasets.load_dataset("../data/glotlid-corpus", language[0]).shuffle()
        except:
            print(f"Can't read in {language}")
            continue


        for data_point in dataset['train']:
            if n > 20:
                break
            text = data_point['text'] #'ⴳ ⵓⵢⵏⴰⵙ, ⵍⴰⵖⴰⵏ ⵉⵎⵓⵙⵏⴰⵡⵏ ⵏ ⵜⵉⵏⵎⵍ ⵏ ⵜⴰⵙⵏⵓⵊⵢⴰ ⵏ ⵜⵙⴷⴰⵡⵉⵜ ⵏ ⵙⵜⴰⴷⴼⵓⵔ ⵙ ⵙⵓⵏⵓⵍⴼⵓ ⵏ ⵢⴰⵏ ⵡⴰⵍⵍⴰⵍ ⴰⵎⴰⵢⵏⵓ ⵏ ⵡⴰⴽⴰⵣ ⵍⵍⵉ ⵉⵣⵎⵔⵏ ⴰⴷ ⵉⵙⵎⵎⵙⵜⵉ ⴳⵔ ⵜⵖⵔⴰⵙⵉⵏ ⵙ ⵡⴰⵏⴰⵡⵏ: ⵢⴰⵜ ⵜⵛⵔⵉⵃⵜ ⵜⴰⵎⵥⵥⴰⵏⵜ ⵉⵖⵉⵏ ⴰⴷ ⵉⵜⵜⵓⵙⵉⴳⴳⵣ ⴷ ⵉⵖⵉⵏ ⴰⴷ ⵉⵜⵜⵓⵀⵢⵢⵓ ⵙ ⵜⵙⴰⴳⴳⵣⵉⵏ ⵏ ⵓⵙⴰⵎⵖ ⴰⵏⴰⵡⴰⵢ ⵙ ⵡⴰⵜⵜⴰⵢ ⵏ ⵙⴰⵏⵜ ⵏ ⵉⵡⵓⵏⴰⴽ ⵉⵎⵓⵏⵏ ⵉ ⵢⴰⵜ.' # 

            unk_ratio   = unk_number(text, tokenizer)
            avg_tk      = avg_token_length(text, tokenizer)
            tok_ratio   = toks_per_word(text, tokenizer)
            #print('Unk ratio: ', unk_ratio)
            unk_ratios.append(unk_ratio)
            #print('Average token length: ', avg_tk)
            avg_tk_lenghts.append(avg_tk)
            #print('Tokens per word: ', tok_ratio)
            tokens_per_word.append(tok_ratio)

            n += 1

        print(f"Average unk ratio for {language[0]}: {np.mean(unk_ratio)}")
        print(f"Average token length for {language[0]}: {np.mean(avg_tk_lenghts)}")
        print(f"Average token per word ratio for {language[0]}: {np.mean(tokens_per_word)}")

        # Write JSON to the language presence corresponding file
        with open(dir_path / f"{present}.json", "a") as f:
            json.dump({
                "language": language[0],
                "unk ratio": np.mean(unk_ratio), 
                "token_lengt:":np.mean(avg_tk_lenghts),
                "token/word":np.mean(tokens_per_word)
                }, f)
            f.write("\n")
