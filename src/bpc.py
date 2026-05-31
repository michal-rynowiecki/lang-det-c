import math

import numpy as np
import pandas as pd
import torch

from pathlib import Path

import re
import json

import datasets
from huggingface_hub import ModelCard

from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, CanineTokenizer, CanineModel, AutoModelForImageTextToText

'''
Takes in a model for Masked LM and returns the loss obtained by
masking each token one-by-one.

Inputs
--------
tokenizer       - the tokenizer (actual tokenizer, not path)
model           - the model (actual model, not path)
text: str       - text input; not tokenized
batch_size: int - size to which divide up the matrix resulting from
                  creating a len(sentence)xlen(sentence) matrix
'''
def encoder_full_loss(inputs, tokenizer, model, text, batch_size=8):
    model.eval()
    device = model.device

    # Check if the model needs to do truncation (some don't have a max length set)
    # (the 100000 is arbitrary but i don't think there's any longer data points in the dataset)
    full_ids = inputs['input_ids'][0]
    print("Number of columns: ",len(full_ids))

    # Create a batch with a row for each length
    batch = full_ids.unsqueeze(0).repeat(len(full_ids), 1)
    labels = torch.full_like(batch, -100)
    
    # Create an identity matrix of size [batch.size(0) x batch.size(0)] and
    # obtain a mask for it
    mask = torch.eye(batch.size(0), dtype=bool)
    
    # Using the created mask, set all values in the labels on the diagonal
    # to the values of the actual batch, so that loss only gets calculated
    # for a single token per row in the batch
    labels[mask] = batch[mask]
    batch[mask] = tokenizer.mask_token_id

    # Drop the first and last rows corresponding to start and end token
    batch = batch[1: -1]
    labels = labels[1: -1]
    
    # Now the batch size is equal to [len(sentence) x len(sentence)]
    # This is too long so I split it to multiple batches of max row count 16
    new_batch = torch.split(batch, batch_size)
    new_labels = torch.split(labels, batch_size)

    # This loop will accumulate the loss over all the
    # batch_size length batches
    single_data_point_loss = 0
    with torch.no_grad():
        for b, l in zip(new_batch, new_labels):
            # Send to GPU (if available)
            b = b.to(device)
            l = l.to(device)
            output = model(b, labels=l)
            single_data_point_loss += output.loss.item() * b.shape[0]

    # Calculate BPC
    bpc = single_data_point_loss / (len(text) * np.log(2))
    return bpc

def decoder_full_loss(tokenizer, model, text):
    model.eval()
    device = model.device

    kwargs = {"return_tensors": "pt",
            "truncation": True,
            "max_length": 512,
        }
    
    inputs = tokenizer(text, **kwargs)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    
    with torch.no_grad():
        output = model(**inputs, labels=labels)
    
    num_valid_tokens = (labels != -100).sum()
    total_loss = output.loss.item() * num_valid_tokens

    bpc = total_loss / (len(text) * math.log(2))

    return bpc.item()
    
'''
Takes in a list of X BPC values each representing the average at a particular
number of data points being calculated, then determined if they stay within
the prob. range of each other

Parameters
-----------
bpcs        - list of values
prob        - the probability to check

Outputs
------------
outcome     - boolean vale representing if the values are within the prob range of each other
'''
def within_range(bpcs, prob):
    avg_val = sum(bpcs) / len(bpcs)
    spread = max(bpcs) - min(bpcs)

    outcome = (spread / avg_val) <= prob
    return outcome

def lang_len(lm, alpha=0.1, rang=5, encoder=True, langs=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load in the model based on if its an encoder or not
    ModelClass = AutoModelForMaskedLM if encoder else AutoModelForCausalLM
    if "Ministral" in lm:
        ModelClass = AutoModelForImageTextToText
    model = ModelClass.from_pretrained(lm,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(lm, trust_remote_code=True)

    # Open files for writing in data
    dir_path = Path("..") / "data" / "lang_lengths" / lm
    dir_path.mkdir(parents=True, exist_ok=True)

    # Get full language name from the current langauge id
    #configs = datasets.get_dataset_config_names('cis-lmu/glotlid-corpus') # Online
    configs = datasets.get_dataset_config_names("../data/glotlid-corpus") # Offline

    model_languages = ModelCard.load(lm).data['language']
    if langs:
        model_languages = [x for x in re.findall(r"[A-Za-z]+", langs)]

    # Read in the ISO language dataset
    lang_df = pd.read_csv('../data/language_codes.txt', sep='\t')

    # Convert full language names (referant names) to id names for filtering the datasets later on. Need different ISO formats because of Hugging Face's lack of consistency
    languages = [(i, np.squeeze(lang_df.loc[lang_df['Id'] == i[:3]][['Ref_Name', 'Id', 'Part2b', 'Part2t', 'Part1']].values.tolist())) for i in configs]

    # Determine if languages are present or not in the current model
    languages_to_check = [
        (any(elem in model_languages for elem in a[1]), a)
        for a in languages
    ]
    #languages_to_check = [(True, ('eng_Latn', 1))]
    
    len_dict = {}
    for present, language in languages_to_check:
        # 1. get dataset
        # had to add the try except because some languages seem to be bugged
        try:
            dataset = datasets.load_dataset("../data/glotlid-corpus", language[0]).shuffle()
        except:
            print(f"Can't read in {language}")
            continue
        # 2. create variables for calculating bpc
        n = 0
        bpc, avg_bpc = [], []

        # 3. Calculate bpc
        for data_point in dataset['train']:
            # Some data points that were too short were throwing errors
            kwargs = {"return_tensors": "pt",
                "truncation": True,
                "max_length": 512
            }
            inputs = tokenizer(data_point['text'], **kwargs)
            if len(inputs['input_ids'][0]) < 5:
                continue
            
            # Maintain the number of traversed points
            n += 1
            # 3a) if encoder, go through datapoints one by one
            if encoder:    
                # Retrieve the BPC for the current data point
                batch_size = 8
                if lm == "facebook/xlm-roberta-xxl":
                    batch_size = 4
                res = encoder_full_loss(inputs, tokenizer, model, data_point['text'], batch_size)

            # 3b) if decoder, still go one by one since the avg needs to be
            # calculated for every single data point
            if not encoder:
                res = decoder_full_loss(tokenizer, model, data_point['text'])

            # Add the average and the result to a list for points
            bpc.append(res)
            avg_val = sum(bpc) / len(bpc)
            avg_bpc.append(avg_val)

            # If the change within the last rang points is below alpha
            # stop the iteration and append the length required
            if len(bpc) >= rang and within_range(avg_bpc[n-rang:n], alpha):
                len_dict[language[0]] = (n, avg_val)

                # Write JSON to the language presence corresponding file
                with open(dir_path / f"{alpha}_{rang}_{present}.json", "a") as f:
                    q = np.quantile(bpc, [0.25, 0.5, 0.75])
                    print("Saving in:")
                    print(dir_path / f"{alpha}_{rang}_{present}.json", "a")
                    json.dump({language[0]: {"n": n, "avg_bpc": avg_val, "0.25q":q[0], "0.5q": q[1], "0.75q":q[2]}}, f)
                    f.write("\n")
                break