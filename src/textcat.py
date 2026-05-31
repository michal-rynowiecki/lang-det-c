# TODO
# 1. function to compare profiles
#   a) convert list to dictionary

from collections import Counter
from pathlib import Path
import string

from transformers import AutoTokenizer
# Get a language for detection and retrieve the data for this from glotlid, then use a tokenizer on same text to see the most common tokens

def get_ngrams(word, minn=4, maxn=7):
    ngrams = []
    for n in range(minn, maxn + 1):
        for i in range(len(word) - n + 1):
            ngrams.append(word[i:i+n])
    return ngrams


def extract_top_ngrams(dir_path, top_k=100, minn=5, maxn=8):
    counter = Counter()
    
    for file_path in Path(dir_path).rglob("*"):
        if file_path.is_file():
            with open(file_path, "r") as f:
                for line in f:
                    words = line.strip().split()
                    for word in words:
                        ngrams = get_ngrams(word, minn, maxn)
                        counter.update(ngrams)
    
    ranked_list = [word.lower() for word, _ in counter.most_common(top_k)]
    return ranked_list

def clean_token(token, min_len=3, max_len=6):
    # Remove leading/trailing punctuation
    token = token.strip(string.punctuation)
    
    # Optional: lowercase for consistency
    token = token.lower()
    
    # Keep only alphabetic tokens
    if not token.isalpha():
        return None

    #if 5 <= len(token) <= 8:
    return token

    # Length filter
    #return None

def extract_top_tokens_from_dir(dir_path, tokenizer, top_k=200):
    counter = Counter()
    
    for file_path in Path(dir_path).rglob("*"):
        if not file_path.is_file():
            continue
            
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                tokens = tokenizer(line.strip())
                
                cleaned_tokens = (
                    clean_token(t) for t in tokens
                )
                
                # Filter out None values
                counter.update(t for t in cleaned_tokens if t is not None)
    
    ranked_list = [word for word, _ in counter.most_common(top_k)]
    return ranked_list

def build_rank_dict(profile):
    return {ngram: rank for rank, ngram in enumerate(profile, start=1)}

hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
def hf_tokenize(text):
    return hf_tokenizer.tokenize(text)

def out_of_place_distance(profile_a, profile_b, max_rank=None):
    """
    Compute the out-of-place distance between two profiles.

    Args:
        profile_a: list of n-grams (ranked)
        profile_b: list of n-grams (ranked)
        max_rank: penalty for missing n-grams (default = len(profile_b))

    Returns:
        distance (int)
    """
    rank_b = build_rank_dict(profile_b)

    if max_rank is None:
        max_rank = len(profile_b)

    distance = 0

    for rank_a, ngram in enumerate(profile_a, start=1):
        if ngram in rank_b:
            distance += abs(rank_a - rank_b[ngram])
        else:
            distance += max_rank  # penalty for missing

    return distance

top_tokens = extract_top_tokens_from_dir("/Users/michal/Projects/Thesis/data/glotlid-corpus/v3.1/dan_Latn/", hf_tokenize)
top_grams  = extract_top_ngrams("/Users/michal/Projects/Thesis/data/glotlid-corpus/v3.1/dan_Latn", top_k=200)
print(top_tokens)
print(top_grams)
print(out_of_place_distance(top_grams,top_tokens))