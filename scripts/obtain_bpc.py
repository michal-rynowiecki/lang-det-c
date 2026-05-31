'''
This script determines the necessary number of data points to use from each language,
after which the average BPC stops changing significantly
'''

import sys
from pathlib import Path

# Add parent directory to Python's module search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse

import thesis.paths as paths

import pandas as pd
import numpy as np

from src.bpc import lang_len

def main():
    parser = argparse.ArgumentParser(description="Determine BPC for a given language and token")
    
    parser.add_argument("-lm", "--language_model", required=True, 
        help="Hugging Face model name")
    
    parser.add_argument("-a", "--alpha", type=float, default=0.1,
        help="The spread that r consecutive values must be in")

    parser.add_argument("-r", "--rang", type=int, default=5,
        help="Number of consecutive values")

    parser.add_argument("-en", "--encoder", action="store_true",
        help="Is the model an encoder?")

    parser.add_argument("-l", "--langs", required=False,
        help="Manually provided list of languages")
   
    args = parser.parse_args()

    language_model  = args.language_model
    alpha           = args.alpha
    rang            = args.rang
    encoder         = args.encoder
    langs           = args.langs
    
    parser.set_defaults(encoder=False)

    # Run the src method for calculating the bpc and optimal number of data points for BPC
    lang_len(args.language_model, alpha, rang, encoder, langs)

main()