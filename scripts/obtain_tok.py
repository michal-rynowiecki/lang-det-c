'''
This script determines the necessary number of data points to use from each language,
after which the average BPC stops changing significantly
'''

import sys
from pathlib import Path

# Add parent directory to Python's module search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse

import pandas as pd
import numpy as np

from src.tok import tokenizer_based

def main():
    parser = argparse.ArgumentParser(description="Determine BPC for a given language and token")
    
    parser.add_argument("-p", "--path", required=True, 
        help="path to the model whose tokenizer to use")
    
    parser.add_argument("-l", "--langs", required=False, 
        help="path to the model whose tokenizer to use")
   
    args = parser.parse_args()

    path  = args.path
    langs = args.langs
    
    tokenizer_based(path, langs)
    
main()