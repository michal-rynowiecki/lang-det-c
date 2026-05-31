# Language Detection

This repository contains the code and data for training and evaluating language detection classifiers based on tokenizer metrics (BPC – bytes per character – and token counts). The project supports multiple tokenizer-based and language-length-based models, and includes scripts to reproduce all results reported in the paper.

## Repository Structure

```
language-detection/
├── data/
│   ├── lang_lengths/         # HF repos → model names
│   ├── tokenizer_based/      # HF repos → model names
│   └── ...                   # pre‑computed metrics (zipped)
├── scripts/
│   ├── download_dataset.py   # downloads the GlotLid dataset
│   ├── obtain_bpc.py         # computes BPC for a given model
│   ├── obtain_tok.py         # computes token counts for a tokenizer
│   └── detect.py             # trains the classifier
├── src/                      # core modules (model, hyperparameter search)
├── utils/                    # helper functions
└── requirements.txt
```

## Setup

1. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

First, download the GlotLid dataset (required for the following steps).  
This dataset is large; ensure sufficient disk space.

```bash
python scripts/download_dataset.py
```

After running the script, the complete dataset will be placed in the `data/` directory.

## Computing Metrics

Two scripts are provided to obtain metrics.  
**Note:** Pre‑computed metrics are already available as zip files in the `data/` folder. You can extract them into the corresponding subdirectories and skip the computation steps below.

### `obtain_bpc.py`

Computes bytes per character (BPC) for a given model.

**Arguments:**
- `--model_name` (required): Name of the Hugging Face model.
- `--is_encoder` (required): Whether the model is an encoder (`True`/`False`).
- `--languages` (optional): List of languages to evaluate. If not provided, defaults to all languages available from the Hugging Face path.

**Example:**
```bash
python scripts/obtain_bpc.py --model_name "bert-base-uncased" --is_encoder True --languages en fr de
```

### `obtain_tok.py`

Computes token counts for a given tokenizer.

**Arguments:**
- `-p` (required): Path to the tokenizer.

**Example:**
```bash
python scripts/obtain_tok.py -p "path/to/tokenizer"
```

## Training the Classifier

Use `detect.py` to train the language detection classifier.  
Training is very fast (approx. 30 seconds on the pre‑computed metrics). The repository does not include a pre‑trained classifier to encourage reproducibility.

```bash
python scripts/detect.py [arguments]
```

You can pass various arguments, for example to run a hyperparameter search. For details, inspect:

- `src/model.py`
- `src/hyperparameter_search.py`

All predictions for every model described in the paper are available as a pickle CSV file in the repository.