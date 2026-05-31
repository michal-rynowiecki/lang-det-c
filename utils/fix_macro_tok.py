import json
from pathlib import Path
from huggingface_hub import ModelCard
from huggingface_hub.utils import EntryNotFoundError

root_dir = Path("../data/tokenizer_based")

macro_mapping = {
    'est': 'ekk', 'et': 'ekk', 'zho': 'cmn', 'chi': 'cmn', 'zh': 'cmn',
    'grn': 'gug', 'gn': 'gug', 'nep': 'npi', 'ne': 'npi',
    'lav': 'lvs', 'lv': 'lvs', 'ara': 'arb', 'ar': 'arb',
    'ori': 'ory', 'or':  'ory', 'msa': 'zlm', 'may': 'zlm', 'ms': 'zlm',
    'kom': 'kpv', 'kv': 'kpv'
}


for folder in root_dir.iterdir():
    if not folder.is_dir(): continue
        
    for subfolder in folder.iterdir():
        if not subfolder.is_dir(): continue
        
        
        model_name = f"{folder.name}/{subfolder.name}"
        card = ModelCard.load(model_name)
        print(model_name)
            
        # Coerce to list if string
        langs = card.data.get("language", [])
        langs = [langs] if isinstance(langs, str) else langs
            
        # Check if ANY language matches (prevents duplicate runs)
        if any(lang in macro_mapping for lang in langs):
            matching_langs = [macro_mapping[lang] for lang in langs if lang in macro_mapping]
            print(matching_langs)


            positive = subfolder / "True.json"
            negative = subfolder / "False.json"
            temp_negative = subfolder / "temp_False.json"

            # Open both files once. Iterate line-by-line instead of .readlines()
            with open(negative, 'r') as neg_file, open(positive, 'a') as pos_file, open(temp_negative, 'w') as temp_neg:
                for index, line in enumerate(neg_file):
                    line = line.strip()
                    if not line: continue
                        
                    try:
                        data = json.loads(line)
                        first_key = data["language"][:3]
                        
                        if first_key in matching_langs:
                            print(data)
                            pos_file.write(line)
                            pos_file.write('\n')
                        else:
                            temp_neg.write(line)
                            temp_neg.write('\n')
                            
                    except json.JSONDecodeError:
                        # Safely keep unparseable lines in the negative file
                        temp_neg.write(line)
                        temp_neg.write('\n')
                        
            # Replace the old negative file with the filtered temporary file
            temp_negative.replace(negative)