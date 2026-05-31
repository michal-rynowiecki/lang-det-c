import pandas as pd
import thesis.paths as paths

def convert(languages_str):
    """This function converts a string of full language names to their ISO 639-1 (Part1) format,
    which is the default for Hugging Face."""
    languages = languages_str.strip("[]").split(", ")
    lang_df = pd.read_csv(f'{paths.DATA_DIR}/language_codes.txt', sep='\t')
    updated_list = []
    for language in languages:
        match = lang_df[lang_df['Ref_Name'] == language]
        if not match.empty:
            new_lang = match['Part1'].values[0]
            if pd.isna(new_lang):
                new_lang = match['Part2t'].values[0]
            if pd.isna(new_lang):
                new_lang = match['Part2b'].values[0]
            if pd.isna(new_lang):
                new_lang = match['Id'].values[0]
        else:
            new_lang = None
        updated_list.append(new_lang)
    return updated_list