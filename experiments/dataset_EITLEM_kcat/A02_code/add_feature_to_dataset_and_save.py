import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import json
import joblib
import os
pd.set_option('display.max_columns', None)
with open('./../A01_dataset/kcat_data.json', 'r') as file:
    data = json.load(file)
data= pd.DataFrame(data)
data = data.rename(columns={"ec": "ECNumber", "smiles": "Smiles","value":"label","substrate":"Substrate","sequence":"Sequence"})
print(data.head())
# data['label_type']= data['label'].apply(type)
print(data['label'].apply(type).value_counts())

# Find and display string labels
string_labels = data[data['label'].apply(lambda x: isinstance(x, str))]
print(f"\nNumber of string labels: {len(string_labels)}")

# if there are string labels, we have to turn them into numeric or NaN
if len(string_labels) > 0:
    print("\nUnique string label values:")
    unique_str_labels = string_labels['label'].unique()
    print(unique_str_labels)
    
    print("\nSample rows with string labels:")
    print(string_labels[['label', 'Smiles', 'ECNumber']].head(10))
    
    # Try to convert string labels to numeric
    def convert_label_to_numeric(label):
        if isinstance(label, str):
            try:
                # Try to convert to float
                return float(label)
            except ValueError:
                # If conversion fails, check for common string patterns
                label_lower = label.lower().strip()
                if label_lower in ['na', 'nan', 'null', 'none', '']:
                    return np.nan
                elif label_lower in ['inf', 'infinity', '+inf']:
                    return np.inf
                elif label_lower in ['-inf', '-infinity']:
                    return -np.inf
                else:
                    print(f"Warning: Cannot convert '{label}' to numeric. Setting to NaN.")
                    return np.nan
        return label
    
    # Apply conversion
    print("\nConverting string labels to numeric...")
    data['label'] = data['label'].apply(convert_label_to_numeric)
    
    # Show results after conversion
    print("Label types after conversion:")
    print(data['label'].apply(type).value_counts())
    
    # Check for any remaining issues
    remaining_strings = data[data['label'].apply(lambda x: isinstance(x, str))]
    if len(remaining_strings) > 0:
        print(f"\nWarning: {len(remaining_strings)} labels still remain as strings:")
        print(remaining_strings['label'].unique())
        print("\nRows with unconverted string labels:")
        print(remaining_strings[['label', 'Smiles', 'ECNumber', 'Substrate', 'Sequence']].head(20))
        
        # Show row indices for debugging
        print(f"\nRow indices with string labels: {remaining_strings.index.tolist()}")
    
    # Also check for NaN values created during conversion
    nan_labels = data[data['label'].isna()]
    if len(nan_labels) > 0:
        print(f"\nNumber of NaN labels after conversion: {len(nan_labels)}")
        print("Sample rows with NaN labels:")
        print(nan_labels[['label', 'Smiles', 'ECNumber']].head(10))

from transformers import AutoTokenizer, AutoModel
import torch
print(torch.__version__)

# Define local cache directories
chem_model_cache = "./models/ChemBERTa-zinc-base-v1"
esm_model_cache = "./models/esm2_t6_8M_UR50D"

# Create cache directories if they don't exist
os.makedirs(chem_model_cache, exist_ok=True)
os.makedirs(esm_model_cache, exist_ok=True)

# Load ChemRoBERTa tokenizer and model with local caching
print("Loading ChemRoBERTa model...")
if os.path.exists(os.path.join(chem_model_cache, "config.json")):
    print("Loading from local cache...")
    chem_tokenizer = AutoTokenizer.from_pretrained(chem_model_cache)
    chem_model = AutoModel.from_pretrained(chem_model_cache)
else:
    print("Downloading and caching ChemRoBERTa model...")
    chem_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    chem_model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    # Save to local cache
    chem_tokenizer.save_pretrained(chem_model_cache)
    chem_model.save_pretrained(chem_model_cache)
    print("ChemRoBERTa model cached locally")

# Load ESM2 tokenizer and model with local caching
print("Loading ESM2 model...")
if os.path.exists(os.path.join(esm_model_cache, "config.json")):
    print("Loading from local cache...")
    esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_cache)
    esm_model = AutoModel.from_pretrained(esm_model_cache)
else:
    print("Downloading and caching ESM2 model...")
    esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    esm_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    # Save to local cache
    esm_tokenizer.save_pretrained(esm_model_cache)
    esm_model.save_pretrained(esm_model_cache)
    print("ESM2 model cached locally")

def extract_chem_features(smiles):
    """Extract ChemRoBERTa features from SMILES."""
    try:
        tokens = chem_tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = chem_model(**tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings
    except:
        return np.zeros(768)  # Return zero vector if extraction fails

def extract_esm_features(sequence):
    """Extract ESM2 features from protein sequence."""
    try:
        tokens = esm_tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = esm_model(**tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings
    except:
        return np.zeros(768)  # Return zero vector if extraction fails
# Extract unique mols and proteins
unique_mols = data[['Smiles']].drop_duplicates()
unique_proteins = data[['Sequence']].drop_duplicates()
# Extract features for unique mols
tqdm.pandas()  # Enable progress bar for pandas
unique_mols['metabolite_features'] = unique_mols['Smiles'].progress_apply(extract_chem_features)

# Extract features for unique proteins
unique_proteins['protein_features'] = unique_proteins['Sequence'].progress_apply(extract_esm_features)
# Merge features back into the combined dataframe
data = data.merge(unique_mols, on='Smiles', how='left')
data = data.merge(unique_proteins, on='Sequence', how='left')
joblib.dump(data, './../A01_dataset/kcat_data_with_features.joblib')