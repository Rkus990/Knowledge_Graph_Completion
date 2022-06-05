from os.path import join
import pandas as pd
import numpy as np
import os
import numpy as np
import torch
import copy
from transformers import BertTokenizer
from tqdm import tqdm
import pudb

data_dir = "../datasetdbp5l/"
data_entity = data_dir + "entity/"
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

def bert_tokenize(text):
    encoded = bert_tokenizer.encode_plus(
        text=text,  # the sentence to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length = 10,  # maximum length of a sentence
        pad_to_max_length=True,  # Add [PAD]s
        return_attention_mask = True,  # Generate the attention mask
        truncation = True,
        # return_tensors = 'pt',  # ask the function to return PyTorch tensors
    )
    input_ids = encoded['input_ids']
    attn_mask = encoded['attention_mask']
    
    return input_ids, attn_mask

languages = ['el', 'en', 'es', 'fr', 'ja']

all_lang_entities = [l + '.tsv' for l in languages]

all_input_ids = []
all_attn_masks = []
for f in all_lang_entities:
    entities = pd.read_csv(join(data_entity, f), sep='\t', header=None).values.astype(str).squeeze()
    for en in tqdm(entities):
        input_ids, attn_mask = bert_tokenize(en[en.find("resource") + len("resource/"):])
        all_input_ids.append(input_ids)
        all_attn_masks.append(attn_mask)

# pu.db
all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
all_attn_masks = torch.tensor(all_attn_masks, dtype=torch.long)

torch.save(all_input_ids, data_dir + 'all_input_ids.pt')
torch.save(all_attn_masks, data_dir + 'all_attn_masks.pt')