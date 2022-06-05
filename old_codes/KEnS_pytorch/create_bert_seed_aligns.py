import os
from random import seed
from transformers import BertTokenizer, BertModel
from os import listdir
from os.path import isfile, join
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pudb
from tqdm import tqdm
from zmq import device
import torch
import numpy as np

device = "cuda:3"
embedder = SentenceTransformer('distiluse-base-multilingual-cased', device=device)

data_dir = 'data'
entity_dir = data_dir + '/entity/'
seedlinks_dir = data_dir + '/seed_alignlinks2/'

all_lang_entities = [f for f in listdir(entity_dir) if isfile(join(entity_dir, f))]

entities_emd = {}
for f in all_lang_entities:
    entities = pd.read_csv(join(entity_dir, f), sep='\t', header=None).values.astype(str).squeeze()
    texts = [ en[en.find("resource") + len("resource/"):] for en in entities]
    embeds = [ ]
    for i in tqdm(range(len(texts))):
        text = texts[i]
        output = embedder.encode(text)
        embeds.append(output)
    entities_emd[f[:2]] = embeds

if seedlinks_dir in listdir(data_dir):
    os.system("rm -r " + seedlinks_dir)
os.system("mkdir " + seedlinks_dir)

for i in range(len(all_lang_entities)):
    for j in range(i + 1, len(all_lang_entities)):
        l1, l2 = all_lang_entities[i][:2], all_lang_entities[j][:2]

        seeds_len = 5000
        if l1 == 'el' or l2 == 'el':
            seeds_len = 2000
        
        embeds1 = torch.FloatTensor(entities_emd[l1]).to(device)
        embeds2 = torch.FloatTensor(entities_emd[l2]).to(device)
        cosine_sims_matrix = torch.mm(embeds1, torch.transpose(embeds2, 0, 1)).cpu()
        rows = cosine_sims_matrix.shape[0]
        cols = cosine_sims_matrix.shape[1]
        highest_values = torch.topk(torch.flatten(cosine_sims_matrix), seeds_len)
        indices = highest_values.indices
        # pu.db
        pairs = []
        for k in indices:
            pairs.append([ float((int(k) // cols) + 1), float((int(k) % cols) + 1)])
        df = pd.DataFrame(pairs)
        df.to_csv( seedlinks_dir + l1 +  "-" + l2 + ".tsv", header=False, index=False, sep="\t")
