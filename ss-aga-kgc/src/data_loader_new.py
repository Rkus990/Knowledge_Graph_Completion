from os.path import join
import pandas as pd
import numpy as np
import os
from src.knowledgegraph_pytorch import KnowledgeGraph
import numpy as np
from torch.utils.data import Dataset
import torch
from src.utils import get_language_list, get_subgraph_list, subgrarph_list_from_alignment
import copy
from transformers import AutoTokenizer
from tqdm import tqdm

bert_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/distiluse-base-multilingual-cased")

class ParseData(object):
    def __init__(self, args):
        self.data_path = args.data_path + args.dataset
        self.data_entity = self.data_path + "/entity/"
        self.data_kg = self.data_path + "/kg/"
        self.data_align = self.data_path + "/seed_alignlinks/"
        self.args = args

        self.target_kg = args.target_language
        self.kg_names = get_language_list(self.data_path) # all kg names, sorted
        self.num_kgs = len(self.kg_names)

        self.text_entities = self.entities_list()

    def load_data(self):
        '''
        # NOTE：ORDER IS SORTED（OS.LISTDIR）

        :return:
        1. X (bert embedding matrix), R (bert embedding matrix)
        2. Seed alignment (masked) for calculating alignment loss
        3. list of KG object
        '''


        entity_bert_emb = np.load(self.data_path + "/entity_embeddings.npy")
        # normalize features to be within [-1,1]
        entity_bert_emb = self.normalize_fature(entity_bert_emb)
        kg_object_dict, seeds_masked, seeds_all = self.create_KG_objects_and_alignment()

        self.num_relations = kg_object_dict[self.target_kg].num_relation * self.num_kgs

        return kg_object_dict, seeds_masked, seeds_all, entity_bert_emb


    def normalize_fature(self, input_embedding):
        input_max = input_embedding.max()
        input_min = input_embedding.min()

        # Normalize to [-1, 1]
        input_embedding_normalized = (input_embedding - input_min) * 2 / (input_max - input_min) - 1

        return input_embedding_normalized

    def load_all_to_all_seed_align_links(self):

        seeds_preserved = {}  # { (lang1, lang2): 2-col np.array }
        seeds_masked = {}
        seeds_all = {}
        for f in os.listdir(self.data_align):  # e.g. 'el-en.tsv'
            lang1 = f[0:2]
            lang2 = f[3:5]
            links = pd.read_csv(join(self.data_align, f), sep='\t',header=None).values.astype(int)  # [N,2] ndarray

            total_link_num = links.shape[0]
            if self.args.preserved_ratio != 1.0:
                preserved_idx = list(sorted(
                    np.random.choice(np.arange(total_link_num), int(total_link_num * self.args.preserved_ratio),
                                     replace=False)))
                masked_idx = list(filter(lambda x: x not in preserved_idx, np.arange(total_link_num)))

                assert len(masked_idx) + len(preserved_idx) == total_link_num

                preserved_links = links[preserved_idx, :]
                masked_links = links[masked_idx, :]

                seeds_masked[(lang1, lang2)] = torch.LongTensor(masked_links)
                seeds_all[(lang1, lang2)] = torch.LongTensor(links)
                seeds_preserved[(lang1, lang2)] = torch.LongTensor(preserved_links)  # to be used to generate the whole graph
            else:
                seeds_masked[(lang1, lang2)] = None
                seeds_all[(lang1, lang2)] = torch.LongTensor(links)
                seeds_preserved[(lang1, lang2)] = None


        return seeds_masked, seeds_all, seeds_preserved



    def create_KG_objects_and_alignment(self):
        '''
        Local index.
        :return:
        '''
        # INDEX ONLY!
        entity_base = 0
        relation_base = 0
        kg_objects_dict = {}

        for lang in self.kg_names:
            kg_train_data, kg_val_data, kg_test_data, entity_num, relation_num= self.load_kg_data(lang)  # use suffix 1 for supporter kg, 0 for target kg

            if lang == self.target_kg:
                is_supporter_kg = False
            else:
                is_supporter_kg = True

            kg_each = KnowledgeGraph(lang, kg_train_data, kg_val_data, kg_test_data, entity_num, relation_num, is_supporter_kg,
                                     entity_base, relation_base, self.args.device)
            kg_objects_dict[lang] = kg_each

            entity_base += entity_num
            relation_base += relation_num

        self.num_entities = entity_base

        # TODO: create subgraph list, using worker if possible
        for lang in self.kg_names:
            if lang == self.target_kg:
                is_target_KG = True
            else:
                is_target_KG = False
            kg_lang = kg_objects_dict[lang]
            subgraph_list_self =  get_subgraph_list(self.data_path, lang, is_target_KG, kg_lang.num_entity, self.args.num_hop, self.args.k, kg_lang.entity_id_base, kg_lang.relation_id_base)
            kg_lang.subgraph_list_kg = subgraph_list_self
            kg_lang.subgraph_list_align = copy.deepcopy(kg_lang.subgraph_list_kg)

        # TODO: adding alignment links
        seeds_masked, seeds_all, seeds_preserved = self.load_all_to_all_seed_align_links()

        # Add aligned_links to subgraph_list_kg
        self.add_subgraph_list_from_align(seeds_all,kg_objects_dict,is_kg_list = True)

        # Add aligned_links to subgraph_list_align
        self.add_subgraph_list_from_align(seeds_preserved,kg_objects_dict,is_kg_list = False)

        return kg_objects_dict, seeds_masked,seeds_all


    def add_subgraph_list_from_align(self, seeds, kg_objects_dict, is_kg_list = False):

        for (kg0_name, kg1_name) in seeds:
            kg0 = kg_objects_dict[kg0_name]
            kg1 = kg_objects_dict[kg1_name]
            align_links = seeds[(kg0_name, kg1_name)]
            subgrarph_list_from_alignment(align_links, kg0, kg1,is_kg_list)



    def load_kg_data(self, language):
        """
        Load triples and stats for each single KG
        :return: triples (n_triple, 3) np.int np.array
        TODO: change indexing to global one.
        """

        train_df = pd.read_csv(join(self.data_kg, language + '-train.tsv'), sep='\t', header=None,names=['v1', 'relation', 'v2'])
        val_df = pd.read_csv(join(self.data_kg, language + '-val.tsv'), sep='\t', header=None,names=['v1', 'relation', 'v2'])
        test_df = pd.read_csv(join(self.data_kg, language + '-test.tsv'), sep='\t', header=None,names=['v1', 'relation', 'v2'])

        # count entity num
        f = open(self.data_entity + language + '.tsv')
        lines = f.readlines()
        f.close()

        entity_num = len(lines) # TODO: check whetehr need to +1/-1

        relation_list = [line.rstrip() for line in open(join(self.data_path, 'relations.txt'))]
        relation_num = len(relation_list) + 1

        triples_train = train_df.values.astype(np.int)
        triples_val = val_df.values.astype(np.int)
        triples_test = test_df.values.astype(np.int)

        return torch.LongTensor(triples_train), torch.LongTensor(triples_val), torch.LongTensor(triples_test), entity_num, relation_num


    def entities_list(self):
        """
        Load entities from file
        :return: entities (n_entity, ) np.int np.array
        """
        languages = ['el', 'en', 'es', 'fr', 'ja']

        all_lang_entities = [l + '.tsv' for l in languages]

        all_input_ids = []
        all_attn_masks = []
        for f in all_lang_entities:
            entities = pd.read_csv(join(self.data_entity, f), sep='\t', header=None).values.astype(str).squeeze()
            for en in tqdm(entities):
                input_ids, attn_mask = self.bert_tokenize(en[en.find("resource") + len("resource/"):])
                all_input_ids.append(input_ids)
                all_attn_masks.append(attn_mask)
        
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_attn_masks = torch.tensor(all_attn_masks, dtype=torch.long)

        return all_input_ids, all_attn_masks

    def bert_tokenize(self, text):
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