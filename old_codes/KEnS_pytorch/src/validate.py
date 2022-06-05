from __future__ import division
import time
import pandas as pd
import src.param as param
import numpy as np
from enum import Enum
from src.ensemble import voting, filt_voting_with_model_weight
from src.model_pytorch import KNN_finder
import logging
from os.path import join
import torch
from tqdm import tqdm


class TestMode(Enum):
    KG0 = 'kg0'
    KG1 = 'kg1'
    Transfer = 'link_redirect'
    VOTING = 'voting'


class MultiModelTester:
    def __init__(self, target_kg, supporter_kgs,device):
        """
        :param target_kg: KnowledgeGraph object
        :param support_kgs: list[KnowledgeGraph]
        """
        self.target_kg = target_kg
        self.supporter_kgs = supporter_kgs
        self.device = device


    def flatten_kNN_finder_output(self, kNN_finder_output):

        """
        :param kNN_finder_output: [array([[id1, id2, ...]]), array([[score1, score2, ...]])]
        :return: [(id1, score1), (id2, score2), ...]
        also, change back to cpu
        """
        # indices, scores = kNN_finder_output
        # kNN_finder returns [array([[id1, id2, ...]]), array([[score1, score2, ...]])]
        indices = kNN_finder_output[0][0]  # flatten [[id1, id2, ...]] -> [id1, id2, ...]
        scores = kNN_finder_output[1][0]  # flatten [[score1, score2, ...]] -> [score1, score2, ...]
        topk_indices_scores = [(indices[i].detach().data.item(), scores[i].detach().data.item()) for i in range(len(indices))]
        return topk_indices_scores

    def extract_entities(self, id_score_tuples):
        return [ent_id for ent_id, score in id_score_tuples]

    def predict(self, h, r, mode, supporter_kg=None, voting_function=None, filtered_reordered_embedding_matrix = None):
        """
        If mode is LINK_TRANSFER and h not in align_dict, return [[]]
        :param h: np.int32
        :param r: np.int32
        :param mode:
        :param supporter_kg: needed when mode == KG1 or LINK_REDIRECT. None for KG0 or VOTING
        :return:
        """
        if (mode == TestMode.KG1 or mode == TestMode.Transfer) and (not supporter_kg):
            supporter_kg = self.supporter_kgs[0]
            print('TestMode: %s but no kg specified. Using supporter_kgs[0], language: %s' % (mode, supporter_kg.lang))

        if mode == TestMode.KG0:
            # input shape must be (1,1) to feed h,r into kNN_finder  (batch_size=1, column=1)
            h = torch.reshape(h, (1,))
            r = torch.reshape(r, (1,))
            top_k_indices_scores = KNN_finder(self.target_kg.model,h, r, self.target_kg.model.entity_embedding_layer,h.device)  # [array([[id1, id2, ...]]), array([[score1, score2, ...]])]
            top_k = self.flatten_kNN_finder_output(top_k_indices_scores)  # [(id1, score1), (id2, score2), ...]
            return top_k

        elif mode == TestMode.KG1:  # query on supporter kgs
            # input shape must be (1,1) to feed h,r into kNN_finder  (batch_size=1, column=1)
            h = torch.reshape(h, (1, ))
            r = torch.reshape(r, (1, ))
            top_k_indices_scores = KNN_finder(
                supporter_kg.model, h, r, supporter_kg.model.entity_embedding_layer,h.device)
            top_k = self.flatten_kNN_finder_output(top_k_indices_scores)  # [(id1, score1), (id2, score2), ...]
            return top_k

        elif mode == TestMode.Transfer:
            return self.__predict_by_knowledge_transfer(h, r, supporter_kg, filtered_reordered_embedding_matrix = filtered_reordered_embedding_matrix)

        elif mode == TestMode.VOTING:
            choice_lists = []  # list[list[(item,score)]]

            choices0 = self.predict(h, r, mode=TestMode.KG0, supporter_kg=None)
            choice_lists.append(choices0)

            for sup_kg in self.supporter_kgs:
                choices1 = self.predict(h, r, mode=TestMode.Transfer, supporter_kg=sup_kg)
                choice_lists.append(choices1)
            if voting_function is None:
                return voting(choice_lists, param.k)
            else:
                return voting_function(choice_lists, param.k)

    def __predict_by_knowledge_transfer(self, h0, r0, supporter_kg=None, filtered_reordered_embedding_matrix = None):
        """
        :param h0: numpy index
        :param r0: shape (1,1) [[id]]
        :return:
        """
        device_original = h0.device
        h0 = h0.detach().data.item() # convert back to np
        if h0 not in supporter_kg.dict0to1:
            return []
        h1 = torch.LongTensor([supporter_kg.dict0to1[h0]]).to(device_original)
        r1 = r0

        # filtered setting
        h1 = torch.reshape(h1, (1, ))
        r1 = torch.reshape(r1, (1, ))
        # The entities will directly be entity id in kg0 if we use filtered_reordered_embedding matrix
        t0_and_scores = KNN_finder(
            supporter_kg.model,h1, r1,filtered_reordered_embedding_matrix,device_original)
        t0_and_scores = self.flatten_kNN_finder_output(t0_and_scores)  # [(id1, score1), (id2, score2), ...]

        return t0_and_scores

    def test(self, mode, supporter_kg=None, voting_function=None):
        """

        Compute Hits@10 on first param.n_test test samples
        :param mode: TestMode. For LINK_TRANSFER, triples without links will be skipped
        :param supporter_kg: needed when mode == KG1 or LINK_REDIRECT. None for KG0 or VOTING
        :param voting_function: used when mode==VOTING. Default: vote by count
        :return:
        """

        time0 = time.time()
        hits = 0
        samples = min(param.n_test, self.target_kg.h_test.shape[0])

        # used when mode is LINK_TRANSFER
        linked_triple = samples
        retrieved_t0 = 0
        no_retreived = 0

        for i in range(samples):
            if mode == TestMode.KG1:
                if i >= supporter_kg.h_test.shape[0]:
                    break
                # input shape must be (1,1) to feed h,r into kNN_finder  (batch_size=1, column=1)
                h, r, t = supporter_kg.h_test[i].to(self.device), supporter_kg.r_test[i].to(self.device), supporter_kg.t_test[i]
            else:
                # input shape must be (1,1) to feed h,r into kNN_finder  (batch_size=1, column=1)
                h, r, t = self.target_kg.h_test[i].to(self.device), self.target_kg.r_test[i].to(self.device), self.target_kg.t_test[i]


            with torch.no_grad():
                top_k = self.predict(h, r, mode, supporter_kg=supporter_kg,
                                     voting_function=voting_function)  # list[(idx,score)] Length may be less than k in LINK_TRANSFER MODE
                top_k_entities = self.extract_entities(top_k)
                if t in top_k_entities:
                    # print('hit')
                    hits += 1
                else:
                    if param.verbose:
                        # print the wrong test cases
                        print('Test case with wrong result: h,r,t', h, r, t)
                        print(top_k)

                if mode == TestMode.Transfer:
                    if h not in supporter_kg.dict0to1:
                        linked_triple -= 1
                    else:
                        retrieved_t0 += len(top_k)
                        if len(top_k) == 0:
                            no_retreived += 1


        # logging.info('===Validation %s===' % mode)
        if mode == TestMode.Transfer:
            hit_ratio = hits / linked_triple
            logging.info(
                'Hits@%d in %d linked triples in %d: %f' % (param.k, linked_triple, samples, hits / linked_triple))
            logging.info('Average retrieved t0: %f' % (retrieved_t0 / linked_triple))
            logging.info('%d queries have h link but no retrieved t' % (no_retreived))
        else:
            hit_ratio = hits / samples
            logging.info('Hits@%d (%d triples): %f' % (param.k, samples, hit_ratio))

        print('time: %s' % (time.time() - time0))
        return hit_ratio


    def test_all_hit_k(self,is_lifted = False,n=10):
        time0 = time.time()
        hits = 0

        testfile = join('data/kg', self.target_kg.lang + '-test.tsv')
        testcases = pd.read_csv(testfile, sep='\t', header=None).values
        param.n_test = testcases.shape[0]

        samples = param.n_test

        hr2t_train = hr2t_from_train_set('data/kg', self.target_kg.lang)  # used for filtered test setting

        num_in_hr2t = 0
        for i in range(samples):

            # input shape must be (1,1) to feed h,r into kNN_finder  (batch_size=1, column=1)
            h, r, t = torch.LongTensor([testcases[i, 0]]).to(self.device), torch.LongTensor([testcases[i, 1]]).to(
                self.device), testcases[i, 2]

            with torch.no_grad():
                top_k = self.predict(h, r, mode = TestMode.KG0, supporter_kg=None,
                                     voting_function=None)  # list[(idx,score)] Length may be less than k in LINK_TRANSFER MODE
                top_k_entities = self.extract_entities(top_k)[:n]

                if is_lifted:
                    h_np = h.detach().data.item()
                    r_np = r.detach().data.item()
                    if (h_np, r_np) in hr2t_train:  # filter
                        top_k_entities = [e for e in top_k_entities if e not in hr2t_train[(h_np, r_np)]]
                        num_in_hr2t += 1

                if t in top_k_entities:
                    # print('hit')
                    hits += 1


        hit_ratio = hits / samples
        logging.info('Hits@%d (%d triples) by itself: %f' % (param.k, samples, hit_ratio))

        # Test results on other KGs
        computed_filtered_reordered_embedding_matrix_list = {}
        for kg1 in self.supporter_kgs:
            filtered_reordered_embedding_matrix_each = kg1.get_filtered_reordered_embedding_matrix(
                self.target_kg.num_entity)
            computed_filtered_reordered_embedding_matrix_list[kg1.lang] = filtered_reordered_embedding_matrix_each


        for kg1 in self.supporter_kgs:
            self.test_supporter_kg_results_each(kg1,computed_filtered_reordered_embedding_matrix_list[kg1.lang],is_lifted = is_lifted, n = n )


        return hit_ratio



    def test_supporter_kg_results_each(self,kg1,kg1_filtered_emb,is_lifted = False,n=10):
        hits = 0

        testfile = join('data/kg', self.target_kg.lang + '-test.tsv')
        testcases = pd.read_csv(testfile, sep='\t', header=None).values

        samples = testcases.shape[0]
        hr2t_train = hr2t_from_train_set('data/kg', self.target_kg.lang)  # used for filtered test setting

        with torch.no_grad():
            for i in range(samples):

                # input shape must be (1,1) to feed h,r into kNN_finder  (batch_size=1, column=1)
                h, r, t = torch.LongTensor([testcases[i, 0]]).to(self.device), torch.LongTensor([testcases[i, 1]]).to(
                    self.device), testcases[i, 2]

                filtered_reordered_embedding_matrix = kg1_filtered_emb
                kg1_top_k = self.predict(h, r, mode=TestMode.Transfer, supporter_kg=kg1,
                                         filtered_reordered_embedding_matrix=filtered_reordered_embedding_matrix)
                top_k_entities = self.extract_entities(kg1_top_k)[:n]

                if is_lifted:
                    h_np = h.detach().data.item()
                    r_np = r.detach().data.item()
                    if (h_np, r_np) in hr2t_train:  # filter
                        top_k_entities = [e for e in top_k_entities if e not in hr2t_train[(h_np, r_np)]]

                if t in top_k_entities:
                    # print('hit')
                    hits += 1

        hit_ratio = hits / samples
        logging.info('Hits@%d (%d triples) from KG %s: %f' % (param.k, samples, kg1.lang, hit_ratio))

        return hit_ratio



    def test_and_record_results(self, testcases):
        # testcases are numpy
        samples = testcases.shape[0]

        results = []  # list[(h ,r, t, nominations)].
        # nominations=list[list[(entity_idx, score)]], len(nominations)=1(target)+len(supporter_kgs)

        computed_filtered_reordered_embedding_matrix_list = []
        with torch.no_grad():
            # pre-compute it!
            computed_filtered_reordered_embedding_matrix_list = {}
            for kg1 in self.supporter_kgs:
                filtered_reordered_embedding_matrix_each = kg1.get_filtered_reordered_embedding_matrix(self.target_kg.num_entity)
                computed_filtered_reordered_embedding_matrix_list[kg1.lang] = filtered_reordered_embedding_matrix_each


            for i in tqdm(range(samples)):
                # input shape must be (1,1) to feed h,r into kNN_finder  (batch_size=1, column=1)

                h, r, t = torch.LongTensor([testcases[i, 0]]).to(self.device), torch.LongTensor([testcases[i, 1]]).to(self.device), testcases[i, 2]

                nominations = []  # list[list[(entity_idx, score)]], len(nominations)=1(target)+len(supporter_kgs)
                kg0_top_k = self.predict(h, r, TestMode.KG0)
                nominations.append(kg0_top_k)

                for kg1 in self.supporter_kgs:
                    filtered_reordered_embedding_matrix = computed_filtered_reordered_embedding_matrix_list[kg1.lang]
                    kg1_top_k = self.predict(h, r, mode=TestMode.Transfer, supporter_kg=kg1,filtered_reordered_embedding_matrix = filtered_reordered_embedding_matrix)
                    nominations.append(kg1_top_k)

                results.append([h.detach().data.item(), r.detach().data.item(), t] + nominations)  # [h,r,t, list[(idx,score)], list[(idx,score)], ...]


        results_df = pd.DataFrame(results, columns=['h', 'r', 't', self.target_kg.lang] + [kg1.lang for kg1 in
                                                                                           self.supporter_kgs])
        # results_df = pd.DataFrame(results, columns=['h', 'r', 't', self.target_kg.lang])                                                                                                                                                                 self.supporter_kgs])

        # print(results_df.head(5))

        return results_df



def extract_entities(id_score_tuples):
    return [ent_id for ent_id, score in id_score_tuples]



def filt_hits_at_n(results, lang, hr2t_train, n):
    """
    Filtered setting Hits@n when testing
    :param hr2t_train: {(h,r):set(t)}
    :param results: df, h,r,t, lang
    :return:
    """
    hits = 0
    for index, row in results.iterrows():
        t = row['t']
        predictions = row[lang]  # list[(entity,socre)]

        predictions = extract_entities(predictions)
        if (row['h'], row['r']) in hr2t_train:  # filter
            h, r = row['h'], row['r']
            predictions = [e for e in predictions if e not in hr2t_train[(h,r)]]
        predictions = predictions[:n]  # top n
        if t in predictions:
            hits += 1
    hits_ratio = hits / results.shape[0]
    logging.info('Hits@%d (%d triples)(filt): %.4f' % (n, results.shape[0], hits_ratio))
    return hits_ratio


def filt_ensemble_hits_at_n(results, weight, model_lists, hr2t_train, n):
    """
    Filtered setting Hits@n
    :param results: a result dataframe of predictions. [h, r, t, el, ja, en, ...]
    :param weight: dict {entity: {model:weight}}
    :param model_lists: a list. ['el', 'es','fr','ja','en']
    :return:
    """
    hits = 0
    no_weight = 0
    center = 'h'  # the key in weights
    for index, row in results.iterrows():
        t = row['t']
        predictions = [row[lang][:n] for lang in model_lists]

        if row[center] in weight:
            model_weights_dict = weight[row[center]]  # {model:weight}
            model_weights = [model_weights_dict[lang] for lang in model_lists]
        elif param.lang in weight:  # weight is {model:weight}
            model_weights = [weight[lang] for lang in model_lists]
        else:
            model_weights = [1 for lang in model_lists]  # majority vote

        if (row['h'], row['r']) in hr2t_train:  # filtered setting
            train_ts = hr2t_train[(row['h'], row['r'])]
        else:
            train_ts = set()  # empty set

        topk = filt_voting_with_model_weight(choice_lists=predictions,
                                             k=n,
                                             train_ts=train_ts,
                                             model_weights=model_weights)

        if t in extract_entities(topk):
            hits += 1
    hits_ratio = hits / results.shape[0]
    logging.info('Hits@%d (%d triples): %.4f' % (n, results.shape[0], hits_ratio))




def hr2t_from_train_set(data_dir, target_lang):
    train_df = pd.read_csv(join(data_dir, f'{target_lang}-train.tsv'), sep='\t')
    tripleset = set([tuple([h,r,t]) for h,r,t in (train_df.values)])

    hr2t = {}  # {(h,r):set(t)}
    for tp in tripleset:
        h,r,t=tp[0],tp[1],tp[2]
        if (h,r) not in hr2t:
            hr2t[(h,r)] = set()
        hr2t[(h,r)].add(t)
    return hr2t




