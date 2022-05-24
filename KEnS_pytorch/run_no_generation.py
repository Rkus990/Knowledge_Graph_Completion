#!/usr/bin/env python
# coding: utf-8


"""
Working directory: project root

In all variable names, 0 denotes the target/sparse kg and 1 denotes the source/dense kg.
"""
import src.param as param


import os
from os.path import join

print('Current working dir', os.getcwd())
import sys

if './' not in sys.path:
    sys.path.append('./')

import torch
import pandas as pd
import src.param as param
from src.model_pytorch import save_model_structure
from src.data_loader import load_support_kgs, load_target_kg, load_all_to_all_seed_align_links
from src.validate import MultiModelTester, TestMode
import numpy as np
import logging
from src.model_pytorch import extend_seed_align_links
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim



def set_logger(param, model_dir):
    '''
    Write logs to checkpoint and console
    '''

    log_file = os.path.join(model_dir, 'train.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='run.py [<args>] [-h | --help]'
    )
    parser.add_argument('-l', '--target_language', type=str, default = 'ja', choices=['ja', 'el', 'es', 'fr', 'en'], help="target kg")
    parser.add_argument('-m', '--knowledge_model', default='rotate', type=str, choices=['transe', 'rotate'])
    parser.add_argument('--use_default', action="store_true", help="Use default setting. This will override every setting except for targe_langauge and knowledge_model")
    parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help="learning rate for knowledge model")
    parser.add_argument('--align_lr', default=1e-3, type=float, help="learning rate for knowledge model")
    parser.add_argument('-b', '--batch_size', default=1024, type=int, help="batch size of queries")
    parser.add_argument('-d', '--dim', default=400, type=int)
    parser.add_argument('--optimizer', type=str, default="AdamW", help='Adam, AdamW')
    parser.add_argument('--transe_margin', default=0.3, type=float)
    parser.add_argument('--rotate_gamma', default=24, type=float)
    parser.add_argument('--reg_scale', default=1e-5, type=float, help="scale for regularization")
    parser.add_argument('--base_align_step', default=5, type=float, help="how many align model epoch to train before switching to knowledge model")
    parser.add_argument('--knowledge_step_ratio', default=2, type=float, help="how many knowledge model epochs for each align epoch")
    parser.add_argument('--round', default=5, type=float,
                        help="how many rounds to train")
    parser.add_argument('-k', default=10, type=int, help="how many nominations to consider")

    return parser.parse_args(args)

def set_params(args):
    param.lang = args.target_language
    param.knowledge = args.knowledge_model

    if args.use_default:
        if param.knowledge == 'transe':
            param.epoch10 = 10
            param.epoch11 = 10
            param.epoch2 = 5
            param.lr = 1e-3
            param.dim = 300
            param.round = 2
        elif param.knowledge == 'rotate':
            param.epoch10 = 100
            param.epoch11 = 100
            param.epoch2 = 5
            param.lr = 1e-2
            param.dim = 400
            param.round = 3

            # # For debug usage
            # param.epoch10 = 1
            # param.epoch11 = 1
            # param.epoch2 = 1
            # param.lr = 1e-2
            # param.dim = 400
            # param.round = 1
    else:
        param.dim = args.dim
        param.lr = args.learning_rate
        param.batch_size = args.batch_size
        param.epoch2 = args.base_align_step
        param.epoch10 = param.epoch11 = args.base_align_step * args.knowledge_step_ratio
        param.gamma = args.rotate_gamma
        param.margin = args.transe_margin
        param.reg_scale = args.reg_scale
        param.round = args.round

def train_align_batch(align_model,align_data,optimizer):
    # TODO: add logging

    align_data_loader = DataLoader(align_data,batch_size=param.batch_size, shuffle=True)


    kg_name0 = align_model.other_lang
    kg_name1 = align_model.lang
    for one_epoch in range(param.epoch2):
        align_loss = []
        for align_each in align_data_loader:
            optimizer.zero_grad()
            loss = align_model.calculate_align_loss(align_each[:, 0], align_each[:, 1])
            loss.backward()
            optimizer.step()

            align_loss.append(loss.item())

            del loss
            torch.cuda.empty_cache()


        logging.info('Align {:s} {:s} Epoch {:d} [Train Align Loss {:.6f}|'.format(
            kg_name0,
            kg_name1,
            one_epoch,
            np.mean(align_loss)))



def train_kg_batch(kg,optimizer,num_epoch,device):
    # TODO : add logging

    kg_batch_generator = kg.generate_batch_data(kg.h_train,kg.r_train,kg.t_train,batch_size = param.batch_size,shuffle=False)

    # # Adjust learning rate for kg module
    # for param_group in optimizer.param_groups:  # 在每次更新参数前迭代更改学习率
    #     param_group["lr"] = param.lr


    for one_epoch in range(num_epoch):
        kg_loss = []
        for kg_batch_each in kg_batch_generator:
            kg_batch_each = kg_batch_each.to(device)
            # print(type(kg_batch_each))
            optimizer.zero_grad()
            loss = kg.model(kg_batch_each)
            loss.backward()
            optimizer.step()

            kg_loss.append(loss.item())

            del loss
            torch.cuda.empty_cache()

        logging.info('KG {:s} Epoch {:d} [Train KG Loss {:.6f}|'.format(
            kg.lang,
            one_epoch,
            np.mean(kg_loss)))





def main(args):
    ############ CPU AND GPU related, Mode related, Dataset Related
    if torch.cuda.is_available():
        print("Using GPU" + "-" * 80)
        args.device = torch.device("cuda:2")
    else:
        print("Using CPU" + "-" * 80)
        args.device = torch.device("cpu")
    # args.device = torch.device("cpu")


    set_params(args)


    target_lang = param.lang
    src_langs = ['fr', 'ja', 'es', 'el', 'en']
    src_langs.remove(target_lang)

    target_lang = param.lang

    # load data
    data_dir = './data/kg'  # where you put kg data
    seed_dir = './data/seed_alignlinks'  # where you put seed align links data
    model_dir = join('./trained_model_no_generation', f'kens-{param.knowledge}-{param.dim}', target_lang)  # output
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # logging
    set_logger(param, model_dir)  # set logger
    logging.info('Knowledge model: %s'%(param.knowledge))
    logging.info('target language: %s'%(param.lang))

    # hyper-parameters
    logging.info(f'dim: {param.dim}')
    logging.info(f'lr: {param.lr}')


    # target (sparse) kg
    kg0 = load_target_kg(data_dir, target_lang, testfile_suffix='-val.tsv',device = args.device)  # target kg. KnowledgeGraph object
    # supporter kgs
    supporter_kgs = load_support_kgs(data_dir, seed_dir, target_lang, src_langs,device = args.device)  # list[KnowledgeGraph]


    # Optimizer for all KGs
    optimizer = optim.Adam(kg0.parameters(), lr=param.lr)
    for each_supporter_kg in supporter_kgs:
        optimizer.add_param_group({'params': each_supporter_kg.parameters()})


    # supporter KG use all links to train
    for kg1 in supporter_kgs:
        kg1.h_train = torch.cat([kg1.h_train, kg1.h_test], axis=0)
        kg1.r_train = torch.cat([kg1.r_train, kg1.r_test], axis=0)
        kg1.t_train = torch.cat([kg1.t_train, kg1.t_test], axis=0)
        # kg1.y_train = np.zeros(kg1.h_train.shape[0])

    # seed alignment links
    seed_alignlinks = load_all_to_all_seed_align_links(seed_dir,device = args.device)  # {(lang1, lang2): 2-col np.array} dict of ints

    all_kgs = [kg0] + supporter_kgs

    #build alignment model (all-to-all)
    for kg in all_kgs:
        kg.build_alignment_models(all_kgs)  # kg.align_models_of_all {lang: align_model}

    # create validator
    validator = MultiModelTester(kg0, supporter_kgs,args.device)

    print('model initialization done')


    def get_entity_embedding(kg):
        entity_embedding = kg.model.entity_embedding_layer.cpu().data.numpy()
        print(kg.lang)
        print(np.min(entity_embedding))
        print(np.max(entity_embedding))


    ############ Start Training !!!
    for i in range(param.round):
        logging.info(f'Epoch: {i}')

        # for kg in all_kgs:
        #     kg.train()
        #     get_entity_embedding(kg) #initial embedding are correct



        # train alignment model
        # Adjust optimizer learning rate
        for param_group in optimizer.param_groups:  # 在每次更新参数前迭代更改学习率
            param_group["lr"] = param.align_lr
        for kg in all_kgs:
            # align it with everything else
            for other_lang, align_model in kg.align_models_of_all.items():
                if (other_lang, kg.lang) in seed_alignlinks:  # seed_alignlinks {(lang1, lang2): 2-col np.array}
                    # use if to avoid retrain the same pair of languages
                    align_links = seed_alignlinks[(other_lang, kg.lang)]
                    train_align_batch(align_model,align_links,optimizer)


        # # self-learning
        # for kg in all_kgs:
        #     for other_kg in all_kgs:
        #         if other_kg.lang != kg.lang and (other_kg.lang, kg.lang) in seed_alignlinks:
        #             print(f'self learning[{kg.lang}][{other_kg.lang}]')
        #             seeds = seed_alignlinks[(other_kg.lang, kg.lang)]
        #             print("Original link number is %d" % (len(seeds)))
        #             found = extend_seed_align_links(other_kg, kg, seeds,args.device)
        #             if len(found) > 0:  # not []
        #                 new_seeds = torch.cat([seeds, found], axis=0)
        #                 seed_alignlinks[(other_kg.lang, kg.lang)] = new_seeds
        #                 print("Generated link number is %d" % (len(found)))


        # # train knowledge model
        # Adjust learning rate for kg module
        for param_group in optimizer.param_groups:  # 在每次更新参数前迭代更改学习率
            param_group["lr"] = param.lr

        train_kg_batch(kg0, optimizer, param.epoch10, args.device)


        # get_entity_embedding(kg0)
        # print(param.epoch10)
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        for kg1 in supporter_kgs:
            train_kg_batch(kg1, optimizer, param.epoch11, args.device)


        # for kg in all_kgs:
        #     kg.train()
        #     get_entity_embedding(kg)  # initial embedding are correct


        if i % param.val_freq == 0:  # validation
            logging.info(f'=== round {i}')
            logging.info(f'[{kg0.lang}]')

            for kg in all_kgs:
                kg.eval()

            _ = validator.test(TestMode.KG0) # validation set

            # _ = validator.test_all_self_hit_k(is_lifted=False)
            _ = validator.test_all_hit_k(is_lifted=True)





    kg0.save_model(model_dir)
    for kg1 in supporter_kgs:
        kg1.save_model(model_dir)
        # get_entity_embedding(kg1)


    # For ensemble usage
    for kg1 in supporter_kgs:
        print("%s: Original links are %d" % (kg1.lang,len(kg1.dict0to1)))
        kg1.populate_alignlinks(kg0, seed_alignlinks)
        print("%s: Updated links are %d" % (kg1.lang, len(kg1.dict0to1)))

    choices = ['-val.tsv', '-test.tsv']  # '-val.tsv' if predict on validation data, '-test.tsv' if on test data
    validator = MultiModelTester(kg0, supporter_kgs,args.device)  # TODO: REWRITE INPUT GPU

    kg0.filtered_reordered_embedding_matrix = None
    for kg1 in supporter_kgs:
        kg1.filtered_reordered_embedding_matrix = None


    #Testing
    for kg in all_kgs:
        kg.eval()

    for suffix in choices:
        testfile = join(data_dir, target_lang + suffix)
        output = join(model_dir, 'results'+suffix)
        testcases = pd.read_csv(testfile, sep='\t', header=None).values
        param.n_test = testcases.shape[0]  # test on all training triples
        print('Loaded test cases from: %s'%testfile)

        results_df = validator.test_and_record_results(testcases)

        results_df.to_csv(join(output), sep='\t', index=False)

if __name__ == "__main__":
    main(parse_args())
    # main(parse_args(['--knowledge_model','transe','--target_language','ja','--use_default']))