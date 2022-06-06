# Multilingual Knowledge Graph Completion with Zero Seed Alignment

## UCLA CS 249 S22 Course Project (Group 11)

| Name       | UID |
|---------------|----------|
| Rakesh Bal |   605526216  |
| Rustem Can Aygun | 105349867     |
| Ashwath Radhachandran | 805725917 |
| Rahul Kapur   | 405530587 |

This repository is a PyTorch and Python implementation of the course project for UCLA CS 249 (Special Topics - Graph Neural Networks) S22 Course. The slides and report are in the files, `Multilingual Knowledge Graph Completion with Zero Seed Alignment.pdf` and `CS249-Group11-Report.pdf` respectively.

All of our methods are mentioned in the report and all of our experiments are based on the DBP-5L data set (also present in the repo)The code is a direct modification of the official repository for the paper ["Multilingual Knowledge Graph Completion with Self-Supervised Adaptive Graph Alignment"](https://arxiv.org/abs/2203.14987), hosted in GitHub [here](https://github.com/amzn/ss-aga-kgc)

We thank the authors' effort for making the code public. To run the repository please follow the README in the original repo link above or follow the instructions below:

## Data

**DBP-5L**:  A Public dataset from  https://github.com/stasl0217/KEnS.

**E-PKG**: A new industrial multilingual E-commerce product KG dataset. (To be released.)

Data format: Each dataset contains the following files and folders:

- entity: Folder that contains the entity list for each KG.
- kg: Folder that contains the KG triple list (head_entity_index, relation_index, tail_entity_index) for each kg.
- seed_alignlinks: Folder that contains seed entity alignment pair list between two KGs. 
- relation.txt: File that contains relation that is shared across all KGs.
- entity_embeddings.npy: The numpy file of mbert embedding for each entity from all KGs. Size of [Num_entity_all, 768]. We use the **BERT-Base, Multilingual Cased**  from https://github.com/google-research/bert/blob/master/multilingual.md to generate it. You can download the entity_embeddings.npy for ther DBP-5L dataset from [here](https://drive.google.com/file/d/1-R_2lqS5AQtWqLZXC45SrfkK5XETREe5/view?usp=sharing).

To run the code, create the folders "dataset/dbp5l",  "dataset/epkg" and download the two datasets respectively.

## Setup

To run the code, you need the following dependencies:

- [Python 3.6.10](https://www.python.org/)

- [Pytorch 1.10.0](https://pytorch.org/)
- [pytorch_geometric 2.0.4](https://pytorch-geometric.readthedocs.io/)
  - torch-cluster==1.6.0
  - torch-scatter==2.0.9
  - torch-sparse==0.6.13
- [numpy 1.16.1](https://numpy.org/)

## Usage

Execute the following scripts to train the model on the targeted japanese KG:

```bash
python run_model.py --target_language ja --use_default
```

There are some key options of this scrips:

- `--target_language`: The targeted KG to conduct the KG completion task.
- `--num_hop`: Number of hops for sampling neighbors for each node.
- `--preserved_ratio` : How many align links to preserve in learning alignment embeddings. The rest are served as masked alignments and we ask the model to recover them.
- `--generation_freq`: How many epochs to conduct new pair generation once.
- `--use_default`:  Use the preset hyper-parameter combinations.

The details of other optional hyperparameters can be found in run_model.py.

For various apporaches implemented by us in the Course Project:
1. Apporach 1: Use the generated seed alignments in `datasetdbp5l/seed_alignlinks_bert_generated` with the original SS-AGA code [here](https://github.com/amzn/ss-aga-kgc).
2. Approach 2: Use param approach 2 in the command line execution
3. Approach 3: Default paramters. [Note: Requires high GPU Resources to run]

## References
[1] [Huang, Z., Li, Z., Jiang, H., Cao, T., Lu, H., Yin, B., ... & Wang, W. (2022). Multilingual Knowledge Graph Completion with Self-Supervised Adaptive Graph Alignment.](https://arxiv.org/abs/2203.14987)