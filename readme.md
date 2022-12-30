# Heterogeneous Graph Contrastive Multi-view Learning

This repo is for source code of SDM 2023 paper "Heterogeneous Graph Contrastive Multi-view Learning". [paper](https://arxiv.org/abs/2210.00248)

## Environment Settings

* python==3.8.0
* scipy==1.6.2
* torch==1.11.0
* scikit-learn=1.0.2
* torch_geometric==2.0.4

## Dataset

We utilize five benchmark datasets in the paper to perform node classification and node clustering tasks. We provide ACM, AMiner in [GoogleDrive](https://drive.google.com/drive/folders/1kJfrSP-bMF3MZ8GJx_pHdmWOnq5Rv2R3?usp=sharing). 

* ACM
* DBLP
* IMDB
* AMiner
* FreeBase

You can create the "data" folder in the root directory, then put the datasets in. Like "/HGCML/data/acm/...". 
 
## Positive Sampling

Positive sampling is an critical module in HGCML. We provide the processed positive samples in [GoogleDrive](https://drive.google.com/drive/folders/1kJfrSP-bMF3MZ8GJx_pHdmWOnq5Rv2R3?usp=sharing).

Please put the "pos" folder in the root directory, like "/HGCML/pos/...". 

## How to run

For example, if you want to run HGCML-P on ACM dataset, execute

```
python main.py --dataset acm --lr 1e-3 --tau 0.5 --num_semantic_pos 16 --num_topology_pos 16
```