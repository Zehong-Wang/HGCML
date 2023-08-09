# Heterogeneous Graph Contrastive Multi-view Learning

This repo is for source code of SDM 2023 paper "**Heterogeneous Graph Contrastive Multi-view Learning**". [paper](https://epubs.siam.org/doi/abs/10.1137/1.9781611977653.ch16)

## Environment Settings

* python==3.8.0
* scipy==1.6.2
* torch==1.11.0
* scikit-learn=1.0.2
* torch_geometric==2.0.4

## Dataset

We utilize five benchmark datasets in the paper to perform node classification and node clustering tasks. The DBLP and IMDB datasets are built in PyG. We provide ACM, AMiner, and FreeBase in [GoogleDrive](https://drive.google.com/drive/folders/1kJfrSP-bMF3MZ8GJx_pHdmWOnq5Rv2R3?usp=sharing). 

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
python main.py --dataset acm
```

## Citation

```
@inproceedings{wang2023heterogeneous,
  title={Heterogeneous graph contrastive multi-view learning},
  author={Wang, Zehong and Li, Qi and Yu, Donghua and Han, Xiaolong and Gao, Xiao-Zhi and Shen, Shigen},
  booktitle={Proceedings of the 2023 SIAM International Conference on Data Mining (SDM)},
  pages={136--144},
  year={2023},
  organization={SIAM}
}
```

## Contact

If you have any questions, don't hesitate to contact me (zwang43@nd.edu, zehongwang0414@gmail.com)!
