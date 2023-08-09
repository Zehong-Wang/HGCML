import os.path as osp
from typing import Callable, List, Optional

import numpy as np
from scipy import io as sio
import torch
from torch_geometric.data import (HeteroData, InMemoryDataset)
from torch_geometric.datasets import DBLP, IMDB

from torch_geometric import transforms as T


class ACM(InMemoryDataset):

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['ACM.mat']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        data = HeteroData()

        raw_data = sio.loadmat(osp.join(self.raw_dir, 'ACM.mat'))
        p_vs_l = raw_data['PvsL']
        p_vs_a = raw_data['PvsA']
        p_vs_t = raw_data['PvsT']
        p_vs_p = raw_data['PvsP']
        p_vs_c = raw_data['PvsC']

        conf_ids = [0, 1, 9, 10, 13]
        label_ids = [0, 1, 2, 2, 1]

        p_vs_c_filter = p_vs_c[:, conf_ids]
        p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
        p_vs_c = p_vs_c[p_selected]
        p_vs_p = p_vs_p[p_selected].T[p_selected]
        a_selected = (p_vs_a[p_selected].sum(0) != 0).A1.nonzero()[0]
        p_vs_a = p_vs_a[p_selected].T[a_selected].T
        l_selected = (p_vs_l[p_selected].sum(0) != 0).A1.nonzero()[0]
        p_vs_l = p_vs_l[p_selected].T[l_selected].T
        t_selected = (p_vs_t[p_selected].sum(0) != 0).A1.nonzero()[0]
        p_vs_t = p_vs_t[p_selected].T[t_selected].T

        pc_p, pc_c = p_vs_c.nonzero()
        labels = np.zeros(len(p_selected), dtype=np.int64)
        for conf_id, label_id in zip(conf_ids, label_ids):
            labels[pc_p[pc_c == conf_id]] = label_id
        labels = torch.LongTensor(labels)

        data['paper'].x = torch.FloatTensor(p_vs_t.toarray())
        data['paper'].y = torch.LongTensor(labels)

        data['paper', 'author'].edge_index = torch.tensor(p_vs_a.nonzero(), dtype=torch.long)
        data['author', 'paper'].edge_index = torch.tensor(p_vs_a.transpose().nonzero(), dtype=torch.long)
        data['paper', 'subject'].edge_index = torch.tensor(p_vs_l.nonzero(), dtype=torch.long)
        data['subject', 'paper'].edge_index = torch.tensor(p_vs_l.transpose().nonzero(), dtype=torch.long)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class AMiner(InMemoryDataset):

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['labels.npy', 'pr.txt', 'pa.txt',
                'feature_0.npy', 'feature_1.npy', 'feature_2.npy']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        data = HeteroData()

        node_types = ['paper', 'author', 'reference']
        for i, node_type in enumerate(node_types):
            x = np.load(osp.join(self.raw_dir, f'features_{i}.npy'))
            data[node_type].x = torch.from_numpy(x).to(torch.float)
        labels = np.load(osp.join(self.raw_dir, 'labels.npy')).astype('int32')
        data['paper'].y = torch.from_numpy(labels)

        pa = np.loadtxt(osp.join(self.raw_dir, 'pa.txt'))
        pa = torch.from_numpy(pa).t()
        pr = np.loadtxt(osp.join(self.raw_dir, 'pr.txt'))
        pr = torch.from_numpy(pr).t()

        data['paper', 'reference'].edge_index = pr[[0, 1]].long()
        data['reference', 'paper'].edge_index = pr[[1, 0]].long()
        data['paper', 'author'].edge_index = pa[[0, 1]].long()
        data['author', 'paper'].edge_index = pa[[1, 0]].long()

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class FreeBase(InMemoryDataset):

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['labels.npy', 'ma.txt', 'md.txt', 'mw.txt', 'feature_0.npy',
                'feature_1.npy', 'feature_2.npy', 'feature_3.npy']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        data = HeteroData()

        node_types = ['movie', 'actor', 'director', 'writer']
        for i, node_type in enumerate(node_types):
            x = np.load(osp.join(self.raw_dir, f'features_{i}.npy'))
            data[node_type].x = torch.from_numpy(x).to(torch.float)
        labels = np.load(osp.join(self.raw_dir, 'labels.npy')).astype('int32')
        data['movie'].y = torch.from_numpy(labels)

        ma = np.loadtxt(osp.join(self.raw_dir, 'ma.txt'))
        ma = torch.from_numpy(ma).t()
        md = np.loadtxt(osp.join(self.raw_dir, 'md.txt'))
        md = torch.from_numpy(md).t()
        mw = np.loadtxt(osp.join(self.raw_dir, 'mw.txt'))
        mw = torch.from_numpy(mw).t()

        data['movie', 'actor'].edge_index = ma[[0, 1]].long()
        data['actor', 'movie'].edge_index = ma[[1, 0]].long()
        data['movie', 'director'].edge_index = md[[0, 1]].long()
        data['director', 'movie'].edge_index = md[[1, 0]].long()
        data['movie', 'writer'].edge_index = mw[[0, 1]].long()
        data['writer', 'movie'].edge_index = mw[[1, 0]].long()

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


def get_dataset(dataset_name):
    path = osp.join('data', dataset_name)

    if dataset_name == 'dblp':
        dataset = DBLP(path)
        metapaths = [
            [('author', 'paper'), ('paper', 'author')],
            [('author', 'paper'), ('paper', 'term'), ('term', 'paper'), ('paper', 'author')],
            [('author', 'paper'), ('paper', 'conference'), ('conference', 'paper'), ('paper', 'author')]
        ]
        target = 'author'
    elif dataset_name == 'imdb':
        dataset = IMDB(path)
        metapaths = [
            [('movie', 'director'), ('director', 'movie')],
            [('movie', 'actor'), ('actor', 'movie')]
        ]
        target = 'movie'
    elif dataset_name == 'acm':
        dataset = ACM(path)
        metapaths = [
            [('paper', 'author'), ('author', 'paper')],
            [('paper', 'subject'), ('subject', 'paper')]
        ]
        target = 'paper'
    elif dataset_name == 'aminer':
        dataset = AMiner(path)
        metapaths = [
            [('paper', 'reference'), ('reference', 'paper')],
            [('paper', 'author'), ('author', 'paper')]
        ]
        target = 'paper'
    elif dataset_name == 'freebase':
        dataset = FreeBase(path)
        metapaths = [
            [('movie', 'actor'), ('actor', 'movie')],
            [('movie', 'director'), ('director', 'movie')],
            [('movie', 'writer'), ('writer', 'movie')]
        ]
        target = 'movie'
    else:
        raise TypeError('Unsupported dataset!')

    return dataset, metapaths, target
