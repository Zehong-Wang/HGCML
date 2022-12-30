import argparse
import os.path as osp
from pathlib import Path
import shutil

import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split

import torch
import torch_geometric.transforms as T

from model import HGCML, Encoder
from utils import set_random_seed, get_masks, add_self_loop
from datasets import get_dataset

def get_arguments():
    parser = argparse.ArgumentParser(description='Training model')
    parser.add_argument('--prefix', type=str, default='HGCML')
    parser.add_argument('--dataset', type=str, default='acm')

    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--clf_runs', type=int, default=10)

    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--hid_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tau', type=float, default=0.2)
    parser.add_argument('--num_semantic_pos', type=int, default=0)
    parser.add_argument('--num_topology_pos', type=int, default=0)
    parser.add_argument('--edge_drop_rate', type=float, default=0.2)
    parser.add_argument('--feature_drop_rate', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.)

    parser.add_argument('--train_splits', type=float, nargs='+', default=[0.2])
    parser.add_argument('--combine', type=str, default='concat')

    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    return vars(args)


def train(model, x, edge_indices, mask, optimizer):
    model.train()
    optimizer.zero_grad()

    loss = model.loss(x, edge_indices, mask)

    loss.backward()
    optimizer.step()

    return loss.item()


def test(embeddings, labels, train_split=0.2, runs=10):
    macro_f1_list = list()
    micro_f1_list = list()
    nmi_list = list()
    ari_list = list()

    for i in range(runs):
        x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, train_size=train_split, random_state=i)

        clf = SVC(probability=True)

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        macro_f1 = f1_score(y_test, y_pred, average='macro')
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1_list.append(macro_f1)
        micro_f1_list.append(micro_f1)

    for i in range(runs):
        kmeans = KMeans(n_clusters=len(torch.unique(labels)), algorithm='full')
        y_kmeans = kmeans.fit_predict(embeddings)

        nmi = normalized_mutual_info_score(labels, y_kmeans)
        ari = adjusted_rand_score(labels, y_kmeans)
        nmi_list.append(nmi)
        ari_list.append(ari)

    macro_f1 = np.array(macro_f1_list).mean()
    micro_f1 = np.array(micro_f1_list).mean()
    nmi = np.array(nmi_list).mean()
    ari = np.array(ari_list).mean()

    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1, 
        'nmi': nmi,
        'ari': ari
    }



def main():
    params = get_arguments()
    set_random_seed(params['seed'])

    device = torch.device('cuda:{}'.format(params['gpu']) if torch.cuda.is_available() else 'cpu')

    checkpoints_path = f'checkpoints'
    try:
        shutil.rmtree(checkpoints_path)
    except:
        pass
    Path(checkpoints_path).mkdir(parents=True, exist_ok=False)

    dataset, metapaths, target = get_dataset(params['dataset'])
    data = dataset[0]
    num_relations = len(metapaths)
    num_nodes = data[target].y.shape[0]
    num_feat = data[target].x.shape[1]

    metapath_data = T.AddMetaPaths(metapaths=metapaths, drop_orig_edges=True)(data)
    metapath_data = add_self_loop(metapath_data, num_relations, num_nodes)

    x = metapath_data[target].x.to(device)
    edge_indices = [edge_index.to(device) for edge_index in metapath_data.edge_index_dict.values()]
    labels = metapath_data[target].y

    sem_mask, top_mask = get_masks(dataset_name=params['dataset'], num_sem_pos=params['num_semantic_pos'],
                                   num_top_pos=params['num_topology_pos'])
    mask = torch.logical_or(sem_mask, top_mask).to(device)

    encoder = Encoder(in_dim=num_feat, hid_dim=params['hid_dim'], num_layers=params['num_layers'])                 
    model = HGCML(encoder=encoder, hid_dim=params['hid_dim'], num_relations=num_relations,
                 tau=params['tau'], pe=params['edge_drop_rate'], pf=params['feature_drop_rate'],
                 alpha=params['alpha']).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    model = model

    best_epoch = 0
    best_mif1 = 0
    patience_cnt = 0

    for i in range(1, params['epochs']):
        loss = train(model, x, edge_indices, mask, optimizer)

        if i % params['eval_interval'] == 0:
            embeddings = model(x, edge_indices, params['combine']).detach().cpu().numpy()
            results = test(embeddings, labels, train_split=0.2, runs=params['clf_runs'])
            print('Macro-F1: {:.4f} | Micro-F1: {:.4f} | NMI: {:.4f} | ARI: {:.4f}'
              .format(results['macro_f1'], results['micro_f1'], results['nmi'], results['ari']))

            if results['micro_f1'] > best_mif1:
                best_mif1 = results['micro_f1']
                best_epoch = i
                patience_cnt = 0
                torch.save(model.state_dict(), osp.join(checkpoints_path, f'{i}.pkl'))
            else:
                patience_cnt += 1

            if patience_cnt == params['patience']:
                break

    model.load_state_dict(torch.load(osp.join(checkpoints_path, f'{best_epoch}.pkl')))
    shutil.rmtree(checkpoints_path)

    embeddings = model(x, edge_indices, params['combine']).detach().cpu().numpy()
    labels = metapath_data[target].y
    for train_split in params['train_splits']:
        results = test(embeddings, labels, train_split=train_split, runs=params['clf_runs'])

        print('Train Split: {} | Macro-F1: {:.4f} | Micro-F1: {:.4f} | NMI: {:.4f} | ARI: {:.4f}'
              .format(train_split, results['macro_f1'], results['micro_f1'], results['nmi'], results['ari']))


if __name__ == '__main__':
    main()