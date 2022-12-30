import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import dropout_adj

EPS = 1e-15

def dropout_feat(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, activation = torch.tanh, 
                 base_model=GCNConv, num_layers: int = 1):
        super(Encoder, self).__init__()
        self.base_model = base_model
        self.activation = activation
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(base_model(in_dim, hid_dim))
        for _ in range(num_layers - 1):
            self.convs.append(base_model(hid_dim, hid_dim))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for conv in self.convs:
            x = self.activation(conv(x, edge_index))
        return x


class HGCML(nn.Module):
    def __init__(self, encoder, hid_dim, num_relations, tau: float = 0.2, 
                 pe: float = 0.2, pf: float = 0.2, alpha: float = 0.5):
        super(HGCML, self).__init__()
        self.encoder = encoder
        self.hid_dim = hid_dim
        self.pe = pe
        self.pf = pf
        self.num_relations = num_relations
        self.tau = tau
        self.alpha = alpha

        self.local_projector = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.PReLU(), nn.Linear(hid_dim, hid_dim))
        self.global_projector = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.PReLU(), nn.Linear(hid_dim, hid_dim))
        self.weight = nn.Parameter(torch.Tensor(hid_dim, hid_dim), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight, gain=1.414)
        self.encoder.reset_parameters()

        for model in self.local_projector:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
        for model in self.global_projector:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def forward(self, x, edge_indices, combine):
        zs = [self.encoder(x, edge_index) for edge_index in edge_indices]

        if combine == 'concat':
            embeddings = torch.concat(zs, dim=-1)
        elif combine == 'mean':
            embeddings = torch.stack(zs).mean(dim=0)
        else:
            raise TypeError('Unsupported fuse function!')

        return embeddings

    def loss(self, x, edge_indices, mask):
        loss = 0.
        num_contrasts = 0

        for i in range(self.num_relations):
            for j in range(i, self.num_relations):
                loss += self.contrast(x, edge_indices[i], edge_indices[j], mask)
                num_contrasts += 1

        return loss / num_contrasts

    def contrast(self, x, edge_index_1, edge_index_2, mask):
        edge_index_1 = dropout_adj(edge_index_1, p=self.pe)[0]
        edge_index_2 = dropout_adj(edge_index_2, p=self.pe)[0]
        x_1 = dropout_feat(x, self.pf)
        x_2 = dropout_feat(x, self.pf)

        z1 = self.encoder(x_1, edge_index_1)
        z2 = self.encoder(x_2, edge_index_2)
        
        local_loss = (self.local_loss(z1, z2, mask) + self.local_loss(z2, z1, mask.t())) / 2
        global_loss = (self.global_loss(z1, z2) + self.global_loss(z2, z1)) / 2
        loss = self.alpha * local_loss + (1 - self.alpha) * global_loss

        return loss

    def _sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def infonce(self, z1, z2, mask):
        f = lambda x: torch.exp(x / self.tau)
        sim_intra = f(self._sim(z1, z1))
        sim_inter = f(self._sim(z1, z2))

        loss = -torch.log(
            (sim_inter * mask).sum(1) /
            (sim_intra.sum(1) + sim_inter.sum(1) - (sim_intra * mask).sum(1))
        )
        return loss.mean()

    def local_loss(self, z1: torch.Tensor, z2: torch.Tensor, mask):
        h1 = self.local_projector(z1)
        h2 = self.local_projector(z2)

        loss = self.infonce(h1, h2, mask)

        return loss

    def readout(self, z):
        return z.mean(dim=0)
        
    def discriminate(self, z, summary, sigmoid=True):
        summary = torch.matmul(self.weight, summary)
        value = torch.matmul(z, summary)
        return torch.sigmoid(value) if sigmoid == True else value

    def global_loss(self, pos_z: torch.Tensor, neg_z: torch.Tensor):
        s = self.readout(pos_z)
        h = self.global_projector(s)

        pos_loss = -torch.log(self.discriminate(pos_z, h, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, h, sigmoid=True) + EPS).mean()
        loss = (pos_loss + neg_loss) * 0.5

        return loss        