import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor




class TGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_graphs, dropout=0.1, n_layers=2, bias=False, featureless=True, act='relu'):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.embedding_list = nn.ModuleList([nn.Embedding(in_dim, hidden_dim) for _ in range(num_graphs)])
        for embed in self.embedding_list:
            nn.init.xavier_uniform_(embed.weight)
        self.layers = nn.ModuleList([GraphConv_fix(in_dim=hidden_dim, out_dim=hidden_dim, num_graphs=num_graphs, dropout=dropout, featureless=True, bias=False, act=act)])
        for _ in range(n_layers-2):
            self.layers.append(GraphConv_fix(in_dim=hidden_dim, out_dim=hidden_dim, num_graphs=num_graphs, dropout=dropout, featureless=False, bias=False, act=act))
        self.layers.append(GraphConv_fix(in_dim=hidden_dim, out_dim=out_dim, num_graphs=num_graphs, dropout=dropout, featureless=False, bias=False, act=act))
        self.num_graphs = num_graphs

    def word_dropout(self, inputs, keepprob):
        features = [embed(inputs) for embed in self.embedding_list]
        mask = torch.rand((features[0].size(0),1),device=features[0].device) > keepprob
        features = [f.masked_fill(mask, 0)* (1.0/keepprob) for f in features]
        return features

    def forward(self, inputs, edge_indexs, edge_weights, keepprob):
        features = self.word_dropout(inputs, keepprob)
        for layer in self.layers:
            features = layer(features, edge_indexs, edge_weights, keepprob)
        features = torch.stack(features, dim=0)
        features = torch.mean(features, dim=0)
        return features

class GraphConv_fix(nn.Module):
    def __init__(self, in_dim, out_dim, num_graphs, dropout=0.1, featureless=False, bias=False, act='relu'):
        super().__init__()
        # net_dict = {'gcn':GCNConv, 'sage':SAGEConv, 'gat':GATConv}
        # model_func = net_dict[kernel]
        self.intra_convs = nn.ModuleList([GCNConv(in_dim, out_dim,add_self_loops=False,normalize=False) for _ in range(num_graphs)])


        self.inter_convs = nn.ParameterList([nn.Parameter(torch.zeros((out_dim, out_dim), dtype=torch.float), requires_grad=True) for _ in range(num_graphs)])
        for tmp in self.inter_convs:
            nn.init.xavier_uniform_(tmp)
        if act == 'relu':
            self.act = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.act = nn.Tanh()
        self.bias = bias
        # if self.bias:
        #     self.bias = nn.Parameter(torch.zeros(out_dim), requires_grad=True)
        #     nn.init.xavier_normal_(self.bias)
        self.dropout = nn.Dropout(dropout)
        self.featureless = featureless
        
    def atten(self, supports):
        tmp_supports = []
        for i in range(len(supports)):
            supports[i] = torch.matmul(supports[i], self.inter_convs[i])
            tmp_supports.append(supports[i])
        tmp_supports = torch.stack(tmp_supports, dim=0)
        tmp_supports_sum = torch.sum(tmp_supports, dim=0)
        att_features = []
        for support in supports:
            att_features.append(self.act(tmp_supports_sum-support))
        
        return att_features

    def forward(self, inputs, edge_indexs, edge_weights, keepprob):
        num_nodes = inputs[0].size(0)
        if not self.featureless:
            for i in range(len(inputs)):
                inputs[i] = self.dropout(inputs[i])
        supports = []
        for i, conv in enumerate(self.intra_convs):
            # support = conv(inputs[i], edge_indexs[i], edge_weights[i])
            adj = SparseTensor(row=edge_indexs[i][0], col=edge_indexs[i][1], value=edge_weights[i],
                   sparse_sizes=(num_nodes, num_nodes))
            # support = conv(inputs[i], edge_indexs[i], edge_weights[i])
            support = conv(inputs[i], adj.t())
            support = self.act(support)
            supports.append(support)
        supports = self.atten(supports)
        self.embedding = torch.stack(supports, dim=0)
        self.embedding = torch.mean(self.embedding, dim=0)

        return supports