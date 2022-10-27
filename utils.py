import numpy as np
import pickle as pkl
import scipy.sparse as sp
import re
import torch
import json

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_corpus_torch(dataset, device):
    """
    Loads input corpus from gcn/data directory, torch tensor version

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    adjs = []
    for one in ['seq','sem','syn']:
        adjs.append(pkl.load(open('./data/{}.{}_adj'.format(dataset,one),'rb')))
    adj, adj1, adj2 = adjs[0], adjs[1], adjs[2]
    
    data = json.load(open('./data/{}_data.json'.format(dataset),'r'))
    train_ids, test_ids, corpus, labels, vocab, word_id_map, id_word_map, label_list = data

    num_labels = len(label_list)
    train_size = len(train_ids)

    val_size = int(0.1*len(train_ids))
    test_size = len(test_ids)

    labels = np.asarray(labels[:train_size]+[0]*len(vocab)+labels[train_size:])
    print(len(labels))


    idx_train = range(train_size-val_size)
    idx_val = range(train_size-val_size, train_size)
    idx_test = range(train_size+len(vocab), train_size+len(vocab)+test_size)
    
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_test[test_mask] = labels[test_mask]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj1 = adj1 + adj1.T.multiply(adj1.T > adj1) - adj1.multiply(adj1.T > adj1)
    adj2 = adj2 + adj2.T.multiply(adj2.T > adj2) - adj2.multiply(adj2.T > adj2)

    # tensor
    # adj = torch.sparse_csr_tensor(adj.indptr, adj.indices, adj.data, dtype=torch.float).to_sparse_coo().to(device)
    # adj1 = torch.sparse_csr_tensor(adj1.indptr, adj1.indices, adj1.data, dtype=torch.float).to_sparse_coo().to(device)
    # adj2 = torch.sparse_csr_tensor(adj2.indptr, adj2.indices, adj2.data, dtype=torch.float).to_sparse_coo().to(device)
    # features = torch.sparse_csr_tensor(features.indptr, features.indices, features.data, dtype=torch.float).to_sparse_coo().to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    train_mask = torch.tensor(train_mask, dtype=torch.float).to(device)
    val_mask = torch.tensor(val_mask, dtype=torch.float).to(device)
    test_mask = torch.tensor(test_mask, dtype=torch.float).to(device)

    return adj, adj1, adj2, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, val_size,test_size, num_labels
    
def get_edge_tensor_list(adj_list, device):
    indice_list, data_list = [], []
    for adj in adj_list:
        row = torch.tensor(adj.row, dtype=torch.long).to(device)
        col = torch.tensor(adj.col, dtype=torch.long).to(device)
        data = torch.tensor(adj.data, dtype=torch.float).to(device)
        indice = torch.stack((row,col),dim=0)
        indice_list.append(indice)
        data_list.append(data)
    return indice_list, data_list

def get_edge_tensor(adj):
    row = torch.tensor(adj.row, dtype=torch.long)
    col = torch.tensor(adj.col, dtype=torch.long)
    data = torch.tensor(adj.data, dtype=torch.float)
    indice = torch.stack((row,col),dim=0)
    return indice, data

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

def preprocess_features_origin(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def preprocess_adj_mix(adj):
    adj_normalized = adj + sp.eye(adj.shape[0])
    return sparse_to_tuple(adj)

def preprocess_adj_tensor(adj, device):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return torch.sparse_coo_tensor(np.stack([adj_normalized.row, adj_normalized.col], axis=0), adj_normalized.data, adj_normalized.shape, dtype=torch.float).to(device)

def preprocess_adj_mix_tensor(adj, device):
    adj_normalized = adj + sp.eye(adj.shape[0])
    # return torch.sparse_csr_tensor(crow_indices=adj.indptr, col_indices=adj.indices, values=adj.data, dtype=torch.float).to_sparse_coo().to(device)
    return torch.tensor(adj.todense(), dtype=torch.float).to(device)