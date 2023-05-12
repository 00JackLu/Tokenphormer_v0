import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import pickle
# from torch_sparse import spspmm
import os
import re
import copy
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from dgl import DGLGraph, RandomWalkPE
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import dgl
import itertools
import dgl.function as fn


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_batch(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(
        torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix


# pos_enc_dim: position embedding size
def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian

    # adjacency_matrix(transpose, scipy_fmt="csr")
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(
        g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(
        L, k=pos_enc_dim+1, which='SR', tol=1e-2)  # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float()

    return lap_pos_enc


# add RWPE from dgl
def randomwalk_positional_encoding(adj, rw_dim):
    edge_index = adj._indices()
    G = dgl.graph((edge_index[0], edge_index[1]))
    transform = RandomWalkPE(k=rw_dim)
    return transform(G)


def random_walk_gen(adj, t_num, w_len, dataset):
    edge_index = adj._indices()
    G = dgl.graph((edge_index[0], edge_index[1]))
    nodes_features = []
    # 遍历张量
    for i in range(G.nodes().size(dim=0)):
        strat_nodes = [i] * t_num

        token_path = dgl.sampling.random_walk(G, strat_nodes, length=w_len, restart_prob=0.2)
        nodes_features.append(token_path[0])
    torch.save(nodes_features, dataset + '_t_num=' + str(t_num) + '_w_len=' + str(w_len) + '.pt')
    
    nodes_features2 = []
    for i in range(G.nodes().size(dim=0)):
        strat_nodes = [i] * t_num

        token_path = dgl.sampling.random_walk(G, strat_nodes, length=w_len, restart_prob=0.2)
        nodes_features2.append(token_path[0])
    torch.save(nodes_features2, dataset + '_t_num=' + str(t_num) + '_w_len=' + str(w_len) + '_back.pt')


# 上下文 sum + cat
# def get_token(features, W, num_steps, dataset, device):
#     # print(features.shape[1])
#     # nodes_features = torch.empty(features.shape[0], 2*W+1, features.shape[1]*(num_steps+1))
#     nodes_features = torch.empty(features.shape[0], W+1, features.shape[1]*3)
    
#     print('loading the pt file now')
#     pt = torch.load(dataset + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '.pt', map_location=device)
#     pt_back = torch.load(dataset + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '_back.pt', map_location=device)
#     # print(f"Random Walk Begin! Random Walk nums:{W}, length:{num_steps}")
#     for node in range(features.shape[0]):
#         print(f'load the {node} node')
#         walk = pt[node].tolist()
#         walk_back = pt_back[node].tolist()
#         feature_raw = features[node]
#         i = 0
#         # 遍历walk
#         for j in range(W):
#             sub_list = walk[j]
#             sub_list2 = walk_back[j]
#             feature = torch.zeros(features.shape[1])
#             feature2 = torch.zeros(features.shape[1])
#             feature_raw = features[node]
#             # node2 = sub_list[0]
#             isFirstNode = True
#             for node2 in sub_list:
#                 if isFirstNode == True:
#                     isFirstNode = False
#                     continue
#                 if node2 == -1:  continue
#                 else:   nf = features[node2]
                
#                 feature = feature + nf
                
#             for node3 in sub_list2[1:]:
#                 if node3 == -1:  nf = torch.zeros(features[node3].size(dim=0))
#                 else:   nf = features[node3]
                
#                 if feature2 == []:
#                     feature2 = nf
#                 else:
#                     feature2 = torch.cat([feature2, nf], dim=0)
                    
#                 feature2 = nf + feature
                    
#             feature = torch.cat([feature2, feature_raw, feature], dim=0)
#             nodes_features[node, i, :] = feature  
#             i += 1
            
#     print(nodes_features.size())       
#     return nodes_features   


# 上下文 cat  
# def get_token(features, W, num_steps, dataset, device):
#     # print(features.shape[1])
#     # nodes_features = torch.empty(features.shape[0], 2*W+1, features.shape[1]*(num_steps+1))
#     nodes_features = torch.empty(features.shape[0], W+1, features.shape[1]*(num_steps*2 + 1))
    
#     print('loading the pt file now')
#     pt = torch.load(dataset + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '.pt', map_location=torch.device('cpu'))
#     pt_back = torch.load(dataset + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '_back.pt', map_location=torch.device('cpu'))
#     # print(f"Random Walk Begin! Random Walk nums:{W}, length:{num_steps}")
#     for node in range(features.shape[0]):
#         # print(f'load the {node} node')
#         walk = pt[node].tolist()
#         walk_back = pt_back[node].tolist()
#         i = 0
#         # 遍历walk
#         for j in range(W):
#             sub_list = walk[j]
#             sub_list2 = walk_back[j]
#             feature = []
#             feature2 = []
#             # node2 = sub_list[0]
#             for node2 in sub_list:
#                 if node2 == -1:  nf = torch.zeros(features[node2].size(dim=0))
#                 else:   nf = features[node2]
                
#                 if feature == []:
#                     feature = nf
#                 else:
#                     feature = torch.cat([feature, nf], dim=0)

#             for node3 in sub_list2[1:]:
#                 if node3 == -1:  nf = torch.zeros(features[node3].size(dim=0))
#                 else:   nf = features[node3]
                
#                 if feature2 == []:
#                     feature2 = nf
#                 else:
#                     feature2 = torch.cat([feature2, nf], dim=0)
                    
#             feature = torch.cat([feature2, feature], dim=0)
#             nodes_features[node, i, :] = feature  
#             i += 1
            
#     return nodes_features


# 上下文 cat  
# def get_token(features, W, num_steps, dataset, device):
#     # print(features.shape[1])
#     # nodes_features = torch.empty(features.shape[0], 2*W+1, features.shape[1]*(num_steps+1))
#     nodes_features = torch.empty(features.shape[0], W+1, features.shape[1]*(num_steps*2 + 1))
    
#     print('loading the pt file now')
#     pt = torch.load(dataset + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '.pt', map_location=torch.device('cpu'))
#     pt_back = torch.load(dataset + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '_back.pt', map_location=torch.device('cpu'))
#     # print(f"Random Walk Begin! Random Walk nums:{W}, length:{num_steps}")
#     for node in range(features.shape[0]):
#         # print(f'load the {node} node')
#         walk = pt[node].tolist()
#         walk_back = pt_back[node].tolist()
#         i = 0
#         # 遍历walk
#         for j in range(W):
#             sub_list = walk[j]
#             sub_list2 = walk_back[j]
#             feature = []
#             feature2 = []
#             # node2 = sub_list[0]
#             for node2 in sub_list:
#                 if node2 == -1:  nf = torch.zeros(features[node2].size(dim=0))
#                 else:   nf = features[node2]
                
#                 if feature == []:
#                     feature = nf
#                 else:
#                     feature = torch.cat([feature, nf], dim=0)

#             for node3 in sub_list2[1:]:
#                 if node3 == -1:  nf = torch.zeros(features[node3].size(dim=0))
#                 else:   nf = features[node3]
                
#                 if feature2 == []:
#                     feature2 = nf
#                 else:
#                     feature2 = torch.cat([feature2, nf], dim=0)
                    
#             feature = torch.cat([feature2, feature], dim=0)
#             nodes_features[node, i, :] = feature  
#             i += 1
            
#     return nodes_features


# 上下文 cat, then linear  
# def get_token(features, W, num_steps, dataset, device):
#     # print(features.shape[1])
#     # nodes_features = torch.empty(features.shape[0], 2*W+1, features.shape[1]*(num_steps+1))
#     nodes_features = torch.empty(features.shape[0], W+1, features.shape[1]*(num_steps*2 + 1))
    
#     print('loading the pt file now')
#     pt = torch.load(dataset + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '.pt', map_location=torch.device('cpu'))
#     pt_back = torch.load(dataset + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '_back.pt', map_location=torch.device('cpu'))
#     # print(f"Random Walk Begin! Random Walk nums:{W}, length:{num_steps}")
#     for node in range(features.shape[0]):
#         # print(f'load the {node} node')
#         walk = pt[node].tolist()
#         walk_back = pt_back[node].tolist()
#         i = 0
#         # 遍历walk
#         for j in range(W):
#             sub_list = walk[j]
#             sub_list2 = walk_back[j]
#             feature = []
#             feature2 = []
#             # node2 = sub_list[0]
#             for node2 in sub_list:
#                 if node2 == -1:  nf = torch.zeros(features[node2].size(dim=0))
#                 else:   nf = features[node2]
                
#                 if feature == []:
#                     feature = nf
#                 else:
#                     feature = torch.cat([feature, nf], dim=0)

#             for node3 in sub_list2[1:]:
#                 if node3 == -1:  nf = torch.zeros(features[node3].size(dim=0))
#                 else:   nf = features[node3]
                
#                 if feature2 == []:
#                     feature2 = nf
#                 else:
#                     feature2 = torch.cat([feature2, nf], dim=0)
                    
#             feature = torch.cat([feature2, feature], dim=0)
#             nodes_features[node, i, :] = feature  
#             i += 1
            
#     return nodes_features


# pre-process
def get_token(features, W, num_steps, dataset, device):
    # print(features.shape[1])
    # nodes_features = torch.empty(features.shape[0], 2*W+1, features.shape[1]*(num_steps+1))
    nodes_features = torch.empty(features.shape[0], W+1, features.shape[1])
    
    print('loading the pt file now')
    pt = torch.load(dataset + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '.pt', torch.device('cpu'))
    # print(f"Random Walk Begin! Random Walk nums:{W}, length:{num_steps}")
    for node in range(features.shape[0]):
        print(f'load the {node} node')
        walk = pt[node].tolist()
        i = 0
        # 遍历walk
        for j in range(W):
            sub_list = walk[j]
            feature = []
            for node in sub_list:
                if feature == []:
                    feature = features[node]
                else:  
                    feature = feature + features[node] 

            nodes_features[node, i, :] = feature  
            i += 1

    torch.save(nodes_features, dataset + '_fea_t_num=' + str(W) + '_w_len=' + str(num_steps) + '.pt')
        
    # return nodes_features


def nor_matrix(adj, a_matrix):

    nor_matrix = torch.mul(adj, a_matrix)
    row_sum = torch.sum(nor_matrix, dim=1, keepdim=True)
    nor_matrix = nor_matrix / row_sum

    return nor_matrix