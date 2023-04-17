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
import walker
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

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix


# pos_enc_dim: position embedding size
def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian

    #adjacency_matrix(transpose, scipy_fmt="csr")
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    return lap_pos_enc


# add RWPE from dgl
def randomwalk_positional_encoding(adj):
    edge_index = adj._indices()
    G = dgl.graph((edge_index[0], edge_index[1]))
    transform = RandomWalkPE(k=5)
    return transform(G)


def random_walk_gen(adj, t_num, w_len, dataset):
    edge_index = adj._indices()
    G = dgl.graph((edge_index[0], edge_index[1]))
    nodes_features = [] 
    # 遍历张量
    for i in range(G.nodes().size(dim=0)):
        strat_nodes = [i] * t_num
        token_path = dgl.sampling.random_walk(G, strat_nodes, length=w_len)
        nodes_features.append(token_path[0])
    torch.save(nodes_features, dataset + '_t_num=' + str(t_num) + '_w_len=' + str(w_len) + '.pt')
        

def get_feature(node, features, num_steps, pt, W):   
    processed_features = torch.empty((W*(num_steps + 1))+1, features.shape[1]*(num_steps + 1))
    walk = pt[node].tolist()     
    i = 1
    # 遍历walk
    for j in range(len(walk)):
        sub_list = walk[j]
        feature = []
        for sub_node in sub_list:
            if feature == []:
                feature = features[sub_node]
            else:
                feature = torch.cat([feature, features[sub_node]], dim=0)
            temp = torch.zeros(features[sub_node].size(dim=0) * (num_steps + 1) - feature.size(dim=0))               
                
            processed_features[i, :] = torch.cat([feature, temp], dim=0)      
            i += 1
    return processed_features


# def get_token(features, W, num_steps, dataset, device, idx_train, idx_val, idx_test):
#     # initial the tensor_size
#     # print(f"initial the tensor_size:{features.shape[0],(W*(num_steps + 1))+1,features.shape[1]*(num_steps + 1)}")
#     print('--------------------------------------------')
#     processed_train_features = torch.empty(len(idx_train), (W*(num_steps + 1))+1, features.shape[1]*(num_steps + 1))
#     processed_val_features = torch.empty(len(idx_val), (W*(num_steps + 1))+1, features.shape[1]*(num_steps + 1))
#     processed_test_features = torch.empty(len(idx_test), (W*(num_steps + 1))+1, features.shape[1]*(num_steps + 1))
    
#     # nodes_features = torch.empty(features.shape[0], (W*(num_steps + 1))+1, features.shape[1]*(num_steps + 1))
    
#     print('loading the random walk file now')
#     print('--------------------------------------------')
#     # pt = torch.load(dataset + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '.pt', map_location=torch.device('cpu'))
#     pt = torch.load(dataset + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '.pt', map_location=device)

#     for node in range(features.shape[0]):
#         print(f'load the {node} node')
#         if node in idx_train:                
#             processed_train_features[node,:] = get_feature(node, features, num_steps, pt, W)          
#         elif node in idx_val:
#             processed_val_features[node,:] = get_feature(node, features, num_steps, pt, W)       
#         elif node in idx_test:
#             processed_test_features[node,:] = get_feature(node, features, num_steps, pt, W)       
#     return processed_train_features, processed_val_features, processed_test_features
        
    
#     # save the nodes_feature tensor       
#     # feature_path = dataset + '_feature' + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '.pt'
#     # torch.save(nodes_features, feature_path)     
#     # print("save feature success")
#     print("nodes_features.size() :", nodes_features.size())      
#     print('--------------------------------------------')     
#     return nodes_features


# def get_token(features, W, num_steps, dataset, device):
#     # initial the tensor_size
#     print(f"initial the tensor_size:{features.shape[0],(W*(num_steps + 1))+1,features.shape[1]*(num_steps + 1)}")
#     print('--------------------------------------------')
#     nodes_features = torch.empty(features.shape[0], (W*(num_steps + 1))+1, features.shape[1]*(num_steps + 1))
    
#     print('loading the random walk file now')
#     print('--------------------------------------------')
#     # pt = torch.load(dataset + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '.pt', map_location=torch.device('cpu'))
#     pt = torch.load(dataset + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '.pt', map_location=device)

#     for node in range(features.shape[0]):
#         print(f'load the {node} node')
#         walk = pt[node].tolist()     
#         i = 1
#         # 遍历walk
#         for j in range(len(walk)):
#             sub_list = walk[j]
#             feature = []
#             for sub_node in sub_list:  
#                 if feature == []:
#                     feature = features[sub_node]
#                 else:
#                     feature = torch.cat([feature, features[sub_node]], dim=0)
#                 temp = torch.zeros(features[sub_node].size(dim=0) * (num_steps + 1) - feature.size(dim=0))               
                    
#                 nodes_features[node, i, :] = torch.cat([feature, temp], dim=0)      
#                 i += 1
    
#     # save the nodes_feature tensor       
#     # feature_path = dataset + '_feature' + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '.pt'
#     # torch.save(nodes_features, feature_path)     
#     # print("save feature success")
#     print("nodes_features.size() :", nodes_features.size())      
#     print('--------------------------------------------')     
#     return nodes_features


# wrong 1
def get_token(features, W, num_steps, dataset, device):
    # initial the tensor_size
    print(f"initial the tensor_size:{features.shape[0],(W*(num_steps + 1))+1,features.shape[1]*(num_steps + 1)}")
    print('--------------------------------------------')
    nodes_features = torch.empty(features.shape[0], (W*(num_steps + 1))+1, features.shape[1]*(num_steps + 1))
    
    print('loading the random walk file now')
    print('--------------------------------------------')
    # pt = torch.load(dataset + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '.pt', map_location=torch.device('cpu'))
    pt = torch.load(dataset + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '.pt', map_location=device)

    for node in range(features.shape[0]):
        print(f'load the {node} node')
        walk = pt[node].tolist()     
        i = 1
        # 遍历walk
        for j in range(len(walk)):
            sub_list = walk[j]
            feature = []
            for node in sub_list:  
                if feature == []:
                    feature = features[node]
                else:
                    feature = torch.cat([feature, features[node]], dim=0)
                temp = torch.zeros(features[node].size(dim=0) * (num_steps + 1) - feature.size(dim=0))               
                # print(f'the {node}')    
                nodes_features[node, i, :] = torch.cat([feature, temp], dim=0)      
                i += 1
    
    # save the nodes_feature tensor       
    # feature_path = dataset + '_feature' + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '.pt'
    # torch.save(nodes_features, feature_path)     
    # print("save feature success")
    print("nodes_features.size() :", nodes_features.size())      
    print('--------------------------------------------')     
    return nodes_features


def nor_matrix(adj, a_matrix):

    nor_matrix = torch.mul(adj, a_matrix)
    row_sum = torch.sum(nor_matrix, dim=1, keepdim=True)
    nor_matrix = nor_matrix / row_sum

    return nor_matrix


# class RandomWalkPE(BaseTransform):
#     r"""Random Walk Positional Encoding, as introduced in
#     `Graph Neural Networks with Learnable Structural and Positional Representations
#     <https://arxiv.org/abs/2110.07875>`__

#     This module only works for homogeneous graphs.

#     Parameters
#     ----------
#     k : int
#         Number of random walk steps. The paper found the best value to be 16 and 20
#         for two experiments.
#     feat_name : str, optional
#         Name to store the computed positional encodings in ndata.
#     eweight_name : str, optional
#         Name to retrieve the edge weights. Default: None, not using the edge weights.

#     Example
#     -------

#     >>> import dgl
#     >>> from dgl import RandomWalkPE

#     >>> transform = RandomWalkPE(k=2)
#     >>> g = dgl.graph(([0, 1, 1], [1, 1, 0]))
#     >>> g = transform(g)
#     >>> print(g.ndata['PE'])
#     tensor([[0.0000, 0.5000],
#             [0.5000, 0.7500]])
#     """
#     def __init__(self, k, feat_name='PE', eweight_name=None):
#         self.k = k
#         self.feat_name = feat_name
#         self.eweight_name = eweight_name

#     def __call__(self, g):
#         PE = functional.random_walk_pe(g, k=self.k, eweight_name=self.eweight_name)
#         g.ndata[self.feat_name] = F.copy_to(PE, g.device)

#         return g


# [docs]class LaplacianPE(BaseTransform):
#     r"""Laplacian Positional Encoding, as introduced in
#     `Benchmarking Graph Neural Networks
#     <https://arxiv.org/abs/2003.00982>`__

#     This module only works for homogeneous bidirected graphs.

#     Parameters
#     ----------
#     k : int
#         Number of smallest non-trivial eigenvectors to use for positional encoding.
#     feat_name : str, optional
#         Name to store the computed positional encodings in ndata.
#     eigval_name : str, optional
#         If None, store laplacian eigenvectors only.
#         Otherwise, it's the name to store corresponding laplacian eigenvalues in ndata.
#         Default: None.
#     padding : bool, optional
#         If False, raise an exception when k>=n.
#         Otherwise, add zero paddings in the end of eigenvectors and 'nan' paddings
#         in the end of eigenvalues when k>=n.
#         Default: False.
#         n is the number of nodes in the given graph.

#     Example
#     -------
#     >>> import dgl
#     >>> from dgl import LaplacianPE
#     >>> transform1 = LaplacianPE(k=3)
#     >>> transform2 = LaplacianPE(k=5, padding=True)
#     >>> transform3 = LaplacianPE(k=5, feat_name='eigvec', eigval_name='eigval', padding=True)
#     >>> g = dgl.graph(([0,1,2,3,4,2,3,1,4,0], [2,3,1,4,0,0,1,2,3,4]))
#     >>> g1 = transform1(g)
#     >>> print(g1.ndata['PE'])
#     tensor([[ 0.6325,  0.1039,  0.3489],
#             [-0.5117,  0.2826,  0.6095],
#             [ 0.1954,  0.6254, -0.5923],
#             [-0.5117, -0.4508, -0.3938],
#             [ 0.1954, -0.5612,  0.0278]])
#     >>> g2 = transform2(g)
#     >>> print(g2.ndata['PE'])
#     tensor([[-0.6325, -0.1039,  0.3489, -0.2530,  0.0000],
#             [ 0.5117, -0.2826,  0.6095,  0.4731,  0.0000],
#             [-0.1954, -0.6254, -0.5923, -0.1361,  0.0000],
#             [ 0.5117,  0.4508, -0.3938, -0.6295,  0.0000],
#             [-0.1954,  0.5612,  0.0278,  0.5454,  0.0000]])
#     >>> g3 = transform3(g)
#     >>> print(g3.ndata['eigval'])
#     tensor([[0.6910, 0.6910, 1.8090, 1.8090,    nan],
#             [0.6910, 0.6910, 1.8090, 1.8090,    nan],
#             [0.6910, 0.6910, 1.8090, 1.8090,    nan],
#             [0.6910, 0.6910, 1.8090, 1.8090,    nan],
#             [0.6910, 0.6910, 1.8090, 1.8090,    nan]])
#     >>> print(g3.ndata['eigvec'])
#     tensor([[ 0.6325, -0.1039,  0.3489,  0.2530,  0.0000],
#             [-0.5117, -0.2826,  0.6095, -0.4731,  0.0000],
#             [ 0.1954, -0.6254, -0.5923,  0.1361,  0.0000],
#             [-0.5117,  0.4508, -0.3938,  0.6295,  0.0000],
#             [ 0.1954,  0.5612,  0.0278, -0.5454,  0.0000]])
#     """
#     def __init__(self, k, feat_name='PE', eigval_name=None, padding=False):
#         self.k = k
#         self.feat_name = feat_name
#         self.eigval_name = eigval_name
#         self.padding = padding

#     def __call__(self, g):
#         if self.eigval_name:
#             PE, eigval = functional.laplacian_pe(g, k=self.k, padding=self.padding,
#                                                  return_eigval=True)
#             eigval = F.repeat(F.reshape(eigval, [1,-1]), g.num_nodes(), dim=0)
#             g.ndata[self.eigval_name] = F.copy_to(eigval, g.device)
#         else:
#             PE = functional.laplacian_pe(g, k=self.k, padding=self.padding)
#         g.ndata[self.feat_name] = F.copy_to(PE, g.device)

#         return g

