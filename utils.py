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
from dgl import DGLGraph
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


def random_walk_gen(G, t_num, w_len, dataset):
    G = dgl.DGLGraph(G)
    # nodes_features = torch.empty(G.nodes().size(dim=0), t_num*w_len)
    nodes_features = [] 
    # 遍历张量
    for i in range(G.nodes().size(dim=0)):
        strat_nodes = [i] * t_num

        token_path = dgl.sampling.random_walk(G, strat_nodes, length=w_len)
        print(token_path[0])
        nodes_features.append(token_path[0])
    torch.save(nodes_features, dataset + '_t_num=' + str(t_num) + '_w_len=' + str(w_len) + '.pt')
        
    
# def get_token(G, features, W, num_steps, dataset):
#     # print(features.shape[1])
#     nodes_features = torch.empty(features.shape[0], (W*(num_steps + 1))+1, features.shape[1]*(num_steps + 1))
    
#     print('loading the pt file now')
#     # print(f"Random Walk Begin! Random Walk nums:{W}, length:{num_steps}")
#     for node in range(features.shape[0]):
#         # walk = walker.random_walks(G, n_walks=W, walk_len=num_steps, start_nodes=[node])
#         # walk = walk.tolist()
#         # print(walk)
       
#         pt = torch.load(dataset + '_t_num=' + str(W) + '_w_len=' + str(num_steps) + '.pt', map_location=torch.device('cpu'))
#         walk = pt[node].tolist()
#         # print(walk)
#         # add the features of the nodes in the random walk to the nodes_features tensor      
#         i = 1
#         # 遍历walk
#         for j in range(len(walk)):
#             sub_list = walk[j]
#             feature = []
#             for node in sub_list:  
#                 if feature == []:
#                     feature = features[node]
#                 else:
#                     # feature = feature+features[node]
#                     feature = torch.cat([feature, features[node]], dim=0)
#                 temp = torch.zeros(features[node].size(dim=0) * (num_steps + 1) - feature.size(dim=0))               
                    
                   
#                 # print(features[node])
#                 nodes_features[node, i, :] = torch.cat([feature, temp], dim=0)      
#                 i += 1
#             # nodes_features[node, j, :] = torch.cat(node_features, dim=0)
            
#     print(f"Random Walk Done!")
#     print(nodes_features.size())
#     # nodes_features = nodes_features.squeeze()            
#     return nodes_features


def get_token(G, features, W, num_steps):
    print(features.shape[1])
    
    nodes_features = torch.empty(features.shape[0], (W*num_steps)+1, features.shape[1]*num_steps)
    
    print(f"Random Walk Begin! Random Walk nums:{W}, length:{num_steps}")
    for node in range(features.shape[0]):
        walk = walker.random_walks(G, n_walks=W, walk_len=num_steps, start_nodes=[node])
        walk = walk.tolist()
        print(walk)
        # add the features of the nodes in the random walk to the nodes_features tensor      
        i = 1
        # 遍历walk
        for j in range(len(walk)):
            sub_list = walk[j]
            feature = []
            

            for node in sub_list:  
                if feature == []:
                    feature = features[node]
                else:
                    # feature = feature+features[node]
                    feature = torch.cat([feature, features[node]], dim=0)
                temp = torch.zeros(features[node].size(dim=0) * num_steps - feature.size(dim=0))               
                    
                   
                # print(features[node])
                nodes_features[node, i, :] = torch.cat([feature, temp], dim=0)      
                i += 1
            # nodes_features[node, j, :] = torch.cat(node_features, dim=0)
            
    print(f"Random Walk Done!")
    print(nodes_features.size())
    # nodes_features = nodes_features.squeeze()            
    return nodes_features


# def compute_pagerank(g):
#     DAMP = 0.85
#     K = 10
#     N = g.number_of_nodes()
#     g.ndata["pv"] = torch.ones(N) / N
#     degrees = g.out_degrees(g.nodes()).type(torch.float32)
#     for k in range(K):
#         g.ndata["pv"] = g.ndata["pv"] / degrees
#         g.update_all(
#             message_func=fn.copy_u(u="pv", out="m"),
#             reduce_func=fn.sum(msg="m", out="pv"),
#         )
#         g.ndata["pv"] = (1 - DAMP) / N + DAMP * g.ndata["pv"]
#     return g.ndata["pv"]


# def get_token(G, features, W, num_steps):
    
#     nodes_features = torch.empty(features.shape[0], (W*num_steps)+1, features.shape[1])

#     g = dgl.from_networkx(G)

    
#     pv = compute_pagerank(g)

#     print(f"Random Walk Begin! Random Walk nums:{W}, length:{num_steps}, ")
#     p = 1.0
#     q = 0.5


#     # Convert the PageRank scores to probabilities
#     pr_probs = torch.softmax(pv, dim=0)
    
#     walks = dgl.sampling.random_walk(
#         G, 
#         prob=pr_probs, num_walks=W, 
#         walk_length=num_steps, p=p, q=q
#     )
#     # walks = dgl.sampling.random_walk(G, seeds=[0], length=num_steps, num_walks=W)
#     print(f"Random Walk Done!")
#     for node in range(features.shape[0]):
#         walk = walks.tolist()[node]
#         print(walk)
#         # add the features of the nodes in the random walk to the nodes_features tensor      
#         i = 1
#         # 遍历walk
#         for j in range(len(walk)):
#             sub_list = walk[j]
#             feature = []
            
#             for node in sub_list:  
#                 if feature == []:
#                     feature = features[node]
#                 else:
#                     feature = feature+features[node]
#                 # print(features[node])
#                 nodes_features[node, i, :] = feature      
#                 i += 1
#             # nodes_features[node, j, :] = torch.cat(node_features, dim=0)
    
#     print(nodes_features)
#     # nodes_features = nodes_features.squeeze()            
#     return nodes_features
     
            
# def re_features(adj, features, K):
#     #传播之后的特征矩阵,size= (N, 1, K+1, d )
#     nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])

#     for i in range(features.shape[0]):

#         nodes_features[i, 0, 0, :] = features[i]

#     x = features + torch.zeros_like(features)

#     for i in range(K):

#         x = torch.matmul(adj, x)

#         for index in range(features.shape[0]):

#             nodes_features[index, 0, i + 1, :] = x[index]        
  
#     nodes_features = nodes_features.squeeze()


#     return nodes_features


def nor_matrix(adj, a_matrix):

    nor_matrix = torch.mul(adj, a_matrix)
    row_sum = torch.sum(nor_matrix, dim=1, keepdim=True)
    nor_matrix = nor_matrix / row_sum

    return nor_matrix




