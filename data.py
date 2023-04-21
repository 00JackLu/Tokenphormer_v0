import utils
import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
import scipy.sparse as sp
import os.path
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import  CoraFullDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset,CoauthorCSDataset,CoauthorPhysicsDataset
import random
import pickle as pkl
import numpy as np
from make_dataset import get_train_val_test_split
from sklearn.preprocessing import StandardScaler
# from cache_sample import cache_sample_rand_csr

def get_dataset(dataset, pe_dim, rw_dim, split_seed=0):
    if dataset in {"arxiv", "products", "proteins", "papers100M", "mag"}:
        if dataset == "arxiv":
            dataset = DglNodePropPredDataset(name="ogbn-arxiv")
        elif dataset == "products":
            dataset = DglNodePropPredDataset(name="ogbn-products")
        elif dataset == "proteins":
            dataset = DglNodePropPredDataset(name="ogbn-proteins")
        elif dataset == "papers100M":
            dataset = DglNodePropPredDataset(name="ogbn-papers100M")
        elif dataset == "mag":
            dataset = DglNodePropPredDataset(name="ogbn-mag")
        split_idx = dataset.get_idx_split()
        graph, labels = dataset[0]
        features = graph.ndata['feat']
        adj = graph.adj(scipy_fmt="csr")
        # adj = cache_sample_rand_csr(adj, s_len)
        # print(labels)

        idx_train = split_idx['train']
        idx_val = split_idx['valid']
        idx_test = split_idx['test']

        graph = dgl.from_scipy(adj)

        adj = utils.sparse_mx_to_torch_sparse_tensor(adj)

        labels = labels.reshape(-1)

        # RWPE
        # lpe = utils.randomwalk_positional_encoding(adj, rw_dim)
        # features = torch.cat((features, lpe.ndata['PE']), dim=1)

        # LPE
        lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
        features = torch.cat((features, lpe), dim=1)
        
        

    elif dataset in {"pubmed", "corafull", "computer", "photo", "cs", "physics","cora", "citeseer"}:



        file_path = "dataset/"+dataset+".pt"

        data_list = torch.load(file_path)

        # data_list = [adj, features, labels, idx_train, idx_val, idx_test]
        adj = data_list[0]
        features = data_list[1]
        labels = data_list[2]

        idx_train = data_list[3]
        idx_val = data_list[4]
        idx_test = data_list[5]

        if dataset == "pubmed":
            graph = PubmedGraphDataset()[0]
        elif dataset == "corafull":
            graph = CoraFullDataset()[0]
        elif dataset == "computer":
            graph = AmazonCoBuyComputerDataset()[0]
        elif dataset == "photo":
            graph = AmazonCoBuyPhotoDataset()[0]
        elif dataset == "cs":
            graph = CoauthorCSDataset()[0]
        elif dataset == "physics":
            graph = CoauthorPhysicsDataset()[0]
        elif dataset == "cora":
            graph = CoraGraphDataset()[0]
        elif dataset == "citeseer":
            graph = CiteseerGraphDataset()[0]

        graph = dgl.to_bidirected(graph)

        # RWPE
        # lpe = utils.randomwalk_positional_encoding(adj, rw_dim)
        # features = torch.cat((features, lpe.ndata['PE']), dim=1)

        # LPE
        lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
        features = torch.cat((features, lpe), dim=1)

        # col_normalize
        # features = col_normalize(features)

        
    elif dataset == 'aminer':
        path = './dataset/'+dataset
        adj = pkl.load(open(os.path.join(path, "{}.adj.sp.pkl".format(dataset)), "rb"))
        features = pkl.load(
            open(os.path.join(path, "{}.features.pkl".format(dataset)), "rb"))
        labels = pkl.load(
            open(os.path.join(path, "{}.labels.pkl".format(dataset)), "rb"))
        random_state = np.random.RandomState(split_seed)
        idx_train, idx_val, idx_test = get_train_val_test_split(
            random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
        # idx_unlabel = np.concatenate((idx_val, idx_test))
        features = col_normalize(features)
        
        labels = torch.tensor(labels)
        idx_train = torch.tensor(idx_train)
        idx_val = torch.tensor(idx_val)
        idx_test = torch.tensor(idx_test)
        
        
        # graph = dgl.from_scipy(adj)
        
        # lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
       
        # features = torch.cat((features, lpe), dim=1)

        adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
         
        print(adj)
        
        labels = torch.argmax(labels, -1)


    elif dataset in {"reddit", "Amazon2M"}:

 
        file_path = './dataset/'+dataset+'.pt'

        data_list = torch.load(file_path)

        #adj, features, labels, idx_train, idx_val, idx_test

        adj = data_list[0]
        
        #print(type(adj))
        features = torch.tensor(data_list[1], dtype=torch.float32)
        labels = torch.tensor(data_list[2])
        idx_train = torch.tensor(data_list[3])
        idx_val = torch.tensor(data_list[4])
        idx_test = torch.tensor(data_list[5])

        graph = dgl.from_scipy(adj)

        adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
        

        labels = torch.argmax(labels, -1)

        # RWPE
        # lpe = utils.randomwalk_positional_encoding(adj, rw_dim)
        # features = torch.cat((features, lpe.ndata['PE']), dim=1)

        # LPE
        lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
        features = torch.cat((features, lpe), dim=1)
        

    return adj, features, labels, idx_train, idx_val, idx_test


def col_normalize(mx):
    """Column-normalize sparse matrix"""
    scaler = StandardScaler()

    mx = scaler.fit_transform(mx)

    return mx

