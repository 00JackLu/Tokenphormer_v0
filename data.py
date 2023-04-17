import utils
import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
import scipy.sparse as sp
import os.path
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import  CoraFullDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset,CoauthorCSDataset,CoauthorPhysicsDataset
import random
# from cache_sample import cache_sample_rand_csr

def get_dataset(dataset, pe_dim):
    if dataset in {"ogbn-arxiv", "ogbn-products", "ogbn-proteins", "ogbn-papers100M", "ogbn-mag"}:
        if dataset == "ogbn-arxiv":
            dataset = DglNodePropPredDataset(name="ogbn-arxiv")
        elif dataset == "ogbn-products":
            dataset = DglNodePropPredDataset(name="ogbn-products")
        elif dataset == "ogbn-proteins":
            dataset = DglNodePropPredDataset(name="ogbn-proteins")
        elif dataset == "ogbn-papers100M":
            dataset = DglNodePropPredDataset(name="ogbn-papers100M")
        elif dataset == "ogbn-mag":
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
        lpe = utils.randomwalk_positional_encoding(adj)
        features = torch.cat((features, lpe), dim=1)
        labels = labels.reshape(-1)

    elif dataset in {"pubmed", "corafull", "computer", "photo", "cs", "physics","cora", "citeseer"}:



        file_path = "dataset/"+dataset+".pt"

        data_list = torch.load(file_path)

        # data_list = [adj, features, labels, idx_train, idx_val, idx_test]
        adj = data_list[0]
        print(adj)
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
        # lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
        lpe = utils.randomwalk_positional_encoding(adj)
        
        features = torch.cat((features, lpe.ndata['PE']), dim=1)



    elif dataset in {"aminer", "reddit", "Amazon2M"}:

 
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


        lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
       
        features = torch.cat((features, lpe), dim=1)

        adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
        

        labels = torch.argmax(labels, -1)
        

    return adj, features, labels, idx_train, idx_val, idx_test




