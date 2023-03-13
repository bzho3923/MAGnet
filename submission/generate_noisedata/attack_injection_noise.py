import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian
import math
import argparse
import os.path as osp
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
import os
from torch_geometric.datasets import Planetoid, WikiCS,Coauthor
from torch_geometric.data import Data
import networkx as nx
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import random

##read the cora feature and inject the anomal
##after the inkection, the anomal dataset will be saved as npy file for further processing

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wisconsin',
                        help='name of dataset (default: cora)')
    parser.add_argument('--ratio_anomal', type=float, default=9150,
                        help='number of repetitions (default: 10)')
    parser.add_argument('--compare_num', type=int, default=500,
                        help='number of repetitions (default: 10)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--filename', type=str, default='results',
                        help='filename to store results and the model (default: results)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Training on CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # load dataset
    dataname = args.dataset
    rootname = osp.join('./', dataname)
    if dataname.lower() == 'wisconsin' or dataname.lower() == 'texas':
        graph_adjacency_list_file_path = os.path.join('./new_data', dataname, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataname,f'out1_node_feature_label.txt')
        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))
        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        feat_mat = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        num_nodes = feat_mat.shape[0]
        labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    if dataname.lower()=="ogbn-arxiv":
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/')
        data = dataset[0]
    if dataname.lower() == 'wikics':
        dataset = WikiCS(root=rootname)
        data = dataset[0]
    if dataname.lower() == 'cs':
        dataset = Coauthor(root="./CS",name = dataname)
        train_mask = torch.zeros_like(dataset[0].y,dtype=torch.bool)
        train_mask[0:300] = 1
        val_mask = torch.zeros_like(dataset[0].y, dtype=torch.bool)
        val_mask[300:500] = 1
        test_mask = torch.zeros_like(dataset[0].y, dtype=torch.bool)
        test_mask[500:1500]=1
        data = Data(x=dataset[0].x,edge_index=dataset[0].edge_index,y=dataset[0].y,train_mask=train_mask,val_mask = val_mask,test_mask=test_mask)
    if dataname.lower() == 'cora' or dataname.lower() == 'citeseer' or dataname.lower() == 'pubmed':
        dataset = Planetoid(root=rootname, name=dataname)
        data = dataset[0]

    ##feature matrix
    if dataname.lower() != 'wisconsin' and dataname.lower() != 'texas':
        num_nodes = data.x.shape[0]
        feat_mat = np.array(dataset[0].x)
    out_feat = np.array(feat_mat, copy=True)
    ratio_list = [0.1,0.5]
    for ratio in ratio_list:
        num_anomal = int(num_nodes*ratio)
        anomal_list = np.random.randint(0,num_nodes,num_anomal)
        switched_node = np.zeros((num_anomal,2))
        ##randomly select 50 nodes for distance measuring
        count = 0
        for i in anomal_list:
            tmp_node = np.random.randint(0, num_nodes, args.compare_num)
            max_dist=0
            for j in tmp_node:
                tmp_dist = np.sum((feat_mat[i]-feat_mat[j])**2)
                # print("dist:",tmp_dist)
                if tmp_dist>max_dist:
                    max_dist = tmp_dist
                    max_node = j
            print("switching node",str(i),"and node", str(j))
            switched_node[count,0] = i
            switched_node[count, 1] = j
            count +=1
            out_feat[i] = feat_mat[j]
        save_pth="./"+str(dataname)+"_injection/"
        os.makedirs(save_pth,exist_ok=True)
        np.save(save_pth+str(dataname)+"noise_mat_ratiio"+str(ratio),out_feat)
    #np.save("feat_mat",feat_mat)
    # np.save(str(dataname)+"switched_node"+str(args.num_anomal), switched_node)
    # print("out_feat:",out_feat[100:120,100:120],feat_mat[100:120,100:120])

    # extract the data
    #data = dataset[0].to(device)
    #print(data.y[data.train_mask].shape,"\n",data.y[0:15],data.train_mask.shape,data.train_mask[120:150])