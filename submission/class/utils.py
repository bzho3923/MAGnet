import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
from torch_geometric.datasets import Planetoid,WikiCS,Coauthor
from torch_geometric.data import Data
from scipy.sparse import csr_matrix
from scipy import sparse
import networkx as nx
import os
import torch_geometric.transforms as T
from new_dataset import HetroDataSet
from ogb.nodeproppred import PygNodePropPredDataset
@torch.no_grad()
def scipy_to_torch_sparse(A):
    A = sparse.coo_matrix(A)
    row = torch.tensor(A.row)
    col = torch.tensor(A.col)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data)

    return torch.sparse_coo_tensor(index, value, A.shape)

def ChebyshevApprox(f, n):  # assuming f : [0, pi] -> R
    quad_points = 500
    c = np.zeros(n)
    a = np.pi / 2
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(a * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)

    return c

def get_operator(L, DFilters, n, s, J, Lev):
    r = len(DFilters)
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)
    a = np.pi / 2  # consider the domain of masks as [0, pi]
    # Fast Tight Frame Decomposition (FTFD)
    FD1 = sparse.identity(L.shape[0])
    d = dict()
    for l in range(1, Lev + 1):
        for j in range(r):
            T0F = FD1
            T1F = ((s ** (-J + l - 1) / a) * L) @ T0F - T0F
            d[j, l - 1] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J + l - 1)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l - 1] += c[j][k] * TkF
        FD1 = d[0, l - 1]
    return d

def WT_Recon(d, L, RFilters, n, s, J, Lev):
    r = len(RFilters)
    a = np.pi / 2  # consider the domain of masks as [0, pi]
    c_rec = [None] * r
    for j in range(r):
        c_rec[j], _ = ChebyshevApprox(RFilters[j], n)
    FD_recl = 0
    for l in np.arange(1, Lev + 1)[::-1]:
        for j in range(r):
            if (l == Lev) or (j > 0):
                T0F = d[j, l - 1]
            else:
                T0F = FD_rec
            T1F = ((s ** (-J + l - 1) / a) * L) @ T0F - T0F
            djl = (1 / 2) * c_rec[j][0] * T0F + c_rec[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J + l - 1)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                djl += c_rec[j][k] * TkF
            FD_recl += djl
        FD_rec = FD_recl
        FD_recl = 0

    return FD_rec



def load_anomaly_detection_dataset(dataset, datadir='data'):
    
    data_mat = sio.loadmat(f'{datadir}/{dataset}.mat')
    adj = data_mat['Network']
    feat = data_mat['Attributes']
    truth = data_mat['Label']
    truth = truth.flatten()

    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_norm = adj_norm.toarray()
    adj = adj + sp.eye(adj.shape[0])
    adj = adj.toarray()
    feat = feat.toarray()
    return adj_norm, feat, truth, adj


def load_anomaly_npy_dataset(dataname="PubMed", datanum = 10, datadir='data'):  ##load cora dataset
    rootname = dataname#"./Citeseer"
    #dataset = Planetoid(root=rootname, name=dataname)
    if dataname.lower()=="ogbn-arxiv":
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/', transform=T.ToSparseTensor())
        data = dataset[0]
        feat = np.load("./" + str(dataname.lower()) + "_injection/" + str(dataname.lower()) + "noise_mat_ratiio" + str(datanum) + ".npy")
    if dataname.lower() == 'cs':
        dataset = Coauthor(root="./CS", name=dataname)
        data = Data(x=dataset[0].x, edge_index=dataset[0].edge_index, y=dataset[0].y)
        feat = np.load("./" + str(dataname.lower()) + "_injection/" + str(dataname.lower()) + "noise_mat_ratiio" + str(datanum) + ".npy")
    if dataname.lower() == 'wikics':
        rootname = "./wikics/"
        dataset = WikiCS(root=rootname)
        feat = np.load("./"+str(dataname.lower())+"_injection/"+str(dataname.lower())+"noise_mat_ratiio"+str(datanum)+ ".npy")#np.load("./wikicsout_feat_mat" + str(datanum) + ".npy")
    if dataname.lower() == 'cora' or dataname.lower() == 'citeseer' or dataname.lower() == 'pubmed':
        dataset = Planetoid(root=rootname, name=dataname)
        feat = np.load("./"+str(dataname.lower())+"_injection/"+str(dataname.lower())+"noise_mat_ratiio"+str(datanum)+ ".npy")#np.load("./"+str(dataname.lower())+"_injection/"+str(dataname.lower())+"out_feat_mat"+str(datanum)+ ".npy")
    if dataname.lower() == 'wisconsin' or dataname.lower() == 'texas':
        dataset = HetroDataSet(root=rootname, name=dataname)
        feat = np.load("./"+str(dataname.lower())+"_injection/"+str(dataname.lower())+"noise_mat_ratiio"+str(datanum)+ ".npy")

    edge_index = dataset[0].edge_index
    truth = np.array(dataset[0].y)
    if dataname.lower() == "ogbn-arxiv":
        edge_index = dataset[0].adj_t.to_symmetric()
    return edge_index, feat, truth


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()