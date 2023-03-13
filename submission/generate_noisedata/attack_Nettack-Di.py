import torch
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from deeprobust.graph.defense import *#GCN, GAT, GIN, JK, GCN_attack,accuracy_1
from tqdm import tqdm
import scipy
import numpy as np
from sklearn.preprocessing import normalize
import pickle
from torch_geometric.datasets import Planetoid
import os.path as osp
from scipy.sparse import csr_matrix
from torch_geometric.utils import get_laplacian, degree, to_scipy_sparse_matrix
import argparse
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


def accuracy_1(output, labels):
    """"""
    try:
        num = len(labels)
    except:
        num = 1

    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor([labels])

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / num, preds, labels

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=14, help='Random seed.')
# cora and citeseer are binary, pubmed has not binary features
parser.add_argument('--dataset', type=str, default='ogbn', help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.1,  help='pertubation rate')
parser.add_argument('--modelname', type=str, default='GCN',  choices=['GCN', 'GAT','GIN', 'JK'])
parser.add_argument('--defensemodel', type=str, default='GCNJaccard',  choices=['GCNJaccard', 'RGCN', 'GCNSVD'])
parser.add_argument('--GNNGuard', type=bool, default=False,  choices=[True, False])
parser.add_argument('--DPlabel', type=int, default=9,  help='0-10')
parser.add_argument('--num_targets', type=int, default=541,
                        help='number of target nodes')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dataset.lower()=="cora":
    rootname = osp.join('./', "cora")
    dataset = Planetoid(root=rootname, name="cora")
if args.dataset.lower()=="ogbn":
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    #dataset = DglNodePropPredDataset(name='ogbn-arxiv')
data = dataset[0]
num_node = data.x.shape[0]
row, col = data.edge_index[0], data.edge_index[1]
adj_value = np.ones_like(row)
adj = csr_matrix((adj_value, (row, col)), shape=(num_node, num_node))
features = data.x
# adj = data.edge_index
labels = data.y
print("adj",adj.shape)



if scipy.sparse.issparse(features)==False:
    features = scipy.sparse.csr_matrix(features)

"""set the number of training/val/testing nodes"""
# data = Dataset(root='../Datasets/', name=args.dataset)
# idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx = np.array(range(0,2708))
if args.dataset.lower()=="cora":
    idx_train, idx_val, idx_test = idx[data.train_mask],idx[data.val_mask],idx[data.test_mask]
if args.dataset.lower()=="ogbn":
    split_idx = dataset.get_idx_split()
    labels = data.y.numpy().reshape(1,-1)[0]
    idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
print("idx:",idx_train,idx_val.shape)
"""add undirected edges, orgn-arxiv is directed graph, we transfer it to undirected closely following 
https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv
"""
adj = adj + adj.T
adj[adj>1] = 1


# Setup Surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
                with_relu=False, device=device)
surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, train_iters=201)  # change this train_iters to 201: train_iters=201
print("suuogate model trained!!")
# Setup Attack Model
target_node = 859

model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=False, attack_features=True, device=device)###main attack model
model = model.to(device)

def main():
    degrees = adj.sum(0).A1
    # How many perturbations to perform. Default: Degree of the node
    n_perturbations = int(degrees[target_node])

    # # indirect attack/ influencer attack
    model.attack(features, adj, labels, target_node, n_perturbations, direct=True)
    modified_adj = model.modified_adj
    modified_features = model.modified_features
    modified_features = modified_features.toarray()
    np.save("mod_feat400", modified_features)
    print('=== testing GNN on original(clean) graph ===')
    test(adj, features, target_node,  attention=args.GNNGuard)

    print('=== testing GCN on perturbed graph ===')
    test(modified_adj, modified_features, target_node,attention=args.GNNGuard)


def test(adj, features, target_node, attention=False):
    ''
    """test on GCN """
    """model_name could be 'GCN', 'GAT', 'GIN','JK'  """
    # for orgn-arxiv: nhid =256, layers =3, epoch =500

    gcn = globals()[args.modelname](nfeat=features.shape[1], nhid=256,  nclass=labels.max().item() + 1, dropout=0.5,
              device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train, idx_val=idx_val,
            idx_test=idx_test,
            attention=attention, verbose=True, train_iters=81)
    gcn.eval()
    _, output = gcn.test(idx_test=idx_test)

    probs = torch.exp(output[[target_node]])[0]
    print('probs: {}'.format(probs.detach().cpu().numpy()))
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("Test set results:",
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()


def multi_test(args):   ###multi attack and multi test
    cnt = 0
    degrees = adj.sum(0).A1
    #node_list = select_nodes(num_target=100)
    node_list = np.random.randint(0,num_node,args.num_targets)
    print("selectesd node list:",node_list)

    num = len(node_list)
    print('=== Attacking %s nodes respectively ===' % num)
    num_tar = 0
    out_feat = features
    tmp_feat = features
    for target_node in tqdm(node_list):
        n_perturbations = int(degrees[target_node])
        print("current node:",target_node,n_perturbations)
        if n_perturbations <1:  # at least one perturbation
            continue
        model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=False, attack_features=True, device=device)
        model = model.to(device)
        model.attack(tmp_feat, adj, labels, target_node, n_perturbations, direct=True, verbose=False)
        modified_features = model.modified_features
        # np.save("mod_adj",modified_adj)
        tmp_feat[target_node] = modified_features[target_node]
        # sp.save_npz("./mod_feat.npz",modified_features)
        # acc = single_test(modified_adj, modified_features, target_node)
        # if acc == 0:
        #     cnt += 1
        # num_tar += 1
        # print('classification rate : %s' % (1-cnt/num_tar), '# of targets:', num_tar)
        np.save(str(args.dataset)+"im2out_feat_mat"+str(args.num_targets), tmp_feat.toarray())
"""Set attention"""
attention = args.GNNGuard

def single_test(adj, features, target_node):
    'ALL the baselines'

    # """defense models"""
    # classifier = globals()[args.defensemodel](nnodes=adj.shape[0], nfeat=features.shape[1], nhid=16,
    #                                           nclass=labels.max().item() + 1, dropout=0.5, device=device)

    # ''' test on GCN (poisoning attack), model could be GCN, GAT, GIN'''
    classifier = globals()[args.modelname](nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    classifier = classifier.to(device)
    classifier.fit(features, adj, labels, idx_train,
                   idx_val=idx_val,
                   idx_test=idx_test,
                   verbose=False, attention=attention) #model_name=model_name
    classifier.eval()
    #acc_overall, output = classifier.test(idx_test, ) #model_name=model_name
    acc_overall = classifier.test(idx_test, )  # model_name=model_name

    # probs = torch.exp(output[[target_node]])
    # acc_test, pred_y, true_y = accuracy_1(output[[target_node]], labels[target_node])
    # print('target:{}, pred:{}, label: {}'.format(target_node, pred_y.item(), true_y.item()))
    # print('Pred probs', probs.data)

    print("accuracy overall",acc_overall)
    return acc_overall#acc_test.item()

"""=======Basic Functions============="""
def select_nodes(num_target = 10):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''
    gcn = globals()[args.modelname](nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train, idx_test, verbose=True)
    gcn.eval()
    output = gcn.predict()
    degrees = adj.sum(0).A1

    margin_dict = {}
    for idx in tqdm(idx_test):    ##select node  according to the test accuracy
        margin = classification_margin(output[idx], labels[idx])
        acc, _, _ = accuracy_1(output[[idx]], labels[idx])
        if acc==0 or int(degrees[idx])<1: # only keep the correctly classified nodes
            continue
        """check the outliers:"""
        neighbours = list(adj.todense()[idx].nonzero()[1])
        y = [labels[i] for i in neighbours]
        node_y = labels[idx]
        aa = node_y==y
        outlier_score = 1- aa.sum()/len(aa)
        if outlier_score >=0.5:
            continue

        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    high = [x for x, y in sorted_margins[: num_target]]
    low = [x for x, y in sorted_margins[-num_target: ]]
    other = [x for x, y in sorted_margins[num_target: -num_target]]
    other = np.random.choice(other, 2*num_target, replace=False).tolist()

    return other + high + low


if __name__ == '__main__':
    # main()
    multi_test(args)
