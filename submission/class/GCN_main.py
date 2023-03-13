import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from scipy.sparse import coo_matrix, tril
from torch_geometric.utils import to_dense_adj, sort_edge_index
from torch_geometric.nn import GCNConv
import argparse
import os.path as osp
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

# function for pre-processing
@torch.no_grad()
def to_undirected_my(edge_index):
    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


# function for pre-processing
@torch.no_grad()
def scipy_to_torch_sparse(A):
    A = coo_matrix(A)
    row = torch.tensor(A.row)
    col = torch.tensor(A.col)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data)

    return torch.sparse_coo_tensor(index, value, A.shape)


def edge_attack(dataset,ratio):
    edge_old = dataset.edge_index # 10556
    n = dataset.num_nodes # 2708
    
    m = torch.zeros((n, n))
    m[edge_old[0],edge_old[1]]=1
    tril_idx = torch.tril_indices(n, n)

    num_nodes = dataset.num_nodes
    edge_index = dataset.edge_index
    edge = edge_index

    if ratio < 1:
        # prepare idx of the upper triangular matrix
        m[tril_idx[0],tril_idx[1]] = 0
        uptri_idx = m.to_sparse().indices()
        num_edge = round((1-ratio)*len(edge_old[0])/2) # 1056
        # random sample
        r,c = zip(*random.sample(list(zip(uptri_idx[0], uptri_idx[1])), num_edge))
        # prepare the new adj matrix (dense)
        m[torch.stack(r),torch.stack(c)]=0 # 4222
        sym_m = m + m.t()
        edge = sym_m.to_sparse().indices()
        
    elif ratio > 1:
        # convert to scipy sparse which is more powerful than torch sparse
        new_adj = 1 - to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        new_adj = coo_matrix(new_adj.numpy())
        lower_tri = tril(new_adj, k=-1)
        row = torch.from_numpy(lower_tri.row)
        col = torch.from_numpy(lower_tri.col)
        new_edge_index = torch.vstack((row, col))
        num_edge_add = round((ratio - 1) * len(edge_index[0]) / 2)
        mask = np.array([0] * len(row))
        mask[:num_edge_add] = 1
        np.random.shuffle(mask)
        mask = torch.from_numpy(mask).type(torch.bool)
        add_edge = new_edge_index[:, mask]
        add_edge = to_undirected_my(add_edge)
        edge_index = torch.cat((edge_index, add_edge), dim=1)
        edge = sort_edge_index(edge_index, num_nodes=num_nodes)[0]
        
    return edge.to(device)


class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, dropout_prob=0.5):
        super(Net, self).__init__()
        self.Conv1 = GCNConv(num_features, nhid)
        self.Conv2 = GCNConv(nhid, num_classes)
        self.drop1 = nn.Dropout(dropout_prob)

    def forward(self, x, edge_index):
        x = x.to(torch.float64)
        x = F.relu(self.Conv1(x, edge_index))
        x = self.drop1(x)
        x = self.Conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Citeseer',
                        help='name of dataset (default: Cora)')
    parser.add_argument('--reps', type=int, default=10,
                        help='number of repetitions (default: 10)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--wd', type=float, default=5e-3,
                        help='weight decay (default: 5e-3)')
    parser.add_argument('--nhid', type=int, default=64,
                        help='number of hidden units (default: 16)')
    parser.add_argument('--Lev', type=int, default=2,
                        help='level of transform (default: 2)')
    parser.add_argument('--s', type=float, default=2,
                        help='dilation scale > 1 (default: 2)')
    parser.add_argument('--n', type=int, default=2,
                        help='n - 1 = Degree of Chebyshev Polynomial Approximation (default: n = 2)')
    parser.add_argument('--FrameType', type=str, default='Haar',
                        help='frame type (default: Haar)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability (default: 0.5)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--attackEdgeRatio', type=float, default=1,
                        help='node ratio for attack (default: 5e-3)')
    parser.add_argument('--filename', type=str, default='results',
                        help='filename to store results and the model (default: results)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Training on CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    # load dataset
    dataname = args.dataset
    rootname = osp.join(osp.abspath(''), dataname)
    dataset = Planetoid(root=rootname, name=dataname)
    num_nodes = dataset[0].x.shape[0]

    '''
    Training Scheme
    '''

    # Hyper-parameter Settings
    learning_rate = args.lr
    weight_decay = args.wd
    nhid = args.nhid

    # extract the data
    data = dataset[0].to(device)
    edge_index_attack = edge_attack(dataset[0], args.attackEdgeRatio)
    #print("edge:",edge_index_attack,dataset[0].edge_index)

    # create result matrices
    num_epochs = args.epochs
    num_reps = args.reps
    epoch_loss = dict()
    epoch_acc = dict()
    epoch_loss['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['test_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['test_mask'] = np.zeros((num_reps, num_epochs))
    saved_model_val_acc = np.zeros(num_reps)
    saved_model_test_acc = np.zeros(num_reps)

    for rep in range(num_reps):
        print('****** Rep {}: training start ******'.format(rep + 1))
        max_acc = 0.0

        # initialize the model
        model = Net(dataset.num_node_features, nhid, dataset.num_classes, #r, Lev, num_nodes, shrinkage=None, threshold=1e-3,
                    dropout_prob=args.dropout).to(device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # initialize the learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        # training
        for epoch in range(num_epochs):
            # training mode
            model.train()
            optimizer.zero_grad()
            out = model(data.x, dataset[0].edge_index.to(device))
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # evaluation mode
            model.eval()
            out = model(data.x, dataset[0].edge_index.to(device))
            for i, mask in data('train_mask', 'val_mask', 'test_mask'):
                pred = out[mask].max(dim=1)[1]
                correct = float(pred.eq(data.y[mask]).sum().item())
                e_acc = correct / mask.sum().item()
                epoch_acc[i][rep, epoch] = e_acc
                e_loss = F.nll_loss(out[mask], data.y[mask])
                epoch_loss[i][rep, epoch] = e_loss

            # scheduler.step(epoch_loss['val_mask'][rep, epoch])

            # print out results
            # print('Epoch: {:3d}'.format(epoch + 1),
            #       'train_loss: {:.4f}'.format(epoch_loss['train_mask'][rep, epoch]),
            #       'train_acc: {:.4f}'.format(epoch_acc['train_mask'][rep, epoch]),
            #       'val_loss: {:.4f}'.format(epoch_loss['val_mask'][rep, epoch]),
            #       'val_acc: {:.4f}'.format(epoch_acc['val_mask'][rep, epoch]),
            #       'test_loss: {:.4f}'.format(epoch_loss['test_mask'][rep, epoch]),
            #       'test_acc: {:.4f}'.format(epoch_acc['test_mask'][rep, epoch]))

            # save model
            if epoch_acc['val_mask'][rep, epoch] > max_acc:
                torch.save(model.state_dict(), args.filename + '.pth')
                #print('=== Model saved at epoch: {:3d}'.format(epoch + 1))
                max_acc = epoch_acc['val_mask'][rep, epoch]
                record_test_acc = epoch_acc['test_mask'][rep, epoch]

        saved_model_val_acc[rep] = max_acc
        saved_model_test_acc[rep] = record_test_acc
        print('#### Rep {0:2d} Finished! val acc: {1:.4f}, test acc: {2:.4f} ####\n'.format(rep + 1, max_acc, record_test_acc))

    print('***************************************************************************************************************************')
    print('Average test accuracy over {0:2d} reps: {1:.4f} with stdev {2:.4f}'.format(num_reps, np.mean(saved_model_test_acc), np.std(saved_model_test_acc)))
    print('Avg. test acc. over top10 reps: {0:.4f} with stdev {1:.4f}\n'.format(np.mean(np.sort(saved_model_test_acc)[::-1][:10]),
                                                                                np.std(np.sort(saved_model_test_acc)[::-1][:10])))
    print('dataset:', args.dataset, '; epochs:', args.epochs, '; reps:', args.reps, '; learning_rate:', args.lr, '; weight_decay:', args.wd, '; nhid:', args.nhid,
          '; Lev:', args.Lev)
    print('s:', args.s, '; n:', args.n, '; FrameType:', args.FrameType, '; dropout:', args.dropout, '; seed:', args.seed, '; filename:', args.filename)
    print('\n')
    print(args.filename + '.pth', 'contains the saved model and ', args.filename + '.npz', 'contains all the values of loss and accuracy.')
    print('***************************************************************************************************************************')

    # save the results
    np.savez(args.filename + '.npz',
             epoch_train_loss=epoch_loss['train_mask'],
             epoch_train_acc=epoch_acc['train_mask'],
             epoch_valid_loss=epoch_loss['val_mask'],
             epoch_valid_acc=epoch_acc['val_mask'],
             epoch_test_loss=epoch_loss['test_mask'],
             epoch_test_acc=epoch_acc['test_mask'],
             saved_model_val_acc=saved_model_val_acc,
             saved_model_test_acc=saved_model_test_acc)


