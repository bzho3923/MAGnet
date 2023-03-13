from torch_geometric.nn import GCNConv, GATConv,GINConv
from denoising_filters import *
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize
from deeprobust.graph.utils import *
from torch.nn import Linear
class ElasticGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, prop, **kwargs):
        super(ElasticGNN, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.prop = prop

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, x ,adj_t ,attention =False):
        # x, adj_t, = data.x, data.adj_t
        x = x.to(torch.float32)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, adj_t  )  # , data=data)
        return F.log_softmax(x, dim=1)

class GuardNet(nn.Module):
    def __init__(self, conv ,conv_type):
        super(GuardNet, self).__init__()
        self.GConv = conv
        self.reset_parameters()
        self.drop1 = nn.Dropout(0.5)
        self.conv_type = conv_type
        self.drop = False

    def att_coef(self, fea, edge_index, is_lil=False, i=0):
        edge_index = sparse_mx_to_torch_sparse_tensor(edge_index).to("cuda")
        # fea = torch.from_numpy(fea).to("cuda")
        if is_lil == False:
            edge_index = edge_index._indices()
        else:
            edge_index = edge_index.tocoo()
        n_node = fea.shape[0]
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]

        fea_copy = fea.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
        sim = sim_matrix[row, col]
        sim[sim <0.1] = 0
        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')
        """add learnable dropout, make character vector"""
        if self.drop:
            character = np.vstack((att_dense_norm[row, col].A1,
                                   att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T).to(self.device)
            drop_score = self.drop_learn_1(character)
            drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
            mm = torch.nn.Threshold(0.5, 0)
            drop_score = mm(drop_score)
            mm_2 = torch.nn.Threshold(-0.49, 1)
            drop_score = mm_2(-drop_score)
            drop_decision = drop_score.clone().requires_grad_()
            # print('rate of left edges', drop_decision.sum().data/drop_decision.shape[0])
            drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
            drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges

        if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
            degree = (att_dense_norm != 0).sum(1).A1
            lam = 1 / (degree + 1) # degree +1 is to add itself
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight  # add the self loop
        else:
            att = att_dense_norm

        row, col = att.nonzero(  )###select all nonzero item index
        att_adj = np.vstack((row, col))
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)   # exponent, kind of softmax
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32  )  # .cuda()
        att_adj = torch.tensor(att_adj, dtype=torch.int64  )  # .cuda()

        shape = (n_node, n_node)
        new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape)
        return new_adj

    def reset_parameters(self):
        for conv in self.GConv:
            conv.reset_parameters()

    def forward(self, data, structure ,attention=False):
        num_node = data.shape[0]
        # row, col = structure[0].cpu().numpy(), structure[1].cpu().numpy()
        # adj_value = np.ones_like(row)
        # adj = csr_matrix((adj_value, (row, col)), shape=(num_node, num_node))
        # # adj = torch.FloatTensor(adj.todense()).to(device="cuda")
        # #adj = csr_matrix(adj)
        # adj = adj + adj.T
        # adj[adj > 1] = 1
        adj = structure
        if attention ==True:
            # print("Using GNNGuard!!!")
            adj1 = self.att_coef(data, adj, i=0).to("cuda")
        else:
            adj1 = sparse_mx_to_torch_sparse_tensor(adj).to("cuda")
        data = data.to(torch.float32)
        # data = torch.from_numpy(data).to("cuda")
        x = self.GConv[0](data ,adj1._indices() ,edge_weight=adj1._values())
        if self.conv_type.lower() == 'gat':
            x = F.elu(x)
        elif self.conv_type.lower() == ('gcn' or 'ufg_s'):
            x = F.relu(x)
        if attention==True:  # if attention=True, use attention mechanism
            adj_2 = self.att_coef(x, adj, i=1)
            adj_memory = adj_2.to_dense()  # without memory
            # adj_memory = self.gate * adj.to_dense() + (1 - self.gate) * adj_2.to_dense()
            row, col = adj_memory.nonzero()[: ,0], adj_memory.nonzero()[: ,1]
            structure = torch.stack((row, col), dim=0)
            adj_values = adj_memory[row, col]
        x = self.drop1(x)
        x = self.GConv[1](x, structure.to("cuda") ,edge_weight=adj_values)
        return F.log_softmax(x, dim=1)

class Net(nn.Module):
    def __init__(self, conv ,conv_type):
        super(Net, self).__init__()
        self.GConv = conv
        self.reset_parameters()
        self.drop1 = nn.Dropout(0.5)
        self.conv_type = conv_type
        self.drop = False
    def reset_parameters(self):
        for conv in self.GConv:
            conv.reset_parameters()
    def forward(self, data, structure ,attention=False):
        data = data.to(torch.float32).cuda()
        structure = structure.cuda()
        x = self.GConv[0](data ,structure)
        if self.conv_type.lower() == 'gat':
            x = F.elu(x)
        elif self.conv_type.lower() == ('gcn' or 'ufg_s'):
            x = F.relu(x)
        x = self.drop1(x)
        x = self.GConv[1](x, structure)
        return F.log_softmax(x, dim=-1)


class MyGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4,
                 dropout=0.5):
        super(MyGCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False ,add_self_loops=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False))
        self.dropout_prob = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t ,attention=False):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.convs[-1](x ,adj_t)
        return x.log_softmax(dim=-1)

