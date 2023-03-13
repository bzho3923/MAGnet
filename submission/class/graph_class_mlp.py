from scipy import sparse
from scipy.sparse.linalg import lobpcg
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid, WikiCS,Coauthor
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, degree
from ufg_layer import UFGConv_S, UFGConv_R
from ufgconfig import parser
from denoising_filters import *
from ufgutils import scipy_to_torch_sparse, get_operator
import os.path as osp
import random
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
import scipy
from deeprobust.graph.utils import *
from torch.nn import Linear
from emp import EMP
import torch_geometric.transforms as T
from torch_geometric.transforms import LineGraph
from appnp_layer import APPNPModel
import networkx as nx
from torch_geometric.data import Data
from new_dataset import HetroDataSet
import time
from torch.nn import Sequential as Seq, Linear, ReLU, GELU, ELU, Dropout
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
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

    def forward(self, x,adj_t,attention =False):
        # x, adj_t, = data.x, data.adj_t
        x = x.to(torch.float32)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, adj_t, data=data)
        return F.log_softmax(x, dim=1)
class GuardNet(nn.Module):
    def __init__(self, conv,conv_type):
        super(GuardNet, self).__init__()
        self.GConv = conv
        self.reset_parameters()
        self.drop1 = nn.Dropout(0.5)
        self.conv_type = conv_type
        self.drop = False

    def att_coef(self, fea, edge_index, is_lil=False, i=0):
        edge_index = sparse_mx_to_torch_sparse_tensor(edge_index).to("cuda")
        #fea = torch.from_numpy(fea).to("cuda")
        if is_lil == False:
            edge_index = edge_index._indices()
        else:
            edge_index = edge_index.tocoo()
        n_node = fea.shape[0]
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]

        fea_copy = fea.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
        sim = sim_matrix[row, col]
        sim[sim<0.1] = 0
        # print('dropped {} edges'.format(1-sim.nonzero()[0].shape[0]/len(sim)))

        # """use jaccard for binary features and cosine for numeric features"""
        # fea_start, fea_end = fea[edge_index[0]], fea[edge_index[1]]
        # isbinray = np.array_equal(fea_copy, fea_copy.astype(bool))  # check is the fea are binary
        # np.seterr(divide='ignore', invalid='ignore')
        # if isbinray:
        #     fea_start, fea_end = fea_start.T, fea_end.T
        #     sim = jaccard_score(fea_start, fea_end, average=None)  # similarity scores of each edge
        # else:
        #     fea_copy[np.isinf(fea_copy)] = 0
        #     fea_copy[np.isnan(fea_copy)] = 0
        #     sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
        #     sim = sim_matrix[edge_index[0], edge_index[1]]
        #     sim[sim < 0.01] = 0

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

        row, col = att.nonzero()###select all nonzero item index
        att_adj = np.vstack((row, col))
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)   # exponent, kind of softmax
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32)#.cuda()
        att_adj = torch.tensor(att_adj, dtype=torch.int64)#.cuda()

        shape = (n_node, n_node)
        new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape)
        return new_adj

    def reset_parameters(self):
        for conv in self.GConv:
            conv.reset_parameters()

    def forward(self, data, structure,attention=False):
        num_node = data.shape[0]
        row, col = structure[0].cpu().numpy(), structure[1].cpu().numpy()
        adj_value = np.ones_like(row)
        adj = csr_matrix((adj_value, (row, col)), shape=(num_node, num_node))
        # adj = torch.FloatTensor(adj.todense()).to(device="cuda")
        #adj = csr_matrix(adj)
        adj = adj + adj.T
        adj[adj > 1] = 1
        # adj = structure
        if attention==True:
            #print("Using GNNGuard!!!")
            adj1 = self.att_coef(data, adj, i=0).to("cuda")
        else:
            adj1 = sparse_mx_to_torch_sparse_tensor(adj).to("cuda")
        data = data.to(torch.float32)
        #data = torch.from_numpy(data).to("cuda")
        x = self.GConv[0](data,adj1._indices(),edge_weight=adj1._values())
        if self.conv_type.lower() == 'gat':
            x = F.elu(x)
        elif self.conv_type.lower() == ('gcn' or 'ufg_s'):
            x = F.relu(x)
        if attention==True:  # if attention=True, use attention mechanism
            adj_2 = self.att_coef(x, adj, i=1)
            adj_memory = adj_2.to_dense()  # without memory
            # adj_memory = self.gate * adj.to_dense() + (1 - self.gate) * adj_2.to_dense()
            row, col = adj_memory.nonzero()[:,0], adj_memory.nonzero()[:,1]
            structure = torch.stack((row, col), dim=0)
            adj_values = adj_memory[row, col]
        x = self.drop1(x)
        x = self.GConv[1](x, structure.to("cuda"),edge_weight=adj_values.to("cuda"))
        return F.log_softmax(x, dim=1)
class Net(nn.Module):
    def __init__(self, conv,conv_type):
        super(Net, self).__init__()
        self.GConv = conv
        self.reset_parameters()
        self.drop1 = nn.Dropout(0.5)
        self.conv_type = conv_type
        self.drop = False
        out_channels = 512
        class_num = 7
        self.mlp1 = Seq(
            ELU(),
            Dropout(0.1),
            Linear(1433, out_channels),)
        self.mlp2 = Seq(
            ELU(),
            Dropout(0.1),
            Linear(out_channels, class_num), )

    def reset_parameters(self):
        for conv in self.GConv:
            conv.reset_parameters()

    def forward(self, data, structure,attention=False):
        data = data.to(torch.float32).cuda()
        x =self.mlp1(data)
        x = self.mlp2(x)
        return F.log_softmax(x, dim=-1)


def graph_reader(edge_list):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(edge_list)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


if __name__ == '__main__':
    # get config
    parser.add_argument('--dataset', type=str, default='cora',
                        help='name of dataset with choices "Cora", "Citeseer", "Wikics"')
    parser.add_argument('--datatype', type=str, default='clean',
                        help='data type with choices "clean", "noisy", "denoised"')
    parser.add_argument('--datanum', type=int, default=1350,
                        help='data number with anomal nodes 150,270,500/40,150,200,540 for adv')
    parser.add_argument('--local', type=int, default=1,
                        help='global/local denoised data')
    parser.add_argument('--gtmask', type=int, default=0,
                        help='using groundtruth mask mat')
    parser.add_argument('--lp', type=int, default=0,
                        help='first term using lp norm')
    parser.add_argument('--lq', type=int, default=2,
                        help='second term using lq norm')
    parser.add_argument('--attention', type=int, default=0,
                        help='GNNGuard')
    parser.add_argument('--guardmodel', type=int, default=0,
                        help='GNNGuard')
    parser.add_argument('--elasticmodel', type=int, default=0,
                        help='elastic')
    parser.add_argument('--appnpmodel', type=int, default=0,
                        help='appnp')
    parser.add_argument('--plaingcn', type=int, default=1,
                        help='gcn')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='data anomal ratio')
    parser.add_argument('--power', type=int, default=1,
                        help='gae using abs1 or mse2')
    args = parser.parse_args()

    # set random seed for reproducible results
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # training on CPU/GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    # load dataset
    dataname = args.dataset
    rootname = osp.join('./', dataname)
    if dataname.lower() == 'wikics':
        print("load wikics")
        dataset = WikiCS(root=rootname)
        num_class = dataset.num_classes
        data = dataset[0]
        clean_data = data.clone().to(device)
    if dataname.lower() == 'cs':
        dataset = Coauthor(root="./CS",name = dataname)
        num_class = dataset.num_classes
        train_mask = torch.zeros_like(dataset[0].y,dtype=torch.bool)
        train_mask[0:300] = 1
        val_mask = torch.zeros_like(dataset[0].y, dtype=torch.bool)
        val_mask[300:500] = 1
        test_mask = torch.zeros_like(dataset[0].y, dtype=torch.bool)
        test_mask[500:1500]=1
        data = Data(x=dataset[0].x,edge_index=dataset[0].edge_index,y=dataset[0].y,train_mask=train_mask,val_mask = val_mask,test_mask=test_mask)
        clean_data = data.clone().to(device)
    if dataname.lower()=="ogbn-arxiv":
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/')
        data = dataset[0]
    if dataname.lower() == 'cora' or dataname.lower() == 'pubmed' or dataname.lower() == 'citeseer':
        print("load Planet data")
        dataset = Planetoid(root=rootname, name=dataname)
        num_class = dataset.num_classes
        print("num class:",num_class,"shape:",dataset[0].x.shape)
        data = dataset[0]
        clean_data = data.clone().to(device)
        cora_linegraph = LineGraph(force_directed=False)(data)
        reverse_cora = LineGraph(force_directed=False)(cora_linegraph)
        # print("line graph:",data.edge_attr,cora_linegraph,"\n",reverse_cora)#,cora_linegraph.x.shape,cora_linegraph.edge_index.shape,cora_linegraph.edge_atrr)
    if dataname.lower() == 'wisconsin' or dataname.lower() == 'texas':
        dataset = HetroDataSet(root=rootname,name = dataname)
        num_class = dataset.num_classes
        data = dataset[0]
        clean_data = data.clone().to(device)

    #######
    #print(str(args.dataset)+"min max:", data.x.min(), data.x.max(), data.x[0, 0:50])
    num_features = data.x.shape[1]
    num_nodes = data.x.shape[0]
    #print("dataset setting:",num_nodes,num_features,(data.x).max(),(data.x).min(),data.x[0].tolist())
    ##UFG elements

    # row, col = data.edge_index[0], data.edge_index[1]
    # adj_value = np.ones_like(row)
    # adj = csr_matrix((adj_value, (row, col)), shape=(num_nodes, num_nodes))
    # features = data.x
    # # adj = data.edge_index
    # labels = data.y
    # if scipy.sparse.issparse(features) == False:
    #     features = scipy.sparse.csr_matrix(features)  ##make features sparse
    # adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
    # # 1. to CSR sparse
    # # print("adj type:", type(adj), type(csr_matrix(adj)))
    # adj, features = csr_matrix(adj), csr_matrix(features)
    # adj = adj + adj.T
    # adj[adj > 1] = 1


    if 'ufg' in args.GConv_type.lower():
        L = get_laplacian(data.edge_index, num_nodes=num_nodes, normalization='sym')
        L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))

        # get maximum eigenvalues of the graph Laplacian
        # lobpcg_init = np.random.rand(num_nodes, 1)
        # lambda_max, _ = lobpcg(L, lobpcg_init)
        # lambda_max = lambda_max[0]

        lambda_max = largest_eigsh(L,k=1,which="LM")

        # get degrees
        deg = degree(data.edge_index[0], num_nodes).to(device)

        # extract decomposition/reconstruction Masks
        D1 = lambda x: np.cos(x / 2)
        D2 = lambda x: np.sin(x / 2)
        DFilters = [D1, D2]
        RFilters = [D1, D2]

        # get matrix operators
        J = np.log(lambda_max / np.pi) / np.log(args.s) + args.Lev - 1  # dilation level to start the decomposition
        d = get_operator(L, DFilters, args.n, args.s, J, args.Lev)

        # store the matrix operators (torch sparse format) into a list: row-by-row
        r = len(DFilters)
        d_list = list()
        for i in range(r):
            for l in range(args.Lev):
                d_list.append(scipy_to_torch_sparse(d[i, l]).to(device))
    if args.GConv_type.lower() == 'gcn':
        gcn_model = nn.ModuleList([GCNConv(num_features, args.nhid, add_self_loops=True),GCNConv(args.nhid, num_class, add_self_loops=True)]).to(device)
    elif args.GConv_type.lower() == 'gat':
        gcn_model = nn.ModuleList([GATConv(num_features, args.nhid), GATConv(args.nhid, num_class)]).to(device)
    elif args.GConv_type.lower() == 'ufg_s':
        gcn_model = nn.ModuleList(
            [UFGConv_S(num_features, args.nhid, r, args.Lev, num_nodes, shrinkage=args.shrinkage, sigma=args.sigma),
             UFGConv_S(args.nhid, num_class, r, args.Lev, num_nodes, shrinkage=args.shrinkage,
                       sigma=args.sigma)]).to(device)
    elif args.GConv_type.lower() == 'ufg_r':
        gcn_model = nn.ModuleList([UFGConv_R(num_features, args.nhid, r, args.Lev, num_nodes),
                                   UFGConv_R(args.nhid, num_class, r, args.Lev, num_nodes)])
    else:
        raise Exception('invalid type of graph convolution')
    ###defining the model
    if args.plaingcn == 1:
        edges = data.edge_index
        model = Net(gcn_model, args.GConv_type).to(device)
    if args.guardmodel==1:
        edges = adj
        model = GuardNet(gcn_model,args.GConv_type).to(device)
    if args.elasticmodel==1:
        ###torch_sparse tensor transform
        transform = T.ToSparseTensor()
        dataset.transform = transform
        if not isinstance(dataset[0].adj_t, torch.Tensor):
            dataset[0].adj_t = dataset[0].adj_t.to_symmetric()
        edges = dataset[0].adj_t.to(device)
        prop = EMP(K=10,lambda1=9,lambda2=3,L21=True,cached=True,normalize=True)

        model = ElasticGNN(in_channels=data.num_features,
                      hidden_channels=args.nhid,
                      out_channels=dataset.num_classes,
                      dropout=args.dropout,
                      num_layers=args.num_layers,
                      prop=prop).to(device)



    learning_rate = args.lr
    weight_decay = args.wd
    nhid = args.nhid
    # initialize the optimizer


    # initialize the learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # training
    if args.elasticmodel==0:
        edges = clean_data.edge_index
    if dataname.lower() == 'wikics':
        clean_data['train_mask'] = clean_data['train_mask'][:, 0]
        clean_data['val_mask'] = clean_data['val_mask'][:, 0]
    if args.datatype.lower() == "noisy":
        # noise_data = np.load("./0.53PubMed_"+str(args.datanum)+"mettack_feat.npy")
        noise_data = np.load("./" + str(dataname.lower()) + "_injection/" + str(dataname.lower()) + "noise_mat_ratiio" + str(0.5) + ".npy")
        noise_data = torch.from_numpy(noise_data).to(device)
    if args.datatype == "denoised":
        if args.local == 1:
            if args.gtmask == 0:
                print("Using local mask")
                # denoised_data = np.load("./newtest/denoised_featmat" + str(args.datanum) + "_lp0lq1.npy")
                args.ratio=0.5
                denoised_data = np.load("./" + str(dataname.lower()) + "_injection/gaepower" + str(args.power) + "denoised_featmat" + str(args.ratio) + "_lp" + str(args.lp) + "lq" + str(args.lq) + ".npy")
            if args.gtmask == 1:
                print("Using ground truth mask")
                # denoised_data = np.load("./newtest/gtmaskdenoised_featmat" + str(args.datanum) + "_lp0lq1.npy")
                denoised_data = np.load("./" + str(dataname.lower()) + "_injection/" + "gtmaskdenoised_featmat" + str(args.ratio) + "_lp" + str(args.lp) + "lq" + str(args.lq) + ".npy")
        if args.local == 0:
            print("uisng global denoised data")
            # denoised_data = np.load("./newtest/globaldenoised_featmat" + str(args.datanum) + "_lp1lq2.npy")args.lp) + "lq" + str(args.lq) + ".npy")
        denoised_data = torch.from_numpy(denoised_data).to(device)

    num_epochs = args.epochs
    if args.datatype == "clean":
        data = clean_data.x
    if args.datatype == "noisy":
        data = noise_data
    if args.datatype == "denoised":
        data = denoised_data


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
    acc_list= []

    for rep in range(num_reps):
        #print('****** Rep {}: training start ******'.format(rep + 1))
        max_acc = 0.0
        if args.appnpmodel == 1:
            origin_edges = clean_data.edge_index.tolist()
            edges = [[node1, node2] for node1, node2 in zip(origin_edges[0], origin_edges[1])]
            for i in range(len(origin_edges[0])):
                node1 = origin_edges[0][i]
                node2 = origin_edges[1][i]
                edges.append([node1, node2])
            graph = graph_reader(edges)
            index_1 = [node for node in graph.nodes() for fet in torch.nonzero(data[node])]
            index_2 = [fet for node in graph.nodes() for fet in torch.nonzero(data[node])]
            values = [1.0 / len(torch.nonzero(data[node])) for node in graph.nodes() for fet in torch.nonzero(data[node])]
            feature_indices = torch.LongTensor([index_1, index_2]).to(device)
            feature_values = torch.FloatTensor(values).to(device)
            model = APPNPModel(number_of_labels=num_class, number_of_features=num_features, graph=graph,device=device).to(device)
        # reset the model parameters
        if args.appnpmodel == 0:
            model.reset_parameters()

        # initialize the optimizer
        start = time.time()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=5, verbose=False,min_lr=0.000000001)
        # training
        for epoch in range(num_epochs):
            # training mode
            model.train()
            optimizer.zero_grad()
            if 'ufg' in args.GConv_type.lower():
                out = model(data, d_list)
            else:
                if args.appnpmodel == 1:
                    out = model(feature_indices,feature_values)
                else:
                    out = model(data, edges,attention=args.attention)
            loss = F.nll_loss(out[clean_data.train_mask], clean_data.y[clean_data.train_mask])
            if args.appnpmodel==1:
                loss = loss +(0.005/2)*(torch.sum(model.layer_2.weight_matrix**2))
            #print("train loss:",loss.item())
            loss.backward()
            optimizer.step()
            #scheduler.step(loss)

            # evaluation mode
            with torch.no_grad():
                model.eval()
                if 'ufg' in args.GConv_type.lower():
                    out = model(data, d_list)
                else:
                    if args.appnpmodel==1:
                        out = model(feature_indices, feature_values)
                    else:
                        out = model(data, edges,attention=args.attention)
                for i, mask in clean_data('train_mask', 'val_mask', 'test_mask'):
                    pred = out[mask].max(dim=1)[1]
                    correct = float(pred.eq(clean_data.y[mask]).sum().item())
                    e_acc = correct / mask.sum().item()
                    epoch_acc[i][rep, epoch] = e_acc
                    e_loss = F.nll_loss(out[mask], clean_data.y[mask])
                    epoch_loss[i][rep, epoch] = e_loss

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
        acc_list.append(record_test_acc)
        #print("time cost:",time.time()-start)
        #print('#### Rep {0:2d} Finished! val acc: {1:.4f}, test acc: {2:.4f} ####\n'.format(rep + 1, max_acc,record_test_acc))
    print(
        '***************************************************************************************************************************')
    print('Average test accuracy over {0:2d} reps: {1:.4f} with stdev {2:.4f}'.format(num_reps,
                                                                                      np.mean(saved_model_test_acc),
                                                                                      np.std(saved_model_test_acc)))
    print('\n')
    print(args)
    print(args.filename + '.pth', 'contains the saved model and ', args.filename + '.npz',
          'contains all the values of loss and accuracy.')
    print(
        '***************************************************************************************************************************')