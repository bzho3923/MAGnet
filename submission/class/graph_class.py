from scipy import sparse
from scipy.sparse.linalg import lobpcg
from torch_geometric.nn import GCNConv, GATConv,GINConv
from torch_geometric.datasets import Planetoid, WikiCS,Coauthor,AttributedGraphDataset,Reddit
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
from ogb.nodeproppred import PygNodePropPredDataset
from AirGNN_model import AirGNN
from baseline_model import *
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
import argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def graph_reader(edge_list):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(edge_list)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

def objective(args,learning_rate=0.01, weight_decay=0.01, nhid=64,epochs=100,dropout=0.5,dataname='ogbn-arxiv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rootname = osp.join("/home/jyh_temp1/Downloads/Graph_Anomaly/", dataname)
    if dataname.lower() == "blogcatalog":
        dataset = AttributedGraphDataset(root=rootname,name=dataname)
        print("blog:",dataset)
        num_class = dataset.num_classes
        data = dataset[0]
        clean_data = data.clone().to(device)
    if dataname.lower() == "reddit":
        dataset = Reddit(root=rootname)
        print("reddit:",dataset)
        num_class = dataset.num_classes
        data = dataset[0]
        clean_data = data.clone().to(device)
    if dataname.lower() == 'wikics':
        print("load wikics")
        dataset = WikiCS(root=rootname)
        num_class = dataset.num_classes
        data = dataset[0]
        clean_data = data.clone().to(device)
        clean_data['train_mask'] = clean_data['train_mask'][:, 0]
        clean_data['val_mask'] = clean_data['val_mask'][:, 0]
    if dataname.lower() == "ogbn-arxiv":
        NoTrans_dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='/home/jyh_temp1/Downloads/Graph_Anomaly/arxiv/',)
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='/home/jyh_temp1/Downloads/Graph_Anomaly/arxiv/', transform=T.ToSparseTensor())
        num_class = dataset.num_classes
        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"]   , split_idx["test"]
        train_mask = torch.zeros((dataset[0].x.shape[0],), dtype=torch.bool)
        train_mask[train_idx] = 1
        val_mask = torch.zeros((dataset[0].x.shape[0],), dtype=torch.bool)
        val_mask[val_idx] = 1
        test_mask = torch.zeros((dataset[0].x.shape[0],), dtype=torch.bool)
        test_mask[test_idx] = 1
        print("mask:", num_class,dataset[0].x.shape,train_mask.shape, train_mask,train_idx.shape,val_idx.shape,test_idx.shape)
        data = Data(x=dataset[0].x, edge_index=NoTrans_dataset[0].edge_index, y=dataset[0].y.squeeze(1), train_mask=train_mask,val_mask=val_mask, test_mask=test_mask)
        clean_data = data.clone().to(device)
    if dataname.lower() == 'cs':
        dataset = Coauthor(root="/home/jyh_temp1/Downloads/Graph_Anomaly/CS", name=dataname)
        num_class = dataset.num_classes
        train_mask = torch.zeros_like(dataset[0].y, dtype=torch.bool)
        train_mask[0:300] = 1
        print("train:", train_mask.shape, train_mask, dataset[0].y.shape)
        val_mask = torch.zeros_like(dataset[0].y, dtype=torch.bool)
        val_mask[300:500] = 1
        test_mask = torch.zeros_like(dataset[0].y, dtype=torch.bool)
        test_mask[500:1500] = 1
        data = Data(x=dataset[0].x, edge_index=dataset[0].edge_index, y=dataset[0].y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        clean_data = data.clone().to(device)
    if dataname.lower() == 'cora' or dataname.lower() == 'pubmed' or dataname.lower() == 'citeseer':
        print("load Planet data")
        dataset = Planetoid(root=rootname, name=dataname)
        num_class = dataset.num_classes
        data = dataset[0]
        clean_data = data.clone().to(device)
        cora_linegraph = LineGraph(force_directed=False)(data)
        reverse_cora = LineGraph(force_directed=False)(cora_linegraph)
        # print("line graph:",data.edge_attr,cora_linegraph,"\n",reverse_cora)#,cora_linegraph.x.shape,cora_linegraph.edge_index.shape,cora_linegraph.edge_atrr)
    if dataname.lower() == 'wisconsin' or dataname.lower() == 'texas':
        dataset = HetroDataSet(root=rootname, name=dataname)
        num_class = dataset.num_classes
        data = dataset[0]
        clean_data = data.clone().to(device)
        print("clean mask:", torch.sum(clean_data.train_mask.int()), torch.sum(clean_data.val_mask.int()),
              torch.sum(clean_data.test_mask.int()))
    #######
    num_features = data.x.shape[1]
    num_nodes = data.x.shape[0]
    edges = clean_data.edge_index
    if dataname.lower() == "ogbn-arxiv":
        edges = dataset[0].adj_t.to_symmetric().to(device)
        # data.edge_index = edges
    ##UFG elements

    if 'ufg' in args.GConv_type.lower():
        L = get_laplacian(data.edge_index, num_nodes=num_nodes, normalization='sym')
        L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))

        # get maximum eigenvalues of the graph Laplacian
        lobpcg_init = np.random.rand(num_nodes, 1)
        lambda_max, _ = lobpcg(L, lobpcg_init)
        lambda_max = lambda_max[0]

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
        gcn_model = nn.ModuleList([GCNConv(num_features, nhid, add_self_loops=True),
                                   GCNConv(nhid, num_class, add_self_loops=True)]).to(device)
    elif args.GConv_type.lower() == 'ufg_s':
        gcn_model = nn.ModuleList(
            [UFGConv_S(num_features, nhid, r, args.Lev, num_nodes, shrinkage=args.shrinkage, sigma=args.sigma),
             UFGConv_S(nhid, num_class, r, args.Lev, num_nodes, shrinkage=args.shrinkage,
                       sigma=args.sigma)]).to(device)
    elif args.GConv_type.lower() == 'ufg_r':
        gcn_model = nn.ModuleList([UFGConv_R(num_features, nhid, r, args.Lev, num_nodes),UFGConv_R(nhid, num_class, r, args.Lev, num_nodes)])
    else:
        raise Exception('invalid type of graph convolution')


    if args.plaingcn == 1:

        if dataname.lower() == "ogbn-arxiv":
            model = MyGCN(in_channels=num_features,hidden_channels=nhid,out_channels=num_class,num_layers=3).to(device)
        else:
            model = Net(gcn_model, args.GConv_type).to(device)
    if args.airgnn == 1:
        model = AirGNN(dataset, args).to(device)
    if args.guardmodel == 1:
        print("Using GUARD model!")
        ###defining the model
        if dataname.lower()=="ogbn-arxiv":
        #     pass
        # else:
            row, col = data.edge_index[0], data.edge_index[1]
            adj_value = np.ones_like(row)
            print("shape:",row)
            adj = csr_matrix((adj_value, (row, col)), shape=(num_nodes, num_nodes))
            features = data.x
            # adj = data.edge_index
            labels = data.y
            if scipy.sparse.issparse(features) == False:
                features = scipy.sparse.csr_matrix(features)  ##make features sparse
            adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
            # 1. to CSR sparse
            # print("adj type:", type(adj), type(csr_matrix(adj)))
            adj, features = csr_matrix(adj), csr_matrix(features)
            adj = adj + adj.T
            adj[adj > 1] = 1
            edges = adj
        model = GuardNet(gcn_model, args.GConv_type).to(device)
    if args.elasticmodel == 1:
        print("Using ELASTIC model!")
        ###torch_sparse tensor transform
        if dataname.lower()=="ogbn-arxiv":
            pass# edges = edges.to(device)
        else:
            transform = T.ToSparseTensor()
            dataset.transform = transform
            if not isinstance(dataset[0].adj_t, torch.Tensor):
                dataset[0].adj_t = dataset[0].adj_t.to_symmetric()
            edges = dataset[0].adj_t.to(device)
        prop = EMP(K=10, lambda1=9, lambda2=3, L21=True, cached=True, normalize=True)

        model = ElasticGNN(in_channels=data.num_features,
                           hidden_channels=nhid,
                           out_channels=dataset.num_classes,
                           dropout=args.dropout,
                           num_layers=args.num_layers,
                           prop=prop).to(device)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    # training
    if args.datatype.lower() == "noisy":
        # noise_data = np.load("./0.53PubMed_"+str(args.datanum)+"mettack_feat.npy")
        noise_data = np.load(
            "/home/jyh_temp1/Downloads/Graph_Anomaly/" + str(dataname.lower()) + "_injection/" + str(dataname.lower()) + "noise_mat_ratiio" + str(
                0.5) + ".npy")
        if args.mettackdata==1:
            noise_data = np.load("/home/jyh_temp1/Downloads/Graph_Anomaly/Mettack_data/"+str(dataname.lower())+"/"+ str(
                    dataname.lower()) + "mettack_feat" + ".npy")
        noise_data = torch.from_numpy(noise_data).to(device)
    if args.datatype == "denoised":
        if args.local == 1:
            if args.gtmask == 0:
                print("Using local mask")
                # denoised_data = np.load("./newtest/denoised_featmat" + str(args.datanum) + "_lp0lq1.npy")
                args.ratio = 0.5
                denoised_data = np.load(
                    "/home/jyh_temp1/Downloads/Graph_Anomaly/" + str(dataname.lower()) + "_injection/gamma500.0thres25.0prob0.25gae1w_gaepower" + str(args.power) + "denoised_featmat" + str(
                        args.ratio) + "_lp" + str(args.lp) + "lq" + str(args.lq) + ".npy")
            if args.gtmask == 1:
                print("Using ground truth mask")
                # denoised_data = np.load("./newtest/gtmaskdenoised_featmat" + str(args.datanum) + "_lp0lq1.npy")
                denoised_data = np.load("/home/jyh_temp1/Downloads/Graph_Anomaly/" + str(dataname.lower()) + "_injection/gamma500.0thres150.0prob0.2gae1w_" + "gtmaskdenoised_featmat" + str(
                    args.ratio) + "_lp" + str(args.lp) + "lq" + str(args.lq) + ".npy")
        if args.local == 0:
            print("uisng global denoised data")
            denoised_data = np.load("/home/jyh_temp1/Downloads/Graph_Anomaly/" + str(
                dataname.lower()) + "_injection/" + "globaldenoised_featmat" + str(
                args.ratio) + "_lp" + str(args.lp) + "lq" + str(args.lq) + ".npy")
            # denoised_data = np.load("./newtest/globaldenoised_featmat" + str(args.datanum) + "_lp1lq2.npy")args.lp) + "lq" + str(args.lq) + ".npy")
        denoised_data = torch.from_numpy(denoised_data).to(device)

    if args.datatype == "clean":
        data = clean_data.x
    if args.datatype == "noisy":
        if args.airgnn==1:
            print("airgnn dataset")
            # from AirGNN_dataset import prepare_data
            # dataset, permute_masks = prepare_data(args, lcc=args.lcc)
            transform = T.ToSparseTensor()
            dataset.transform = transform
            if not isinstance(dataset[0].adj_t, torch.Tensor):
                data.adj_t = dataset[0].adj_t.to_symmetric().to(device)
            data.x = noise_data
            # data.adj_t = clean_data.edge_index
        else:
            data = noise_data
    if args.datatype == "denoised":
        data = denoised_data

    # create result matrices
    num_epochs = epochs
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
    acc_list = []

    for rep in range(num_reps):
        # print('****** Rep {}: training start ******'.format(rep + 1))
        max_acc = 0.0
        if args.appnpmodel == 1:
            print("using APPNP model!")
            origin_edges = clean_data.edge_index.tolist()
            edges = [[node1, node2] for node1, node2 in zip(origin_edges[0], origin_edges[1])]
            for i in range(len(origin_edges[0])):
                node1 = origin_edges[0][i]
                node2 = origin_edges[1][i]
                edges.append([node1, node2])
            graph = graph_reader(edges)
            index_1 = [node for node in graph.nodes() for fet in torch.nonzero(data[node])]
            index_2 = [fet for node in graph.nodes() for fet in torch.nonzero(data[node])]
            values = [1.0 / len(torch.nonzero(data[node])) for node in graph.nodes() for fet in
                      torch.nonzero(data[node])]
            feature_indices = torch.LongTensor([index_1, index_2]).to(device)
            feature_values = torch.FloatTensor(values).to(device)
            model = APPNPModel(number_of_labels=num_class, number_of_features=num_features, graph=graph,
                               device=device).to(device)
        # reset the model parameters
        if args.appnpmodel == 0:
            model.reset_parameters()

        # initialize the optimizer
        start = time.time()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=5, verbose=False,min_lr=0.000000001)
        # training
        for epoch in range(num_epochs):
            # training mode
            model.train()
            optimizer.zero_grad()
            if 'ufg' in args.GConv_type.lower():
                out = model(data, d_list)
            else:
                if args.appnpmodel == 1:
                    out = model(feature_indices, feature_values)
                if args.airgnn == 1:
                    out = model(data)
                else:
                    out = model(data, edges, attention=args.attention)
            # print("show::",out[clean_data.train_mask][0:20],clean_data.y[clean_data.train_mask][0:20],torch.max(clean_data.y),torch.min(clean_data.y))
            loss = F.nll_loss(out[clean_data.train_mask], clean_data.y[clean_data.train_mask])
            if args.appnpmodel == 1:
                loss = loss + (0.005 / 2) * (torch.sum(model.layer_2.weight_matrix ** 2))
            loss.backward()
            optimizer.step()
            # scheduler.step(loss)

            # evaluation mode
            with torch.no_grad():
                model.eval()
                if 'ufg' in args.GConv_type.lower():
                    out = model(data, d_list)
                else:
                    if args.appnpmodel == 1:
                        out = model(feature_indices, feature_values)
                    if args.airgnn==1:
                        out = model(data)
                    else:
                        out = model(data, edges, attention=args.attention)
                for i, mask in clean_data('train_mask', 'val_mask', 'test_mask'):
                    pred =  out[mask].max(dim=1)[1]
                    correct = float(pred.eq(clean_data.y[mask]).sum().item())
                    e_acc = correct / mask.sum().item()
                    epoch_acc[i][rep, epoch] = e_acc
                    e_loss = F.nll_loss(out[mask], clean_data.y[mask])
                    epoch_loss[i][rep, epoch] = e_loss

            # save model
            if epoch_acc['val_mask'][rep, epoch] > max_acc:
                torch.save(model.state_dict(), args.filename + '.pth')
                # print('=== Model saved at epoch: {:3d}'.format(epoch + 1))
                max_acc = epoch_acc['val_mask'][rep, epoch]
                record_test_acc = epoch_acc['test_mask'][rep, epoch]

        saved_model_val_acc[rep] = max_acc
        saved_model_test_acc[rep] = record_test_acc
        acc_list.append(record_test_acc)
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
    return np.mean(saved_model_test_acc)
def ray_train(args):
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    class ExperimentTerminationReporter(CLIReporter):
        def should_report(self, trials, done=True):
            """Reports only on experiment termination."""
            return done

    def training_function(config):
        acc = objective(args,**config)
        tune.report(acc=acc)

    ray.shutdown()
    ray.init(num_cpus=8, num_gpus=4)

    asha_scheduler = ASHAScheduler(
        # time_attr='training_iteration',
        metric='loss',
        mode='min',
        max_t=200,
        grace_period=100,
        reduction_factor=2,
        brackets=1)

    analysis = tune.run(
        training_function,
        config={
            "dataname": tune.grid_search(["Citeseer"]),#"cora","PubMed","Citeseer""CS","wikics","ogbn-arxiv","wisconsin","texas"]
            "learning_rate": tune.grid_search([1e-2,1e-3]),
            "weight_decay": tune.grid_search([1e-4]),
            "epochs": tune.grid_search([200,800]),
            "nhid": tune.grid_search([256]),
            "dropout": tune.grid_search([0.1,0.5]),
        },
        progress_reporter=ExperimentTerminationReporter(),
        resources_per_trial={'gpu': 1, 'cpu': 2},
        # scheduler=asha_scheduler
    )

    print("Best config: ", analysis.get_best_config(
        metric="acc", mode="max"))

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    return df
def arg_train(args):
    df = objective(args,learning_rate=args.lr, weight_decay=args.wd, nhid=args.nhid,epochs=args.epochs, dataname=args.dataset)

if __name__ == '__main__':
    # get config
    parser.add_argument("action", type=str, default=ray_train, help="ray or arg train")
    parser.add_argument('--dataset', type=str, default='wisconsin',
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
    parser.add_argument('--airgnn', type=int, default=0,
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
    parser.add_argument('--mettackdata', type=int, default=0,
                        help='use mettack noise data')
    ###airgnn args
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--lambda_amp', type=float, default=0.1)
    parser.add_argument('--lcc', type=str2bool, default=False)
    parser.add_argument('--normalize_features', type=str2bool, default=True)
    parser.add_argument('--random_splits', type=str2bool, default=False)
    parser.add_argument('--hidden', type=int, default=64)
    # parser.add_argument('--dropout', type=float, default=0.8, help="dropout")
    parser.add_argument('--K', type=int, default=10, help="the number of propagagtion in AirGNN")
    parser.add_argument('--model_cache', type=str2bool, default=False)
    args = parser.parse_args()
    if args.action=="arg_train":
        arg_train(args)
    if args.action=="ray_train":
        ray_train(args)