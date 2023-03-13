import numpy as np
from scipy.sparse.linalg import lobpcg
from torch_geometric.utils import get_laplacian, degree
from jw_denoising_filters import *
from utils import scipy_to_torch_sparse, get_operator
from config import parser
import random
import os.path as osp
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
import time
from torch_geometric.data import Data
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from ogb.nodeproppred import PygNodePropPredDataset
import pickle
if __name__ == '__main__':
    # get config
    parser.add_argument('--dataset', type=str, default='cora',
                        help='name of dataset with choices "Cora", "Citeseer", "Wikics"')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='data number with anomal nodes 150,270,500')
    parser.add_argument('--mask', type=int, default=1,
                        help='using mask for local denoising 1/without mask for global denoising 0')
    parser.add_argument('--power', type=int, default=2,
                        help='abs 1 or mse 2')
    parser.add_argument('--gtmask', type=int, default=0,
                        help='using groundtruth mask mat')
    parser.add_argument('--lp', type=int, default=1,
                        help='first term using lp norm')
    parser.add_argument('--lq', type=int, default=1,
                        help='second term using lq norm')
    parser.add_argument('--boost', type=int, default=0,
                        help='using accelerated scheme or not')
    parser.add_argument('--boost_value', type=float, default=0.3,
                        help='boost alue')
    parser.add_argument('--stop_thres', type=float, default= 3000000,
                        help='stopping criteria to stop the ADMM')
    parser.add_argument('--mu2_0', type=float, default=10, ###gmma 10
                        help='initial value of mu2')
    parser.add_argument('--anomal_conf_prob', type=float, default=0.2,##0.001
                        help='boost alue')
    parser.add_argument('--thres_iter', type=float, default=100,
                        help='boost alue')
    parser.add_argument('--nu', type=float, default=500,
                        help='tight wavelet frame transform tuning parameter')
    parser.add_argument('--gae_epoch', type=int, default=20000,
                        help='gae training epoch')
    args = parser.parse_args()
    if args.filter_type.lower() == 'dot':
        args.filter_type = 'Breg'
    if args.dataset=="cora":
        print("dataset:",args.dataset)
        torch.set_default_dtype(torch.float32)
        torch.set_default_tensor_type(torch.FloatTensor)
    # set random seed for reproducible results
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # training on CPU/GPU device
    device = torch.device("cuda")#"cuda" if torch.cuda.is_available() else "cpu")
    print("device:",device,"dataset:",args.dataset)
    # load dataset
    ratio_list=[0.5]#[0.05,0.1,0.3,0.5,0.75]
    for ratio in ratio_list:
        dataname = args.dataset
        rootname = osp.join('./', dataname)
        if dataname.lower() == 'cs':
            dataset = Coauthor(root="./CS", name=dataname)
            data = Data(x=dataset[0].x, edge_index=dataset[0].edge_index, y=dataset[0].y)
            out_feat_mat = np.load("./" + str(dataname.lower()) + "_injection/" + str(dataname.lower()) + "noise_mat_ratiio" + str(ratio) + ".npy")
        if dataname.lower() == 'wikics':
            dataset = WikiCS(root=rootname)
            out_feat_mat = np.load("./" + str(dataname.lower()) + "_injection/" + str(dataname.lower()) + "noise_mat_ratiio" + str(ratio) + ".npy")#np.load("./wikicsout_feat_mat" + str(args.datanum) + ".npy")
            data = dataset[0]
        if dataname.lower() == 'cora' or dataname.lower() == 'citeseer' or dataname.lower() == 'pubmed':
            dataset = Planetoid(root=rootname, name=dataname)
            out_feat_mat = np.load("./" + str(dataname.lower()) + "_injection/" + str(dataname.lower()) + "noise_mat_ratiio" + str(ratio) + ".npy")#np.load("./" + str(dataname.lower()) + "_injection/" + str(dataname.lower()) + "out_feat_mat" + str(2030) + ".npy")
            data = dataset[0]
        if dataname.lower() == "ogbn-arxiv":
            dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/', transform=T.ToSparseTensor())
            data = dataset[0]
            out_feat_mat = np.load("./" + str(dataname.lower()) + "_injection/" + str(dataname.lower()) + "noise_mat_ratiio" + str(ratio) + ".npy")  # np.load("./" + str(dataname.lower()) + "_injection/" + str(dataname.lower()) + "out_feat_mat" + str(2030) + ".npy")
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
            adj = sp.coo_matrix(adj)
            values = adj.data
            indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
            edge_index = torch.LongTensor(indices)
            feat_mat = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
            out_feat_mat = np.load("./" + str(dataname.lower()) + "_injection/" + str(dataname.lower()) + "noise_mat_ratiio" + str(ratio) + ".npy")  ###noisy data
            labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
            data = Data(x=feat_mat, y=labels,edge_index=edge_index)

        num_nodes = data.x.shape[0]
        feat_size = data.x.shape[1]
        feat_mat = np.array(data.x)
        anomal_data = out_feat_mat###input anomal data
        anomal_conf_prob = args.anomal_conf_prob  # hyperparameter maybe learnable


        if args.mask==True:
            print("local denosing mode on!")
            if args.gtmask == 0:
                if args.power == 2:
                    diff_mat =np.load("./" + str(dataname.lower()) + "_injection/mse_loss" + str(dataname.lower()) + str(args.gae_epoch) + "gae_mask" + str(ratio)+".npy")
                if args.power == 1:
                    diff_mat = np.load("./" + str(dataname.lower()) + "_injection/abs_loss" + str(dataname.lower()) + str(args.gae_epoch) + "gae_mask" + str(ratio)+".npy")
                gae_out = out_feat_mat - diff_mat
                mask = np.ones_like(out_feat_mat)
                mask[np.abs(diff_mat)>anomal_conf_prob] = 0 #####o for nosiy item
            if args.gtmask==1:
                if dataname.lower() == 'cora':
                    mask = 1-np.abs(feat_mat-out_feat_mat)
                else:
                    diff = np.abs(out_feat_mat-feat_mat)
                    mask = np.ones_like(feat_mat)
                    mask[diff>anomal_conf_prob]=0
                    #mask = diff
        if args.mask==False:
            print("global denosing mode on!")
            mask = np.ones_like(feat_mat)  ##all ones matrix
        # get graph Laplacian
        if dataname.lower() == "ogbn-arxiv":
            L = get_laplacian(torch.stack((dataset[0].adj_t.storage.row(), dataset[0].adj_t.storage.col())),num_nodes=num_nodes, normalization='sym')
            L_index = L[0].to(device)
            L_value = L[1].to(device)
            L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))
            # find the lambda_max of L
            # lobpcg_init = np.random.rand(num_nodes, 1)
            # lambda_max, _ = lobpcg(L, lobpcg_init)
            # lambda_max = lambda_max[0]
            lambda_max = 1.9782334
            deg = degree(dataset[0].adj_t.storage.row(), num_nodes).to(device)
        else:
            L = get_laplacian(data.edge_index, num_nodes=num_nodes, normalization='sym')
            L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))
            # get maximum eigenvalues of the graph Laplacian
            start = time.time()
            lobpcg_init = np.random.rand(num_nodes, 1)
            lambda_max, _ = lobpcg(L, lobpcg_init)
            # lambda_max , evecs_large_sparse = largest_eigsh(L, 1, which='LM')#
            lambda_max = lambda_max[0]
            print("cost on eigen:",time.time()-start,"lambdamax:",lambda_max)
            # get degrees
            deg = degree(data.edge_index[0], num_nodes).to(device)

        # extract decomposition/reconstruction Masks
        D1 = lambda x: np.cos(x / 2)
        D2 = lambda x: np.sin(x / 2)
        DFilters = [D1, D2]
        RFilters = [D1, D2]
        start1 = time.time()
        J = np.log(lambda_max / np.pi) / np.log(args.s) + args.Lev - 1  # dilation level to start the decomposition
        d = get_operator(L, DFilters, args.n, args.s, J, args.Lev)
        end1 = time.time()
        print("cost of operator:", end1 - start1)
        r = len(DFilters)
        d_list = list()
        sparse_thres=0.01
        for i in range(r):
            for l in range(args.Lev):
                #d[i, l][np.abs(d[i, l]) < sparse_thres] = 0.0       ###enhance sparseness of the matrix operators (optional)
                d_list.append(scipy_to_torch_sparse(d[i, l]))
                print("geting d list of ",i,l)

        W_list = [d.to(device) for d in d_list[args.Lev-1:]]
        # initialize the denoising filter
        smoothing = NodeDenoisingADMM(num_nodes, feat_size, r, args.Lev, args.nu, args.admm_iter,
                                      args.rho, args.mu2_0)
        # create result matrices
        num_epochs = args.epochs
        num_reps = args.reps
        # initialize the model
        model = smoothing.to(device)
        max_acc = 0.0
        # training mode
        start = time.time()
        ##check the framelet integrity
        # F = torch.from_numpy(np.eye(2708)).to(device)
        # WF = [torch.sparse.mm(W_jl, F) for W_jl in W_list]
        # tmp = [torch.mm(W_jl.to_dense().t(), W_jl.to_dense()) for W_jl in W_list]
        # #tmp = torch.sparse.mm(torch.cat(W_list, dim=1), torch.cat(W_list, dim=0))
        # print("FF:", sum(tmp),torch.sum(torch.abs(sum(tmp))))
        #print("Wtrans:",d_list[2].to_dense().t()-d_list[2])

        for lp in [0,1]:
            for lq in [1,2]:
                out,energy,diff = model(torch.from_numpy(anomal_data).to(device).to(torch.float32),W_list=W_list, d=deg,mask=torch.from_numpy(mask).to(device),
                            lp=lp,lq=lq,boost=args.boost,stop_thres=args.stop_thres,boost_value=args.boost_value,thres_iter=args.thres_iter)
                out_data = out.detach().cpu().numpy()
                print("costfor:",time.time()-start1)
                if args.mask == True:
                    print("local denosing mode "+"lp"+str(lp)+"lq"+str(lq)+"finished on "+args.dataset)
                    if args.gtmask==0:
                        if args.boost==0:
                            np.save("./"+str(dataname.lower())+"_injection/gamma"+str(args.mu2_0)+"thres"+str(args.thres_iter)+"prob"+\
                        str(anomal_conf_prob)+"gae1w_gaepower"+str(args.power)+"denoised_featmat"+str(ratio)+"_lp"+str(lp)+"lq"+str(lq), out_data)
                        if args.boost==1:
                            np.save("./"+str(dataname.lower())+"_injection/gae_power"+str(args.power)+"boostdenoised_featmat"+str(ratio)+"_lp"+str(lp)+"lq"+str(lq),out_data)
                    if args.gtmask == 1:
                        if args.boost == 0:
                            np.save("./"+str(dataname.lower())+"_injection/gamma"+str(args.mu2_0)+"thres"+str(args.thres_iter)+"prob"+\
                        str(anomal_conf_prob)+"gae1w_gtmaskdenoised_featmat" + str(ratio) + "_lp"+str(lp)+"lq"+str(lq), out_data)
                        if args.boost == 1:
                            np.save("./"+str(dataname.lower())+"_injection/"+"boostgtmaskdenoised_featmat" + str(ratio) + "_lp"+str(lp)+"lq"+str(lq), out_data)
                if args.mask == False:
                    print("global denosing mode finished!")
                    np.save("./"+str(dataname.lower())+"_injection/"+"globaldenoised_featmat" + str(ratio) + "_lp"+str(lp)+"lq"+str(lq), out_data)