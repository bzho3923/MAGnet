from scipy import sparse
from scipy.sparse.linalg import lobpcg
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid, WikiCS
from torch_geometric.utils import get_laplacian, degree
from denoising_filters import *
from utils import scipy_to_torch_sparse, get_operator
from config import parser
from config import parser
import random
import os.path as osp
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
import time

if __name__ == '__main__':
    # get config
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='name of dataset with choices "Cora", "Citeseer", "Wikics"')
    parser.add_argument('--datanum', type=int, default=1350,
                        help='data number with anomal nodes 150,270,500')
    parser.add_argument('--mask', type=int, default=1,
                        help='using mask for local denoising 1/without mask for global denoising 0')
    parser.add_argument('--gtmask', type=int, default=1,
                        help='using groundtruth mask mat')
    parser.add_argument('--lp', type=int, default=1,
                        help='first term using lp norm')
    parser.add_argument('--lq', type=int, default=2,
                        help='second term using lq norm')
    parser.add_argument('--boost', type=int, default=1,
                        help='using accelerated scheme or not')
    parser.add_argument('--boost_value', type=float, default=0.3,
                        help='boost alue')
    parser.add_argument('--stop_thres', type=float, default= 3000000,
                        help='stopping criteria to stop the ADMM')
    parser.add_argument('--mu2_0', type=float, default=10,
                        help='initial value of mu2')
    parser.add_argument('--anomal_conf_prob', type=float, default=0.001,
                        help='boost alue')
    parser.add_argument('--thres_iter', type=float, default=25,
                        help='boost alue')
    parser.add_argument('--nu', type=float, default=500,
                        help='tight wavelet frame transform tuning parameter')
    args = parser.parse_args()
    if args.filter_type.lower() == 'dot':
        args.filter_type = 'Breg'

    # set random seed for reproducible results
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # training on CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # load dataset
    dataname = args.dataset
    rootname = osp.join('./', dataname)
    if dataname.lower() == 'wikics':
        dataset = WikiCS(root=rootname)
        out_feat_mat = np.load("./wikicsout_feat_mat" + str(args.datanum) + ".npy")
        print("shape:",out_feat_mat.shape)
    else:
        dataset = Planetoid(root=rootname, name=dataname)
        out_feat_mat = np.load("./out_feat_mat" + str(args.datanum) + ".npy")
    data = dataset[0]
    num_nodes = data.x.shape[0]
    feat_size = data.x.shape[1]
    feat_mat = np.array(data.x)

    anomal_data = out_feat_mat###input anomal data
    anomal_conf_prob = args.anomal_conf_prob  # hyperparameter maybe learnable


    if args.mask==True:
        print("local denosing mode on!")
        if args.gtmask == 0:
            if dataname.lower() == 'cora':
                diff_mat = np.load("./out_mask" + str(args.datanum) + ".npy")
                gae_out = out_feat_mat - diff_mat
                ###construct consistent mask mat
                gae_out[abs(gae_out) < anomal_conf_prob] = 0
                gae_out[abs(gae_out) > anomal_conf_prob] = 1
                gae_out[abs(gae_out) == anomal_conf_prob] = 1
                diff_mat = np.abs(out_feat_mat - gae_out)
                mask = 1 - diff_mat  ##standd for the consistent mask
            else:
                ##consistent mask construction
                diff_mat = np.load("./out_mask" + str(args.datanum) + ".npy")
                gae_out = out_feat_mat - diff_mat
                mask = np.zeros_like(out_feat_mat)
                mask[np.abs(diff_mat)>anomal_conf_prob] = 0

        if args.gtmask==1:
            if dataname.lower() == 'cora':
                mask = 1-np.abs(feat_mat-out_feat_mat)
            else:
                diff = np.abs(out_feat_mat-feat_mat)
                mask = np.ones_like(feat_mat)
                diff[diff>anomal_conf_prob]=0
                mask = diff
    if args.mask==False:
        print("global denosing mode on!")
        mask = np.ones_like(feat_mat)  ##all ones matrix
    # get degrees
    deg = degree(data.edge_index[0], num_nodes).to(device)

    # get graph Laplacian
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
    start1 = time.time()
    J = np.log(lambda_max / np.pi) / np.log(args.s) + args.Lev - 1  # dilation level to start the decomposition
    d = get_operator(L, DFilters, args.n, args.s, J, args.Lev)
    end1 = time.time()
    print("cost of operator:",end1-start1)
    # enhance sparseness of the matrix operators (optional)
    # d[np.abs(d) < 0.001] = 0.0

    # store the matrix operators (torch sparse format) into a list: row-by-row
    r = len(DFilters)
    #print("len dlist:",r)
    d_list = list()
    for i in range(r):
        for l in range(args.Lev):
            d_list.append(scipy_to_torch_sparse(d[i, l]))
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
    # reset the model parameters
    #model.reset_parameters()

    # training mode
    start = time.time()
    # model.train()
    ##check the framelet integrity
    # F = torch.from_numpy(np.eye(2708)).to(device)
    # WF = [torch.sparse.mm(W_jl, F) for W_jl in W_list]
    # tmp = [torch.mm(W_jl.to_dense().t(), W_jl.to_dense()) for W_jl in W_list]
    # #tmp = torch.sparse.mm(torch.cat(W_list, dim=1), torch.cat(W_list, dim=0))
    # print("FF:", sum(tmp),torch.sum(torch.abs(sum(tmp))))
    #print("Wtrans:",d_list[2].to_dense().t()-d_list[2])


    out,energy,diff = model(torch.from_numpy(anomal_data).double().to(device),W_list=W_list, d=deg,mask=torch.from_numpy(mask).to(device),
                lp=args.lp,lq=args.lq,boost=args.boost,stop_thres=args.stop_thres,boost_value=args.boost_value,thres_iter=args.thres_iter)
    if args.boost==1:
        np.save("./energyboost_"+str(args.boost_value)+"lp_"+str(args.lp)+"lq_"+str(args.lq)+"iter_"+str(args.thres_iter),np.array(energy))
        np.save("./diffboost_" + str(args.boost_value) + "lp_" + str(args.lp) + "lq_" + str(args.lq)+"iter_"+str(args.thres_iter), np.array(diff))
    if args.boost==0:
        np.save("./energy_"+"lp_"+str(args.lp)+"lq_"+str(args.lq)+"iter_"+str(args.thres_iter)+"nu_"+str(args.nu)+"gama_"+str(args.mu2_0),np.array(energy))
        np.save("./diff_"  + "lp_" + str(args.lp) + "lq_" + str(args.lq)+"iter_"+str(args.thres_iter)+"nu_"+str(args.nu)+"gama_"+str(args.mu2_0), np.array(diff))
    end = time.time()
    print("time cost:",end-start)
    out_data = out.detach().cpu().numpy()
    if args.mask == True:
        print("local denosing mode finished!")
        if args.gtmask==0:
            if args.boost==0:
                np.save("./denoised_featmat"+str(args.datanum)+"_lp"+str(args.lp)+"lq"+str(args.lq),out_data)
            if args.boost==1:
                np.save("./boostdenoised_featmat"+str(args.datanum)+"_lp"+str(args.lp)+"lq"+str(args.lq),out_data)
        if args.gtmask == 1:
            if args.boost == 0:
                np.save("./gtmaskdenoised_featmat" + str(args.datanum) + "_lp"+str(args.lp)+"lq"+str(args.lq), out_data)
            if args.boost == 1:
                np.save("./boostgtmaskdenoised_featmat" + str(args.datanum) + "_lp"+str(args.lp)+"lq"+str(args.lq), out_data)
    if args.mask == False:
        print("global denosing mode finished!")
        np.save("./globaldenoised_featmat" + str(args.datanum) + "_lp"+str(args.lp)+"lq"+str(args.lq), out_data)