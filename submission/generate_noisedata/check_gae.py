import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import lobpcg
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid, WikiCS
from torch_geometric.utils import get_laplacian, degree
from jw_denoising_filters import *
from ufg_layer import UFGConv_S, UFGConv_R
from utils import scipy_to_torch_sparse, get_operator
from config import parser
import random
import os.path as osp
import os
import matplotlib.pyplot as plt
import numba
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
import time
from torch_geometric.data import Data
@numba.jit()
def compute_metric(diff_mask,noise_mask):
    fp,fn,tn=0,0,0
    for i in range(0, (diff_mask).shape[0]):
        if diff_mask[i] == 1 and noise_mask[i] == 0:
            fp += 1
        if diff_mask[i] == 0 and noise_mask[i] == 1:
            fn += 1
        if diff_mask[i] == 0 and noise_mask[i] == 0:
            tn += 1
    return fp,fn,tn

if __name__ == '__main__':
    # get config
    parser.add_argument('--dataset', type=str, default='PubMed',
                        help='name of dataset with choices "Cora", "Citeseer", "Wikics"')
    parser.add_argument('--ratio', type=float, default=0.05,
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
    parser.add_argument('--stop_thres', type=float, default=3000000,
                        help='stopping criteria to stop the ADMM')
    parser.add_argument('--mu2_0', type=float, default=10,
                        help='initial value of mu2')
    parser.add_argument('--anomal_conf_prob', type=float, default=0.001,
                        help='boost alue')
    parser.add_argument('--thres_iter', type=float, default=15,
                        help='boost alue')
    parser.add_argument('--nu', type=float, default=500,
                        help='tight wavelet frame transform tuning parameter')
    parser.add_argument('--gae_epoch', type=int, default=20000,
                        help='gae training epoch')
    args = parser.parse_args()
    if args.filter_type.lower() == 'dot':
        args.filter_type = 'Breg'

    # set random seed for reproducible results
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # training on CPU/GPU device
    device = torch.device("cuda")  # "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # load dataset
    for dataname in ["cora","Citeseer","PubMed","wikics","wisconsin","texas","CS"]:
        ratio_list = [0.5]  # [0.05,0.1,0.3,0.5,0.75]
        for ratio in ratio_list:

            rootname = osp.join('./', dataname)
            if dataname.lower() == 'cs':
                dataset = Coauthor(root="./CS", name=dataname)
                data = Data(x=dataset[0].x, edge_index=dataset[0].edge_index, y=dataset[0].y)
                out_feat_mat = np.load(
                    "./" + str(dataname.lower()) + "_injection/" + str(dataname.lower()) + "noise_mat_ratiio" + str(
                        ratio) + ".npy")
            if dataname.lower() == 'wikics':
                dataset = WikiCS(root=rootname)
                out_feat_mat = np.load(
                    "./" + str(dataname.lower()) + "_injection/" + str(dataname.lower()) + "noise_mat_ratiio" + str(
                        ratio) + ".npy")  # np.load("./wikicsout_feat_mat" + str(args.datanum) + ".npy")
                data = dataset[0]
            if dataname.lower() == 'cora' or dataname.lower() == 'citeseer' or dataname.lower() == 'pubmed':
                dataset = Planetoid(root=rootname, name=dataname)
                out_feat_mat = np.load(
                    "./" + str(dataname.lower()) + "_injection/" + str(dataname.lower()) + "noise_mat_ratiio" + str(
                        ratio) + ".npy")  # np.load("./" + str(dataname.lower()) + "_injection/" + str(dataname.lower()) + "out_feat_mat" + str(2030) + ".npy")
                data = dataset[0]
            if dataname.lower() == 'wisconsin' or dataname.lower() == 'texas':
                from new_dataset import HetroDataSet
                dataset = HetroDataSet(root=rootname, name=dataname)
                num_class = dataset.num_classes
                data = dataset[0]
                out_feat_mat = np.load(
                    "./" + str(dataname.lower()) + "_injection/" + str(dataname.lower()) + "noise_mat_ratiio" + str(
                        ratio) + ".npy")  ###noisy data

                clean_data = data.clone().to(device)

            num_nodes = dataset[0].x.shape[0]
            feat_size = dataset[0].x.shape[1]
            total_item = num_nodes*feat_size
            # np.save("./clean_feat",data.x)
            feat_mat = np.array(data.x)
            anomal_data = out_feat_mat  ###input anomal data
            anomal_list = [0.0001,0.001,0.01,0.1,0.5]
            agree_power1=[]
            agree_power2=[]
            disagree_power1 = []
            disagree_power2 = []

            tp_power1 = []
            tn_power1 = []
            fp_power1 = []
            fn_power1 = []

            tp_power2 = []
            tn_power2 = []
            fp_power2 = []
            fn_power2 = []


            for anomal_conf_prob in anomal_list:
                for power in [1, 2]:
                    if power == 2:
                        diff_mat = np.load(
                            "./" + str(dataname.lower()) + "_injection/mse_loss" + str(dataname.lower()) + str(
                                args.gae_epoch) + "gae_mask" + str(ratio) + ".npy")
                    if power == 1:
                        diff_mat = np.load(
                            "./" + str(dataname.lower()) + "_injection/abs_loss" + str(dataname.lower()) + str(
                                args.gae_epoch) + "gae_mask" + str(ratio) + ".npy")
                    gae_out = out_feat_mat - diff_mat
                    diff_mask = np.zeros_like(out_feat_mat)
                    # print("check clean:::", feat_mat[10:20, 50:70], "\n noise", out_feat_mat[10:20, 50:70], "\n gae",
                    #       gae_out[10:20, 50:70])
                    diff_mask[np.abs(diff_mat) > anomal_conf_prob] = 1

                    noise_mask = np.zeros_like(out_feat_mat)
                    noise_mask[np.abs(out_feat_mat - feat_mat)>0]=1
                    agree = np.sum(diff_mask * noise_mask)
                    disagree = np.sum(np.abs(diff_mask - noise_mask))
                    tp = agree
                    false= disagree
                    diff_mask = diff_mask.flatten()
                    noise_mask = noise_mask.flatten()
                    fp, fn, tn = compute_metric(diff_mask,noise_mask)
                    if power == 2:
                        agree_power2.append(agree)
                        disagree_power2.append(disagree)
                        tp_power2.append(tp)
                        tn_power2.append(tn)
                        fp_power2.append(fp)
                        fn_power2.append(fn)
                    if power == 1:
                        agree_power1.append(agree)
                        disagree_power1.append(disagree)
                        tp_power1.append(tp)
                        tn_power1.append(tn)
                        fp_power1.append(fp)
                        fn_power1.append(fn)
                    print(dataname.lower()+" under anomal_prob "+str(anomal_conf_prob)+" loss_power=", str(power), "agree:", int(agree),"ratio:",agree/total_item,"\ndisagree:",int(disagree), "ratio:", disagree / total_item)
            show ="tp"
            if show=="agree":
                x_width = range(0,len(anomal_list))
                x2_width = [i + 0.2 for i in x_width]
                x3_width = [i + 0.2 for i in x2_width]
                x4_width = [i + 0.2 for i in x3_width]

                plt.bar(x_width, agree_power1, lw=0.5, fc="#8ECFC9", width=0.2, label="MAE-agree")
                plt.bar(x2_width, agree_power2, lw=0.5, fc='#FFBE7A', width=0.2, label="MSE-agree")
                plt.bar(x3_width, disagree_power1, lw=0.5, fc="#FA7F6F", width=0.2, label="MAE-disagree")
                plt.bar(x4_width, disagree_power2, lw=0.5, fc='#82B0D2', width=0.2, label="MSE-disagree")


                plt.legend()
                plt.title("dataset "+dataname)
                plt.xlabel("anomaly threshold")
                plt.ylabel("count")
                plt.xticks(range(0, len(anomal_list)), anomal_list)
                plt.savefig("./figure/agree_disagree of dataset"+dataname+".png", bbox_inches='tight')
                plt.cla()
                plt.clf()
            else:
                x_width = range(0, len(anomal_list))
                x2_width = [i + 0.2 for i in x_width]
                x3_width = [i + 0.2 for i in x2_width]
                x4_width = [i + 0.2 for i in x3_width]
                true=0
                if true==0:
                    #plt.bar(x_width, fp_power1, lw=0.5, fc="#FA7F6F", width=0.2, label="MAE-FP")
                    plt.bar(x_width, fn_power1, lw=0.5, fc='#82B0D2', width=0.2, label="MAE-FN")
                    #plt.bar(x3_width, fp_power2, lw=0.5, fc="#2878b5", width=0.2, label="MSE-FP")
                    plt.bar(x2_width, fn_power2, lw=0.5, fc='#9ac9db', width=0.2, label="MSE-FN")
                else:
                    plt.bar(x_width, tp_power1, lw=0.5, fc="#8ECFC9", width=0.2, label="MAE-TP")
                    #plt.bar(x2_width, tn_power1, lw=0.5, fc='#FFBE7A', width=0.2, label="MAE-TN")
                    plt.bar(x2_width, tp_power2, lw=0.5, fc="#ff8884", width=0.2, label="MSE-TP")
                    #plt.bar(x4_width, tn_power2, lw=0.5, fc='#05B9E2', width=0.2, label="MSE-TN")


                plt.legend()
                plt.title("dataset " + dataname)
                plt.xlabel("anomaly threshold")
                plt.ylabel("count")
                plt.xticks(range(0, len(anomal_list)), anomal_list)
                if true==0:
                    plt.savefig("./figure/tp"+str(args.gae_epoch)+"/false of dataset" + dataname + ".png", bbox_inches='tight')
                else:
                    plt.savefig("./figure/tp"+str(args.gae_epoch)+"/true of dataset" + dataname + ".png", bbox_inches='tight')
                plt.cla()
                plt.clf()