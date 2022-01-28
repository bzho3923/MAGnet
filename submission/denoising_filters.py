import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


def soft_thresholding(x, soft_eta, mode):
    """
    Perform row-wise soft thresholding.
    The row wise shrinkage is specific on E(k+1) updating
    The element wise shrinkage is specific on Z(k+1) updating

    :param x: one block of target matrix, shape[num_nodes, num_features]
    :param soft_eta: threshold scalar stores in a torch tensor
    :param mode: model types selection "row" or "element"
    :return: one block of matrix after shrinkage, shape[num_nodes, num_features]

    """
    assert mode in ('element', 'row'), 'shrinkage type is invalid (element or row)'
    if mode == 'row':
        row_norm = torch.linalg.norm(x, dim=1).unsqueeze(1)
        row_norm.clamp_(1e-12)
        row_thresh = (F.relu(row_norm - soft_eta) + soft_eta) / row_norm
        out = x * row_thresh
    else:
        out = F.relu(x - soft_eta) - F.relu(-x - soft_eta)

    return out


def hard_thresholding(x, soft_eta, mode):
    """
    Perform row-wise hard thresholding.
    The row wise shrinkage is specific on E(k+1) updating
    The element wise shrinkage is specific on Z(k+1) updating

    :param x: one block of target matrix, shape[num_nodes, num_features]
    :param soft_eta: threshold scalar stores in a torch tensor
    :param mode: model types selection "row" or "element"
    :return: one block of matrix after shrinkage, shape[num_nodes, num_features]

    """
    assert mode in ('element', 'row'), 'shrinkage type is invalid (element or row)'
    tmp = torch.zeros_like(x)
    tmp[x - soft_eta > 0] = 1
    tmp[-x - soft_eta > 0] = 1
    # tmp[-x - soft_eta < 0]=0
    # tmp[x - soft_eta < 0] = 0
    # print("tmp:",torch.max(tmp))
    out = x * tmp
    return out


# Node denoising filter
class NodeDenoisingADMM(nn.Module):
    def __init__(self, num_nodes, num_features, r, J, nu, admm_iter, rho, gamma_0):
        super(NodeDenoisingADMM, self).__init__()
        self.r = r
        self.J = J
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.admm_iter = admm_iter
        self.rho = rho
        self.nu = [nu] * J
        print("nu1:", self.nu)
        for i in range(J):
            self.nu[i] = self.nu[i] / np.power(4.0, i)  # from (4.3) in Dong's paper
        self.nu = [0.0] + self.nu  # To include W_{0,J}
        self.gamma_max = 1e+6
        self.initial_gamma = gamma_0
        self.gamma = self.initial_gamma

    def forward(self, F, W_list, d, mask, init_Zk=None, init_Yk=None, lp=1, lq=2, boost=False, stop_thres=0.05,
                boost_value=4, thres_iter=15):
        """
        Parameters
        ----------
        F : Graph signal to be smoothed, shape [Num_node, Num_features].
        W_list : Framelet Base Operator, in list, each is a sparse matrix of size Num_node x Num_node.
        d : Vector of normalized graph node degrees in shape [Num_node, 1].
        init_Zk: Initialized list of (length: j * l) zero matrix in shape [Num_node, Num_feature].
        init_Yk: Initialized lists of (length: j*l) zero matrix in shape [Num_node, Num_feature].

        :returns:  Smoothed graph signal U

        """
        if init_Zk is None:
            Zk = []
            for j in range(self.r - 1):
                for l in range(self.J):
                    Zk.append(torch.zeros(torch.Size([self.num_nodes, self.num_features])).to(F.device))
            Zk = [torch.zeros((self.num_nodes, self.num_features)).to(F.device)] + Zk
        else:
            Zk = init_Zk
        if init_Yk is None:
            Yk = []
            for j in range(self.r - 1):
                for l in range(self.J):
                    Yk.append(torch.zeros(torch.Size([self.num_nodes, self.num_features])).to(F.device))
            Yk = [torch.zeros((self.num_nodes, self.num_features)).to(F.device)] + Yk
        else:
            Yk = init_Yk

        energy = 1000000000
        self.gamma = self.initial_gamma
        vk = [Yk_jl for Zk_jl, Yk_jl in zip(Zk, Yk)]
        Uk = F
        diff = 10000
        k = 1
        ak = boost_value
        v_til = vk
        energy_list = []
        diff_list = []
        while energy > stop_thres:  # or k<thres_iter:
            if lp == 1:
                Zk = [soft_thresholding((2 * Yk_jl - vk_jl) / self.gamma, (nu_jl / self.gamma) * d.unsqueeze(1),
                                        'element')
                      for nu_jl, vk_jl, Yk_jl in zip(self.nu, v_til, Yk)]
            if lp == 0:
                Zk = [hard_thresholding((2 * Yk_jl - vk_jl) / self.gamma, (nu_jl / self.gamma) * d.unsqueeze(1),
                                        'element')
                      for nu_jl, vk_jl, Yk_jl in zip(self.nu, v_til, Yk)]
            if lq == 2:
                if boost == 0:
                    v_til = [Yk_jl - self.gamma * Zk_jl for Zk_jl, Yk_jl in zip(Zk, Yk)]
                if boost == 1:
                    vk_old = vk  ## v_k
                    vk = [Yk_jl - self.gamma * Zk_jl for Zk_jl, Yk_jl in zip(Zk, Yk)]
                    v_til = [item + boost_value * (item - item0) for item0, item in zip(vk_old, vk)]
                U_init = Uk
                Uk = (d.unsqueeze(1) * F * mask * mask - torch.sparse.mm(torch.cat(W_list, dim=1),
                                                                         torch.cat(v_til, dim=0))) / (
                                 d.unsqueeze(1) * mask * mask + self.gamma)
            if lq == 1:
                if boost == 0:
                    v_til = [Yk_jl - self.gamma * Zk_jl for Zk_jl, Yk_jl in zip(Zk, Yk)]
                if boost == 1:
                    # boosta= (k - 1) / (k + boost_value)
                    vk_old = vk
                    vk = [Yk_jl - self.gamma * Zk_jl for Zk_jl, Yk_jl in zip(Zk, Yk)]
                    v_til = [item + boost_value * (item - item0) for item0, item in zip(vk_old, vk)]
                WTV = torch.sparse.mm(torch.cat(W_list, dim=1), torch.cat(v_til, dim=0))
                Yk = soft_thresholding(-F - WTV / self.gamma, (1 / 2 * self.gamma) * d.unsqueeze(1), 'element')
                Yk = mask * Yk + (1 - mask) * (-F - WTV / self.gamma)
                U_init = Uk
                Uk = Yk + F
            if boost == 0:
                Yk = [vk_jl + self.gamma * torch.sparse.mm(W_jl, Uk) for vk_jl, W_jl in zip(v_til, W_list)]
            if boost == 1:
                Yk = [vk_jl + self.gamma * torch.sparse.mm(W_jl, Uk) for vk_jl, W_jl in zip(v_til, W_list)]

            if lp == 1:
                energy_1 = [nu_jl * torch.sum(d.unsqueeze(1) * torch.abs(torch.sparse.mm(W_jl, Uk))) for
                            nu_jl, W_jl in zip(self.nu, W_list)]
            if lp == 0:
                energy_1 = [torch.nonzero(nu_jl * d.unsqueeze(1) * torch.sparse.mm(W_jl, Uk)).shape[0] for
                            nu_jl, W_jl in zip(self.nu, W_list)]
                print("en1", sum(energy_1))
            if lq == 2:
                energy2 = 0.5 * torch.sum(d.unsqueeze(1) * torch.pow(mask * (Uk - F), 2))
            if lq == 1:
                energy2 = 0.5 * torch.sum(d.unsqueeze(1) * torch.abs(mask * (Uk - F)))
            energy = sum(energy_1) + energy2
            energy_list.append(energy.item())
            diff = torch.sum(torch.abs(Uk - U_init)).item()
            diff_list.append(diff)
            print("difference&energy:", round(diff, 2), "    ", round(energy.item(), 0))
            k += 1
            # if k>thres_iter:
            #     break
        return Uk, energy_list, diff_list
