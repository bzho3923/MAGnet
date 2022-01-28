import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--reps', type=int, default=10,
                    help='number of repetitions')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--wd', type=float, default=0.001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=1433,
                    help='number of hidden units')
parser.add_argument('--Lev', type=int, default=4,
                    help='level of transform')
parser.add_argument('--s', type=float, default=2,
                    help='dilation scale > 1')
parser.add_argument('--n', type=int, default=10,
                    help='n - 1 = Degree of Chebyshev Polynomial Approximation')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout probability')
parser.add_argument('--shrinkage', type=str, default='soft',
                    help='soft or hard thresholding')
parser.add_argument('--sigma', type=float, default=1.0,
                    help='standard deviation of the noise')
parser.add_argument('--admm_iter', type=int, default=15,
                    help='number of admm iterations')
parser.add_argument('--rho', type=float, default=0.95,
                    help='piecewise function: constant and > 1')
parser.add_argument('--mu1_0', type=float, default=1.0,
                    help='initial value of mu1')
parser.add_argument('--mu3_0', type=float, default=1.0,
                    help='initial value of mu3')
parser.add_argument('--mu4_0', type=float, default=1.0,
                    help='initial value of mu4')
parser.add_argument('--lam', type=float, default=10.0,
                    help='weight of quadratic term in objective function')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--filter_type', type=str, default='DoT',
                    help='denoising filter type with choices "TV", "Node", "Edge", "DoT", "None"')
parser.add_argument('--filename', type=str, default='results',
                    help='filename to store results and the model')
parser.add_argument('--attack', type=str, default='Mix',
                    help='attack type with choices "Node", "Edge", "Mix", "None"')
parser.add_argument('--GConv_type', type=str, default='GCN',
                    help='graph convolution type with choices "GCN", "GAT", "UFG_S", "UFG_R"')