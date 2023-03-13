from scipy.sparse import data
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
import scipy.io
from sklearn.metrics import roc_auc_score
from datetime import datetime
import argparse

from model import Dominant
from utils import *



def loss_func( attrs, X_hat,power=2):
    # Attribute reconstruction loss
    if power ==1:
        diff_attribute = torch.abs(X_hat - attrs)
        attribute_reconstruction_errors = torch.sum(diff_attribute, 1)
    if power==2:
        diff_attribute = torch.pow(X_hat - attrs,2)
        attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)
    return attribute_cost

def train_dominant(args):
    PATH = "GAE"
    dataname = args.dataset
    ratio_list = [0.5]#[0.05,0.1,0.3, 0.5, 0.75]
    for ratio in ratio_list:
        adj, attrs, label = load_anomaly_npy_dataset(dataname,datanum=ratio) ###load noise data
        #adj = torch.FloatTensor(adj)
        attrs = torch.FloatTensor(attrs)
        model = Dominant(feat_size = attrs.size(1), hidden_size = args.hidden_dim, dropout = args.dropout)
        if args.device == 'cuda':
            device = torch.device(args.device)
            adj = adj.to(device)
            attrs = attrs.to(device)
            model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.98, patience=5, verbose=True,min_lr=0.000000001)
        min_score = 1000
        for epoch in range(args.epoch):
            model.train()
            optimizer.zero_grad()
            X_hat = model(attrs, adj)
            loss = loss_func(attrs, X_hat,power=args.power)
            loss.backward()
            optimizer.step()
            print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item()))
            scheduler.step(loss)

            if epoch%10 == 0 or epoch == args.epoch - 1:
                model.eval()
                X_hat = model(attrs, adj)
                loss = loss_func(attrs, X_hat,power=args.power)
                score = loss.detach().cpu().numpy()
                if score<500 and score < min_score:
                    min_score = score
                    print("dataset "+dataname+" ready to save the mask mat X-X^!")
                    mask = attrs-X_hat
                    mask = mask.detach().cpu().numpy()
                    if epoch%100==0:
                        if args.power==2:
                            np.save("./"+str(dataname.lower())+"_injection/mse_loss"+str(dataname.lower())+str(args.epoch)+"gae_mask"+str(ratio),mask)
                        if args.power==1:
                            np.save("./"+str(dataname.lower())+"_injection/abs_loss"+str(dataname.lower()) + str(args.epoch) + "gae_mask" + str(ratio), mask)
                        print("save @ epoch ", epoch)
                        state = {'net': model.state_dict(), 'optim': optimizer.state_dict(), 'epoch': epoch + 1,"lr":optimizer.state_dict()['param_groups'][0]['lr']}  ##最好保存lr
                        # torch.save(state, PATH + "_cora"+str(args.datanum) + ".pth")
                print("Epoch:", '%04d' % (epoch), 'Score', score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Citeseer', help='dataset name: Flickr/ACM/BlogCatalog')
    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--ratio_anomal', type=float, default=0.05,
                        help='data number with anomal nodes 150,270,500')
    parser.add_argument('--epoch', type=int, default=5000, help='Training epoch')
    parser.add_argument('--power', type=int, default=2, help='abs or mse loss')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.8, help='balance parameter')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')

    args = parser.parse_args()

    train_dominant(args)