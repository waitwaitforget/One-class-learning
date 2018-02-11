import sys
import os
import torch
import torch.nn as nn


from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

from utils.common import *


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cls', type=int, required=True, help='testing class')
parser.add_argument('-e', '--epoch', type=int, default=100)
parser.add_argument('-r', '--ratio', type=float, default=0.3, help='ratio of outliers')
parser.add_argument('-b', '--batchsize', type=int, default=64)
parser.add_argument('-a', '--average', action='store_true', default=False, help='average performance over classes')
parser.add_argument('-d', '--dataset', type=str, default='mnist')
parser.add_argument('--resume', help='Checkpoint path')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

if not os.path.exists('ckpt'):
    os.mkdir('ckpt')

dsname = args.dataset
num_epochs = args.epoch
batch_size = args.batchsize

i_dim = dataset_map[dsname].ndim
h_dim = 128
z_dim = 64

log_filename = dsname + '_ebm_log.txt'
log_filename = logpath(log_filename)

avg_prec, avg_recall, avg_f1 = 0, 0, 0


def test_energy(model, data):
    model.eval()
    energy = model(data)
    return energy


def test_score(model, data):
    model.eval()
    score = model.score(data)
    return score

def train(cid):
    layershape = [i_dim, h_dim, z_dim]
    model = FCEBM(layershape, use_cuda)
    transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train_loader = DataLoader(dataset_map[dsname].dataset('./data',train=True, transform=transform, cid=cid, ratio=args.ratio),
                              batch_size=64)

    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    # hyper parameters
    max_epoch = args.epoch
    if use_cuda:
        model.cuda()

    for epoch in range(max_epoch):
        loss = 0.0
        thres_ = []
        score_thres = []
        for ib, (data, target) in enumerate(train_loader):
            data = Variable(data, requires_grad=True)
            if use_cuda:
                data = data.cuda()
            optimizer.zero_grad()
            #print type(data.data)
            energy = model(data.view(data.size(0), -1))

            thres_.append(energy.data.cpu())
            score_thres.append(model.score(data.view(data.size(0), -1)).cpu())
            energy = energy.mean()
            energy.backward()
            optimizer.step()

            loss += energy.data[0]
        print('Epoch = {} energy = {:.4f}'.format(epoch, loss/ib))


    thres_ = torch.cat(thres_)
    score_thres = torch.cat(score_thres)
    threshold = np.percentile(thres_.numpy(), 95)
   
    score_threshold = np.percentile(score_thres.data.numpy(), 95)

    # save results
    checkpoint = dict(threshold=threshold, state_dict=model.state_dict(), score_threshold=score_threshold)

    torch.save(checkpoint, os.path.join('ckpt', dsname + '_febm.ckpt.pth'))

def test(cid):
    checkpoint = torch.load(os.path.join('ckpt', dsname + '_febm.ckpt.pth'))
    transform = transforms.Compose([transforms.ToTensor()])
    
    layershape = [i_dim, h_dim, z_dim]
    model = FCEBM(layershape, use_cuda)
    model.load_state_dict(checkpoint['state_dict'])
    threshold = checkpoint['threshold']
    score_threshold = checkpoint['score_threshold']
    
    print('threshold :{:.4f}'.format(threshold))
    # test energy
    test_loader = DataLoader(dataset_map[dsname].dataset('./data',train=False, transform=transform, cid=cid, ratio=0.3),
                              batch_size=64)
    test_pred = []
    test_target = []
    test_score2 = []
    for ib, (data, target) in enumerate(test_loader):
        data = Variable(data.view(data.size(0), -1))
        if use_cuda:
            data = data.cuda()
        score =test_score(model, data)
        e = test_energy(model, data)
        #print e
        test_pred.append(e.lt(threshold).cpu())
        test_score2.append(score.lt(score_threshold).cpu())


        test_target.append(target)

    test_pred = torch.cat(test_pred, 0)

    test_target = torch.cat(test_target, 0)
    test_score2 = torch.cat(test_score2, 0)

    test_pred = test_pred.data
    test_score2 = test_score2.data
    test_target = test_target

    precision, recall, f1 = basic_measures(test_target.numpy(), test_pred.numpy())
    
    print('class {} Test precision {:.4f} recall {:.4f} f1 {:.4f}'.format(cid, precision, recall, f1))
    logging(log_filename, '{:.4f} {:.4f} {:.4f}'.format(precision, recall, f1))
    precision = precision_score(test_target.numpy(), test_score2.numpy())
    recall = recall_score(test_target.numpy(), test_score2.numpy())
    f1 = f1_score(test_target.numpy(), test_score2.numpy())

    print('Score Test precision {:.4f} recall {:.4f} f1 {:.4f}'.format(precision, recall, f1))


if __name__=='__main__':
    if args.average:
        for cid in range(1, dataset_map[dsname].nclass+1):
            train(cid)
            test(cid)
        avg_prec /= dataset_map[dsname].nclass
        avg_recall /= dataset_map[dsname].nclass
        avg_f1 /= dataset_map[dsname].nclass
        print('Avergae performance: precision {:.4f}, recall {:.4f}, f1 {:.4f}'.format(avg_prec, avg_recall, avg_f1))
    else:
        train(args.cls)
        test(args.cls)
    print('Done')