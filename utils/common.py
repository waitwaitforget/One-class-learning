import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import torch

from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report

from utils.commondataloader import CommonDataLoader
from utils.NewMnist import NewMnist

sys.path.append('..')
from models.fcebm import FCEBM


# load the data
def load_data(dsname):
	data =pickle.load(open(dsname+'.data.pkl'))
	train_data = data['train_data']
	train_labels = data['train_labels']
	test_data = data['test_data']
	test_labels = data['test_labels']

	return train_data, train_labels, test_data, test_labels

# logging 
def logging(file, content):
	with open(file, 'a+') as f:
		f.write(content + '\n')

# measure
def basic_measures(target, pred):
	recall = recall_score(target, pred)
	precision = precision_score(target, pred)
	f1_ = f1_score(target, pred)
	return precision, recall, f1_

def logpath(file):
	if not os.path.exists('logs'):
		os.mkdir('logs')
	if not os.path.exists('logs/'+file.split('_')[0]):
		os.mkdir('logs/'+file.split('_')[0])
	return os.path.join('logs/'+file.split('_')[0], file)


class DsInfo(object):
	def __init__(self, name, ndim, nclass, dataset):
		self.name = name
		self.ndim = ndim
		self.nclass = nclass
		self.dataset  = dataset


def pairwise_distance(X, Y, type, params):
	if type == 'linear':
		K = torch.mm(X.transpose(1,0), Y)

	elif type == 'poly':
		order = params['order']
		bias = params['bias']
		K = torch.pow(torch.mm(X.transpose(1, 0), Y) + bias, order)

	elif type == 'gauss':
		sigma = params['sigma']

		n = X.size(0)
		m = Y.size(0)

		norm1 = torch.sum(torch.pow(X, 2), 1)
		norm2 = torch.sum(torch.pow(Y, 2), 1)

		mat1 = norm1.unsqueeze(1).repeat(1, m)
		mat2 = norm2.unsqueeze(0).repeat(n, 1)

		K = mat1 + mat2 - 2 * torch.mm(X, Y.transpose(1, 0))
		K = torch.exp(-K / (2*sigma**2))
	else:
		raise NotImplementedError

	return K


# basic configuration
DATASETS = ['kdd99', 'thyroid', 'sensor', 'mnist','svhn','cifar10']

dataset_map = {'mnist': DsInfo('mnist', 784, 10, NewMnist)}
