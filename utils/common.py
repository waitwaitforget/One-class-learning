import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

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


# basic configuration
DATASETS = ['kdd99', 'thyroid', 'sensor', 'mnist','svhn','cifar10']

dataset_map = {'mnist': DsInfo('mnist', 784, 10, NewMnist)}
