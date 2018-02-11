import numpy as np
import pickle

class CommonDataLoader(object):
	
	def __init__(self, name, classes, ratio):
		self.data_dir = './data/'

		self.train = pickle.load(open(self.data_dir+ name +'/'+name+'.train.pkl'))
		self.test  = pickle.load(open(self.data_dir+ name +'/'+name+'.test.pkl'))

		self.nclass = classes
		self.ratio = ratio

	def load_data(self, cid, zero=False):
		train_data = self.train['train_data']
		train_labels = self.train['train_labels']
		#train_labels.dtype= np.int32
		#print train_labels.shape
		id = np.where(train_labels == cid)
		id = id[0]
		train_data = train_data[id]
		train_labels = np.ones((train_data.shape[0],1))

		test_data = self.test['test_data']
		test_labels = self.test['test_labels']

		id = np.where(test_labels==cid)
		id = id[0]
		pos_data = test_data[id]
		#print len(pos_data)
		neg_data = []
		for i in range(self.nclass):
			if not zero:
				cur = i + 1
			else:
				cur = i
			if cur == cid:
				continue
			else:
				neg_data.append(test_data[test_labels==cur])
		neg_data = np.vstack(neg_data)

		km = np.random.permutation(neg_data.shape[0])
		neg_data = neg_data[km]

		xz = int(pos_data.shape[0] * self.ratio)
		xz = min(xz, neg_data.shape[0])
		#print xz
		neg_data = neg_data[:xz,:]

		test_data = np.vstack((pos_data, neg_data))
		test_labels = np.vstack((np.ones((pos_data.shape[0],1)), np.zeros((xz,1))))
		#print test_data.shape
		return train_data, train_labels, test_data, test_labels


