import torch 
import math
from scipy import linalg
import utils.common as common

class KernelPCA:

	def __init__(self, ncomponent=10, kernel='gauss', sigma=0.1, alpha=1.0, fit_inverse_transform=True, cuda=False):
		self.kerneltype = kernel
		self.sigma = sigma
		self.n_component = ncomponent
		self.fit_inverse_transform = fit_inverse_transform
		self.cuda = cuda
		self.alpha = alpha

	def _centering(self, K):
		n_samples = K.size(0)
		m_samples = K.size(1)
		norm1 = torch.ones(n_samples, n_samples) / n_samples
		norm2 = torch.ones(m_samples, m_samples) / m_samples
		Kc = K - torch.mm(norm1, K) - torch.mm(K, norm2) + torch.mm(torch.mm(norm1, K), norm2)
		return Kc


	def _get_kernel(self, X, Y=None):
		if Y is None:
			K = common.pairwise_distance(X, X, self.kerneltype, dict(sigma=self.sigma))
		else:
			K = common.pairwise_distance(X, Y, self.kerneltype, dict(sigma=self.sigma))
		return K


	def _fit_transform(self, K):
		K = self._centering(K)
		if self.n_component is None:
			n_component = K.size(0)
		else:
			n_component = min(K.size(0), self.n_component)
		# compute eigenvectors
		V, E = torch.eig(K, True)

		v = V[:,0]
		v = v[: n_component]
		
		E = E[:, :n_component]
		#for i in range(n_component):
		#	E[:, i] = E[:,i] / math.sqrt(n*v[i])

		self.alphas_ = E 
		self.lambdas_ = v
		return K

	def fit(self, X, y=None):
		"""Fit the model from data in X.

		Parameters
		----------
		X: array-like, shape (n_samples, n_features)
			Training vectors, where n_samples is the number  of samples
			and n_features is the number of features.

		Returns
		-------
		self: object
			Returns the instance itself.
		"""
		K = self._get_kernel(X, X)

		self._fit_transform(K)
		
		if self.fit_inverse_transform:
			#print('inverse')
			sqrt_lambdas = torch.diag(torch.sqrt(self.lambdas_))
			X_transformed = torch.mm(self.alphas_, sqrt_lambdas)
			self._fit_inverse_transform(X_transformed, X)
		
		self.X_fit_ = X
		return self


	def _fit_inverse_transform(self, X_transformed, X):
		n_samples = X_transformed.size(0)
		K = self._get_kernel(X_transformed)
		K += torch.eye(n_samples) * self.alpha
		dual_coef = linalg.solve(K.cpu().numpy(), X.cpu().numpy(), sym_pos=True, overwrite_a=True)
		self.dual_coef_ = torch.from_numpy(dual_coef)
		if self.cuda:
			self.dual_coef_ = self.dual_coef_.cuda()
		self.X_transformed_fit_ = X_transformed


	def fit_transform(self, X, y=None, **params):
		"""Fit the model from data in X and transform X.
		"""
		self.fit(X)

		X_transformed = self.alphas_ #torch.mm(self.alphas_, 
		#if self.fit_inverse_transform:
		#	self._fit_inverse_transform(X_transformed, X)

		return X_transformed


	def inverse_transform(self, X):
		"""Transform X back to original space."""
		if not self.fit_inverse_transform:
			print('Not fit inverse transform')
		K = self._get_kernel(X, self.X_transformed_fit_)
		return torch.mm(K, self.dual_coef_)


	def transform(self, X):

		K = self._get_kernel(X, self.X_fit_)

		return torch.mm(K, self.alphas_ / self.lambdas_)




if __name__=='__main__':
	import matplotlib.pyplot as plt
	import numpy as np
	#print E,v

	from sklearn.datasets import make_circles
	X, y = make_circles(n_samples=1000, noise=.1, factor=.2, random_state=123)
	

	pca = KernelPCA(ncomponent=2, kernel='gauss', sigma=0.25, cuda=False)
	X_ = torch.from_numpy(X).float()

	X_kpca = pca.fit_transform(X_)
	X_kpca = X_kpca.numpy()

	fig, ax = plt.subplots(1, 3, figsize=(8, 4))
	ax[0].scatter(X[y==0, 0], X[y==0, 1], color='r', marker='^', alpha=.4)
	ax[0].scatter(X[y==1, 0], X[y==1, 1], color='b', marker='o', alpha=.4)
	

	ax[1].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='r', marker='^', alpha=.4)
	ax[1].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='b', marker='o', alpha=.4)

	label_count = np.bincount(y)

	ax[2].scatter(X_kpca[y==0, 0], np.zeros(label_count[0]), color='r')
	ax[2].scatter(X_kpca[y==1, 0], np.zeros(label_count[1]), color='b')

	ax[2].set_ylim([-1, 1])
	ax[0].set_xlabel('PC1')
	ax[0].set_ylabel('PC2')
	ax[1].set_xlabel('PC1')
	ax[2].set_xlabel('PC1')

	

	plt.show()