import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from PIL import Image

class NewMnist(MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, cid=1, ratio=0.2):
        super(NewMnist, self).__init__(root, train, transform, target_transform, download)
        '''c is the positive class'''
        self.c = cid-1

        if self.train:
            # split train data
            pos_idx = self.train_labels.eq(self.c).nonzero().squeeze()
 
            self.train_data = self.train_data[pos_idx]
            self.train_labels = torch.ones(pos_idx.nelement(),).long()
            
        else:
            # split test data
            pos_idx = self.test_labels.eq(self.c).nonzero().squeeze()
            nneg_ = int(pos_idx.nelement()* ratio)
            neg_idx = []

            for i in range(10):
                if i == self.c:
                    continue
                neg_idx.append(self.test_labels.eq(i).nonzero().squeeze())
            neg_idx = torch.cat(neg_idx)
            kp = torch.randperm(neg_idx.size(0))
            neg_idx = neg_idx[kp]
            
            self.test_data = torch.cat((self.test_data[pos_idx], self.test_data[neg_idx[:nneg_]]), dim=0)
            self.test_labels = torch.cat((torch.ones((pos_idx.nelement())), torch.zeros((nneg_))))
            
            kp = torch.randperm(self.test_data.size(0))
            self.test_data = self.test_data[kp]
            self.test_labels = self.test_labels[kp]

    def __len__(self):
        if self.train:
            return self.train_labels.nelement()
        else:
            return self.test_labels.nelement()

    def get_batch(self, batch_size):
        
        kp = torch.randperm(len(self))
        if self.train:
            img = self.train_data[kp[:batch_size]]
            target = self.train_labels[kp[:batch_size]]
            
        else:
            img = self.test_data[kp[:batch_size]]
            target = self.test_labels[kp[:batch_size]]
        
        imgs = []
        targets = []
        for i in range(img.size(0)):
            img_tmp = Image.fromarray(img[i].numpy(), mode='L')

            if self.transform is not None:
                img_tmp = self.transform(img_tmp)

            target_tmp = target[i]
            if self.target_transform is not None:
                target_tmp = self.target_transform(target_tmp)
            imgs.append(img_tmp)
            targets.append(target_tmp)

        imgs = torch.cat(imgs, 0)
        targets = torch.FloatTensor(targets).long()
        return imgs, targets