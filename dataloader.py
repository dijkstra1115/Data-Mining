import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing #引入所需函式庫

import ipdb

class house_price_dataset(Dataset):
    def __init__(self, args, mode='train', device='cpu'):
        if mode == 'train':
            x_csv = pd.read_csv("./data/train/x_train_%s.csv" % (args.city))
            y_csv = pd.read_csv("./data/train/y_train_%s.csv" % (args.city))
        else:
            x_csv = pd.read_csv("./data/test/x_test_%s.csv" % (args.city))
            y_csv = pd.read_csv("./data/test/y_test_%s.csv" % (args.city))

        x_dataset = x_csv.values
        y_dataset = y_csv.values
        # ipdb.set_trace()
        x_normalize = preprocessing.normalize(x_dataset, norm='l1')
        # y_normalize = preprocessing.normalize(y_dataset, norm='l2')

        # normalize = preprocessing.StandardScaler() #取一個短的名字
        # # 標準化處理
        # x_normal_data = normalize.fit_transform(x_dataset) #將訓練資料標準化
        # y_normal_data = normalize.fit_transform(y_dataset) #將訓練資料標準化

        # self.Augmentation = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5), (0.5))
        # ])
        self.x_data = x_normalize
        # self.y_data = y_normalize
        self.y_data = y_dataset

        self.d = 0
        self.seq_len = args.seq_len
        self.mode = mode
        self.device = device

        args.feature_len = x_dataset.shape[1]
            
    def __len__(self):
        
        return len(self.x_data)
    
    def get_data(self, index):
        # x_data = self.x_data[self.d:self.d+self.seq_len]
        x_data = self.x_data[index]
        x_data = torch.FloatTensor(x_data).to(self.device)
        # x_data = self.Augmentation(x_data)
        # y_data = self.y_data[self.d:self.d+self.seq_len]
        y_data = self.y_data[index]
        y_data = torch.FloatTensor(y_data).to(self.device)
        # y_data = self.Augmentation(y_data)
        # self.d += self.seq_len
        # if self.d >= (len(self.x_data)-self.seq_len-1):
        #     self.d = 0
        
        return x_data.squeeze(0).float(), y_data.squeeze(0).float()
    
    def __getitem__(self, index):

        x_data, y_data =  self.get_data(index)

        return x_data, y_data