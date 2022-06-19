import string
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing #引入所需函式庫
from argparse import ArgumentParser, ArgumentTypeError, Namespace
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataloader import house_price_dataset

from tqdm import tqdm
import ipdb

torch.backends.cudnn.benchmark = True

#----Data preprocess----#

def chunk_cleaning(data_chunk):
    cleaned_chunk = data_chunk.dropna()
    return cleaned_chunk

def dataloader(data_number):

    data_chunks = pd.read_csv("./2017_to_present_七都房屋買賣交易.csv",nrows=1000, chunksize=1000)

    chunk_list = []  #暫存各區塊的處理結果
    for data_chunk in data_chunks:  #讀取各區塊
        cleaned_chunk = chunk_cleaning(data_chunk)  #清理區塊中的遺漏值
        chunk_list.append(cleaned_chunk)  #將清理後的區塊結果暫存在串列(List)中

    combined_chunk = pd.concat(chunk_list)  #將各區塊的結果進行合併

    df = combined_chunk.drop(['交易年月日', '建築完成年月', '主要用途', '建物型態', \
                            '車位類別', '車位總價元', '車位移轉總面積(平方公尺)'], axis=1)
    df['總價元'] = df['總價元'] / 1000.0
    df['單價元平方公尺'] = df['單價元平方公尺'] / 1000.0

    continuous_index = []
    categorical_index = []
    for index, d in enumerate(df.dtypes):
        if (d != "object"):
            continuous_index.append(index)
        else:
            categorical_index.append(index)

    # continuous_features = df.iloc[:, continuous_index]
    # categorical_features = df.iloc[:, categorical_index]
    continuous_features = df.columns[continuous_index]
    categorical_features = df.columns[categorical_index]
    #df_train[pd.isnull(df_train)]  = 'NaN'
    for i in categorical_features: #將轉換是object的傢伙轉換，從object_data陣列一個一個抓出來改
        df[i] = LabelEncoder().fit_transform(df[i].factorize()[0]) 
        #pd.factorize()[0]會給nans(缺失值)一個-1的值，若沒寫這個，會造成等號兩邊不等的情況
    
    """ ToDo
    *   Feature engineering
    *   Correlation matrix
    *   LSTM need based on time series
    """
    train_targets = df["單價元平方公尺"].values #把SalePrice這行數值整個拉出來
    train_data = df.drop(columns=["單價元平方公尺"]) #刪除SalePrice這行

    x_train, x_test, y_train, y_test = train_test_split(train_data, train_targets, \
                                                        test_size=0.2, random_state=1)
    x_train_dataset = x_train.values #取出數值，轉換回list
    x_test_dataset = x_test.values

    normalize = preprocessing.StandardScaler() #取一個短的名字
    # 標準化處理
    x_train_normal_data = normalize.fit_transform(x_train_dataset) #將訓練資料標準化
    x_test_normal_data = normalize.fit_transform(x_test_dataset) #將驗證資料標準化

    test_split = data_number // 4
    x_train_normal_data = x_train_normal_data[:data_number]
    x_test_normal_data = x_test_normal_data[:test_split]
    y_train = y_train[:data_number]
    y_test = y_test[:test_split]

    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    return x_train_normal_data, x_test_normal_data, y_train, y_test


#----Training----#
class MLP_new(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, dropout=0.5):
        super(MLP_new, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(n_feature, n_hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_hidden),
            nn.Dropout(dropout),

            nn.Linear(n_hidden, n_hidden*2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_hidden*2),
            nn.Dropout(dropout),

            nn.Linear(n_hidden*2, n_hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_hidden),
            nn.Dropout(dropout),
            
            nn.Linear(n_hidden, n_hidden//2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_hidden//2),
            nn.Dropout(dropout),

            nn.Linear(n_hidden//2, n_hidden//4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_hidden//4),
            nn.Dropout(dropout),

            nn.Linear(n_hidden // 4, n_hidden // 8),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_hidden // 8),
            nn.Dropout(dropout),

            nn.Linear(n_hidden//8, n_output)
        )

    def forward(self, input):
        x = self.layers(input)
        return x

class MLP(nn.Module):
    '''
      Multilayer Perceptron for regression.
    '''
    def __init__(self, feature_len, hidden_len, num_layers, batch_size, seq_len, device):
        super(MLP, self).__init__()

        # 重要！！！ make sure tensor is to device and with no_grad in evaluation.
        self.feature_len = feature_len
        self.hidden_len = hidden_len
        self.num_layers = num_layers
        self.batch_size = batch_size
        # LSTM layers
        self.LSTM = nn.LSTM(self.feature_len, self.hidden_len, self.num_layers)

        self.hidden = (Variable(torch.zeros(num_layers, batch_size, hidden_len).to(device)),
                        Variable(torch.zeros(num_layers, batch_size, hidden_len).to(device)))
        # fully connected layers
        self.layers = nn.Sequential(
            nn.Linear(self.hidden_len, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, input):
        '''
          Forward pass
          input size will be (seq_len(16), batch_size(1), feature_len(11))
        '''
        # ipdb.set_trace()
        input = input.transpose(0, 1)
        h_in, (h, c) = self.LSTM(input, self.hidden)
        h_in = h_in.view(len(h_in), -1)
        return self.layers(h_in)

def MAPE_Loss(pred, target):
    loss = 0
    for i in range(len(pred)):
        loss += (abs((pred[i] - target[i]) / target[i]))
    return loss / len(pred)

def train(x_train, y_train,
          args,
          model,
          optimizer,
          device):
    # training
    # ipdb.set_trace()
    model.train()
    x_train.to(device)
    y_train.to(device)

    optimizer.zero_grad()
    output = model(x_train)
    # ipdb.set_trace()
    loss = nn.MSELoss()(output, y_train)
    # loss = MAPE_Loss(output, y_train)
    loss.backward()
    optimizer.step()

    hits = 0
    for index in range(len(output)):
        if (abs(output[index].item()- y_train[index].item()) <= y_train[index].item() * 0.1):
            hits += 1
    return loss.item()/len(output), hits/len(output)

#----Arguments----#

def check_optimizer_type(input_value: str) -> optim:
    """
    Check whether the optimizer is supported
    :param input_value: input string value
    :return: optimizer
    """
    if input_value == 'sgd':
        return optim.SGD
    elif input_value == 'adam':
        return optim.Adam
    elif input_value == 'adadelta':
        return optim.Adadelta
    elif input_value == 'adagrad':
        return optim.Adagrad
    elif input_value == 'adamw':
        return optim.AdamW
    elif input_value == 'adamax':
        return optim.Adamax

    raise ArgumentTypeError(f'Optimizer {input_value} is not supported.')

def parse_arguments() -> Namespace:
    """
    Parse arguments
    :return: arguments
    """
    parser = ArgumentParser(description='Neural Network Regression')
    parser.add_argument('--city', default='台北市', help='city you want to predict')
    parser.add_argument('-b', '--batch_size', default=12, type=int, help='Number of batch size')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('-w', '--weight_decay', default=5e-4, type=float, help='Weight decay (L2 penalty)')
    parser.add_argument('-e', '--epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('-o', '--optimizer', default='adam', type=check_optimizer_type, help='Optimizer')
    parser.add_argument('--seq_len', default=1, type=int, help='Length of sequence')
    parser.add_argument('--feature_len', default=11, type=int, help='Length of feature')
    parser.add_argument('--hidden_len', default=16, type=int, help='Hidden_length of RNN')
    parser.add_argument('--num_layer', default=8, type=int, help='Layers of RNN')
    parser.add_argument('--num_workers', type=int, default=1, help='number of data loading threads')
    parser.add_argument('--cuda', default=False, action='store_true')

    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'

    train_data = house_price_dataset(args, 'train', device)
    test_data = house_price_dataset(args, 'test', device)
    train_loader = DataLoader(train_data, 
                            # num_workers=args.num_workers,
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            drop_last=True)
    test_loader = DataLoader(test_data,
                            # num_workers=args.num_workers,
                            batch_size=args.batch_size, 
                            shuffle=False,
                            drop_last=True)

    train_iterator = iter(train_loader)
    test_iterator = iter(test_loader)
    # 迴圈要幾次以及區分train test資料
    model = MLP_new(args.feature_len, args.feature_len*8, 1, dropout=0.3)
    # model = MLP(feature_len=args.feature_len, hidden_len=args.hidden_len, num_layers=args.num_layer, \
    #             batch_size=args.batch_size, seq_len=args.seq_len, device=device)
    model.to(device)
    model_optimizer = args.optimizer(model.parameters(),lr = args.learning_rate, \
                        weight_decay = args.weight_decay)

    for epoch in tqdm(range(args.epochs)):
        for _ in tqdm(range(len(train_data)//args.seq_len)):
            try:
                x_train, y_train = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                x_train, y_train = next(train_iterator)

            y_train = y_train.view(-1, 1)
            train_loss, train_hit_rate = train(x_train, y_train,
                                                args,
                                                model,
                                                model_optimizer,
                                                device)
            # print("Loss:{:.2f} Accuracy:{:.2%}".format(loss, acc))

        hits = 0
        for _ in tqdm(range(len(test_data)//args.batch_size)):
            try:
                x_test, y_test = next(test_iterator)
            except StopIteration:
                test_iterator = iter(test_loader)
                x_test, y_test = next(test_iterator)
            # evaluation
            model.eval()
            x_test.to(device)
            y_test.to(device)
            y_test = y_test.squeeze(0)
            with torch.no_grad():
                output = model(x_test)
                # loss = nn.L1Loss()(output, y_test)  # sum up batch loss
                for index in range(len(output)):
                    if (abs(output[index].item()- y_test[index].item()) <= y_test[index].item() * 0.1):
                        hits += 1
        print("")
        print("")
        print("epoch: {:0>2d} Testing hit_rate: {:.2%}\n".format(epoch, hits/len(test_data)))
    return 0

if __name__ == '__main__':
    main()