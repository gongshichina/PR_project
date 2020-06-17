import numpy as np
import os.path as osp
import pickle
import torch
import math
import torch.nn as nn
from collections import Counter, OrderedDict
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt



def evaluation_metrics(y_true, y_pred, verbose=False):
    """
    Input:
        y_true : array, e.g. [0, 1, 4, 6]
        y_pred : array, e.g. [0, 1, 1, 2]
    Output:
        {"kappa": value1,
        "OA": value2,
        "AA": value3}
    """

    kappa_value = cohen_kappa_score(y_true, y_pred)
    oa = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    aa = cm.diagonal().mean()

    metric_dict = {
        "kappa": kappa_value,
        "OA": oa,
        "AA": aa
    }
    if verbose:
        for k in metric_dict.keys():
            print(r"{0:<10}:{1:.3f}".format(k, metric_dict[k]))
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        labels = ['class %d' for i in range(1, 10)]
        fig = plt.figure()
        df_cm = DataFrame(confm, index=labels, columns=labels)

        ax = sn.heatmap(df_cm, cmap='Oranges', annot=True)

    return metric_dict

class ResidualBlock(nn.Module):
    '''
    实现子module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
                nn.Linear(inchannel, outchannel, bias=True),
                nn.BatchNorm1d(outchannel),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
                nn.Linear(outchannel, outchannel, bias=True),
                nn.BatchNorm1d(outchannel)
                )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return torch.relu(out)

class Resnet1D(nn.Module):
    def __init__(self, channels=[16, 32, 64], indim=103, num_class=9,  blocks=None):
        super().__init__()
        self.layers = []
        in_channel = indim
        for block_num, out_channel in zip(blocks, channels):
            self.layers.append(self._make_layer(in_channel, out_channel, block_num))
            in_channel = out_channel
        self.layers = nn.Sequential(*self.layers)
        
        self.fcs = OrderedDict()
        in_dim = channels[-1]
        self.fcs.update(
            {
                "fc": nn.Linear(in_dim, num_class)
            }
        )
        self.fcs = nn.Sequential(self.fcs)

    def forward(self, x, label=None):
        assert x.dim() == 2
        x = self.layers(x)
        out = self.fcs(x)
        return out, x

    def _make_layer(self,  inchannel, outchannel, block_num):
        '''
        构建layer,包含多个residual block
        '''
        shortcut = nn.Sequential(
                nn.Linear(inchannel,outchannel, bias=True),
                nn.BatchNorm1d(outchannel)
                )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

class ResidualBlock_bottleneck(nn.Module):
    '''
    实现子module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, shortcut=None):
        super(ResidualBlock_bottleneck, self).__init__()
        self.left = nn.Sequential(
                nn.Linear(inchannel, outchannel, bias=True),
                # nn.BatchNorm1d(outchannel),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(outchannel, outchannel, bias=True),
                nn.BatchNorm1d(outchannel) 
                )
        self.bottlneck = nn.Sequential(nn.Linear(2*outchannel, outchannel),
                                        nn.BatchNorm1d(outchannel),
                                        nn.ReLU())
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out = torch.cat([out, residual], dim=-1)
        out = self.bottlneck(out)
        return out

# ArcFace
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        # Parameter 的用途：
        # 将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并将这个parameter绑定到这个module里面
        # net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的
        # https://www.jianshu.com/p/d8b77cc02410
        # 初始化权重
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # torch.nn.functional.linear(input, weight, bias=None)
        # y=x*W^T+b
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # cos(a+b)=cos(a)*cos(b)-size(a)*sin(b)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            # torch.where(condition, x, y) → Tensor
            # condition (ByteTensor) – When True (nonzero), yield x, otherwise yield y
            # x (Tensor) – values selected at indices where condition is True
            # y (Tensor) – values selected at indices where condition is False
            # return:
            # A tensor of shape equal to the broadcasted shape of condition, x, y
            # cosine>0 means two class is similar, thus use the phi which make it
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # 将cos(\theta + m)更新到tensor相应的位置中
        one_hot = torch.zeros(cosine.size(), device='cuda')

        # scatter_(dim, index, src)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

class Resnet1D_bottleneck(nn.Module):
    def __init__(self, channels=[16, 32, 64], indim=103, num_class=9,  blocks=None):
        super().__init__()
        self.layers = []
        in_channel = indim
        for block_num, out_channel in zip(blocks, channels):
            self.layers.append(self._make_layer(in_channel, out_channel, block_num))
            in_channel = out_channel
        self.layers = nn.Sequential(*self.layers)
        
        self.fcs = OrderedDict()
        in_dim = channels[-1]
        self.laynorm =  nn.LayerNorm((in_dim,))
        self.fcs.update(
            {
                "fc": nn.Linear(in_dim, num_class, bias=False)
            }
        )
        self.fcs = nn.Sequential(self.fcs)

    def forward(self, x):
        assert x.dim() == 2
        x = self.layers(x)
        out = self.fcs(x)
        return out, x

    def _make_layer(self,  inchannel, outchannel, block_num):
        '''
        构建layer,包含多个residual block
        '''
        shortcut = nn.Sequential(
                nn.Linear(inchannel,outchannel, bias=True),
                nn.BatchNorm1d(outchannel)
                )

        layers = []
        layers.append(ResidualBlock_bottleneck(inchannel, outchannel, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock_bottleneck(outchannel, outchannel))
        return nn.Sequential(*layers)

class Dataset(object):
    def __init__(self, split='training', root='./data', oversample=True, binary=False, transform=None):
        self.split = split
        self.oversample = oversample
        self.binary = binary
        self.data = self.load_data(root, split)
        
        self.transform = transform

    def __getitem__(self, idx):
        return self.data['X'][idx], self.data['y'][idx]

    def __len__(self):
        return self.data['y'].__len__()

    def load_data(self, root, split):
        fp = open(osp.join(root, 'data.pkl'), 'rb')
        data = pickle.load(fp)
        fp.close()
        ret = {}
        self.mean = data['X_train'].mean(axis=0)
        self.std = data['X_train'].std(axis=0)

        if split == 'training':
            if self.binary:
                mask = data['y_train'] != 0
                data['y_train'][mask] = 1
            ret['X'] = (data.pop('X_train') - self.mean) / self.std
            ret['y'] = data.pop('y_train') 
            if self.oversample:
                from imblearn.over_sampling import SMOTE
                sm = SMOTE(random_state=23, k_neighbors=5)
                ret['X'], ret['y'] = sm.fit_resample(ret['X'], ret['y'])
            
                print('Load train data success')
                print(Counter(ret['y']))

        elif split == 'testing':
            if self.binary:
                mask = data['y_test'] != 0
                data['y_test'][mask] = 1
            ret['X'] = torch.from_numpy((data.pop('X_test') - self.mean) / self.std)
            ret['y'] = torch.from_numpy(data.pop('y_test'))
            print('Load test data success')
            print(Counter(ret['y'].numpy()))
        else:
            raise ValueError('Please set correct phase (training / testing)')
        del data
        
        return ret

class Dataset_triple(Dataset):
    def __init__(self, split='training', root='./data', n_triple=10000):
        super().__init__(split=split, root=root, oversample=False)
        self.n_triple = n_triple
        self.number = self.get_point()

    def __getitem__(self, idx):
        an_idx, pos_idx, neg_idx = self.get_triple_idx()
        return (self.data['X'][an_idx], self.data['X'][pos_idx], self.data['X'][neg_idx]), \
                (self.data['y'][an_idx], self.data['y'][pos_idx], self.data['y'][neg_idx])

    def get_triple_idx(self):
        an_idx = np.random.randint(0, 103224)
        c = self.data['y'][an_idx]
        pos_idx = np.random.randint(*self.number[c])
        if pos_idx == an_idx:
            pos_idx = np.random.randint(*self.number[c])
        neg_list = list(range(0, self.number[c][0])) + list(range(self.number[c][1], 103224))
        neg_idx = np.random.choice(neg_list)
        return an_idx, pos_idx, neg_idx

    def __len__(self):
        return self.n_triple

    def get_point(self):
        return {0 : (0, 82312),
                1 : (82312, 85627),
                2 : (85627, 87159),
                3 : (87159, 89000),
                4 : (89000, 90049),
                5 : (90049, 90714),
                6 : (90714, 100038),
                7 : (100038, 100710),
                8 : (100710, 103224)}

class MultiFocalLoss(nn.Module):
    """
    Reference : https://www.zhihu.com/question/367708982/answer/985944528
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = torch.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


