import numpy as np
import os.path as osp
import pickle
import torch
import numpy as np
import torch.nn as nn
from collections import Counter

from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

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

    return metric_dict

class ResidualBlock(nn.Module):
    '''
    实现子module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
                nn.Linear(inchannel, outchannel, bias=True),
                nn.Dropout(p=0.5),
                nn.BatchNorm1d(outchannel),
                nn.ReLU(inplace=True),
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
    def __init__(self, channels=[16, 32, 64], indim=103, num_class=9,  blocks=None, aux=False):
        super().__init__()
        self.layers = []
        self.aux = aux
        in_channel = indim
        for block_num, out_channel in zip(blocks, channels):
            self.layers.append(self._make_layer(in_channel, out_channel, block_num))
            in_channel = out_channel
        self.layers = nn.Sequential(*self.layers)

        self.fcs = []
        in_dim = channels[-1]
        self.fcs.append(nn.Linear(in_dim, num_class-1))
        self.fcs = nn.Sequential(*self.fcs)
        self.fc_aux = nn.Linear(in_dim, 2)


    def forward(self, x_):
        assert x_.dim() == 2
        x = self.layers(x_)
        out = self.fcs(x)
        out_aux = self.fc_aux(x)
        return out, out_aux, x

    def predict(self, x_):
        self.eval()
        with torch.no_grad():
            out, out_aux, _ = self.forward(x_)
            out = out.cpu().numpy()
            out_aux = out_aux.cpu().numpy()
        pred = np.zeros(x_.shape[0])
        m = out_aux[:, 1]>=0.5
        pred[m] = out[m].argmax(axis=-1)+1
        return pred

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

class Dataset(object):
    def __init__(self, split='training', root='./data', transform=None, oversample=True, aux=True):
        self.split = split
        self.oversample = oversample
        self.aux = aux
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
            ret['X'] = (data.pop('X_train') - self.mean) / self.std
            ret['y'] = data.pop('y_train') 
            if self.oversample:
                from imblearn.over_sampling import SMOTE
                sm = SMOTE(random_state=23, k_neighbors=5)
                ret['X'], ret['y'] = sm.fit_resample(ret['X'], ret['y'])
                print('Load train data success')
                print(Counter(ret['y']))

        elif split == 'testing':
            ret['X'] = torch.from_numpy((data.pop('X_test') - self.mean) / self.std)
            ret['y'] = torch.from_numpy(data.pop('y_test'))
            print('Load test data success')
            print(Counter(ret['y'].numpy()))
        else:
            raise ValueError('Please set correct phase (training / testing)')
        del data
        
        return ret

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


