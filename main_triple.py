from torch.utils.data import DataLoader as Dataloader
from utils import Dataset, Dataset_triple, MultiFocalLoss, Resnet1D, evaluation_metrics
from easydict import EasyDict as edict
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
import torch.nn as nn
from copy import deepcopy
import numpy as np
import os.path as osp
import torch
import pickle
import os

def init_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class Logger(object):
    def __init__(self):
        self.record = {}

    def update(self, dic):
        for k, v in dic.items():
            if k in self.record:
                self.record[k].append(v)
            else:
                self.record[k] = [v]
    
    def write_to_disk(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        with open(osp.join(path, 'logger.pkl'), 'wb') as fp:
            pickle.dump(self.record, fp)

class Model(nn.Module):
    def __init__(self,  model):
        super(Model, self).__init__()
        self.model = model
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.__init_parameters()
        
    def forward(self, x):
        score, x = self.model(x)
        return score, x

    def __init_parameters(self):
        for name, m in self.named_modules():
            if name.split('.')[-1] == 'fc':
                nn.init.orthogonal_(m.weight)
                print('orthogonal!')
            elif isinstance(m, (nn.Conv2d, nn.Linear)):
                # nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight)
                # nn.init.normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # if 'left.5' in name:
                #     nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def db_center(self, loader):
        for i, (X, y) in enumerate(loader):
            X = X.type(FloatTensor)
            with torch.no_grad():
                pred, _ = self.forward(X)
            y_pred.append(pred.cpu().numpy().argmax(axis=-1))
            y_truth.append(y.numpy())
        y_pred = np.concatenate(y_pred, axis=0)
        y_truth = np.concatenate(y_truth, axis=0)
        metric = evaluation_metrics(y_truth, y_pred, verbose=verbose)


    def eval_model(self, loader, use_cuda=True, verbose=False):
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.eval()
        y_pred = []
        y_truth = []
        
        for i, (X, y) in enumerate(loader):
            X = X.type(FloatTensor)
            with torch.no_grad():
                pred, _ = self.forward(X)
            y_pred.append(pred.cpu().numpy().argmax(axis=-1))
            y_truth.append(y.numpy())
        y_pred = np.concatenate(y_pred, axis=0)
        y_truth = np.concatenate(y_truth, axis=0)
        metric = evaluation_metrics(y_truth, y_pred, verbose=verbose)
        self.train()
        return metric

    def save_model(self, check_dir, name):
        with open(osp.join(check_dir, name), 'wb') as fp:
            pickle.dump(self, fp)
    
    def orthogonal_loss(self, param):
        assert param.shape[0] == 9
        eye = torch.eye(9).cuda()
        matrix = torch.matmul(param, param.transpose(1, 0))
        return torch.norm(matrix - eye, p='fro')

    def cluster_loss(self, param):
        pass

    def train_model(self, train_conf):
        
        optimzer = train_conf['optimizer']
        max_epochs = train_conf['max_epochs']
        loader = train_conf['dataloader']
        loss_function = train_conf['lossf']
        use_cuda = train_conf['use_cuda']
        eval_loader = train_conf['test_dataloader']
        lr_scheduler = train_conf['lr_scheduler']
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
        if use_cuda:
            self.cuda()
        best_model = None
        best_acc = -1
        LEN = len(loader)
        logger = Logger()
        
        if train_conf.resume is not None:
            with open(train_conf.resume, 'rb') as fp:
                resume_model = pickle.load(fp)
                self.model = resume_model.model.cuda()
                # self.model.fcs.fc.weight.requires_grad = False
                # self.model.layers[0][0].left[0].weight.requires_grad = False
                print('-- eval resume model --')
                self.eval_model(eval_loader, verbose=True)
        for epoch in range(max_epochs):
            loss_epoch = 0.
            for i, (X, y) in enumerate(loader):
                batch_size = X[0].shape[0]
                X = torch.cat(X, dim=0).type(FloatTensor)
                y = torch.cat(y, dim=0).type(LongTensor)
                y_pred, features = self.forward(X)
                # ce_loss = loss_function(y_pred, y)
                triple_loss = self.triplet_loss(features[:batch_size], features[batch_size:2*batch_size], features[2*batch_size:])
                # orth_loss= self.orthogonal_loss(self.model.fcs.fc.weight)
                # loss = ce_loss + orth_loss
                loss =  triple_loss
                optimzer.zero_grad()
                loss.backward()
                optimzer.step()
                lr_scheduler.step(epoch+i/LEN)
                loss_epoch += loss.detach().cpu().item()
                # print(r'CE loss {0:.3f}, Orth loss {1: .3f}'.format(ce_loss.item(), orth_loss.item()))

            print('epoch %d loss %.5f' % (epoch, loss_epoch / loader.__len__()))
            print('epoch %d lr   %.5f' % (epoch, lr_scheduler.get_lr()[0]))
            # train_metric = self.eval_model(loader)
            # training_acc = train_metric['OA']
            # print('epoch %d train acc %.3f' % (epoch, training_acc))
            print('--- val ---')
            val_metric = self.eval_model(eval_loader, verbose=True)
            print('=='*10)
            logger.update({
                            'test_acc':val_metric['OA'],
                            'loss':loss_epoch / loader.__len__(),
                            'lr':lr_scheduler.get_lr()[0],
                            'val_metric':val_metric})

            if val_metric['OA'] > best_acc:
                best_model = deepcopy(self)
                self.save_model('./', conf.save_model_name)
                best_acc = val_metric['OA']
        print(best_acc)
        logger.write_to_disk('./')

        return best_model

if __name__ == "__main__":
    init_seed(0)
    # resnet = Resnet1D_bottleneck(channels=[128, 256, 512, 512], blocks=(1, 1, 1, 1))
    resnet = Resnet1D(channels=[128, 256, 512, 512], blocks=(1, 1, 1, 1), num_class=9)
    model = Model(resnet)

    conf = edict()
    conf.max_epochs = 1000
    conf.use_cuda = True
    conf.save_model_name = 'model_new_2.pkl' # weight decay
    conf.resume = None
    conf.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-4)
    # conf.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=2e-4)
    conf.lr_scheduler = CosineAnnealingWarmRestarts(conf.optimizer, T_0=100, T_mult=1, eta_min=1e-5) #
    # conf.lr_scheduler = StepLR(conf.optimizer, 1, 0.95) # 41-- 0.001
    conf.dataloader = Dataloader(Dataset_triple(split='training', n_triple=1000), batch_size=512, shuffle=True)
    conf.test_dataloader = Dataloader(Dataset(split='testing', binary=False), batch_size=4096, shuffle=False)
    # conf.lossf = nn.CrossEntropyLoss(weight=torch.tensor([1]+[0.05]*8).cuda())
    conf.lossf = nn.CrossEntropyLoss()
    best_model = model.train_model(conf)