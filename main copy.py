from torch.utils.data import DataLoader as Dataloader
from utils import Dataset, MultiFocalLoss, Resnet1D, evaluation_metrics
from easydict import EasyDict as edict
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
import torch.nn as nn
from copy import deepcopy
import numpy as np
import os.path as osp
import torch
import pickle

def init_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class Model(nn.Module):
    def __init__(self,  model):
        super(Model, self).__init__()
        self.model = model
        self.__init_parameters()
        

    def __init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

    def eval_model(self, loader, use_cuda=True, verbose=False):
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.eval()
        y_pred = []
        y_truth = []
        for i, (X, y) in enumerate(loader):
            X = X.type(FloatTensor)
            pred = self.model.predict(X)
            y_pred.append(pred)
            y_truth.append(y.numpy())
        y_pred = np.concatenate(y_pred, axis=0)
        y_truth = np.concatenate(y_truth, axis=0)
        metric = evaluation_metrics(y_truth, y_pred, verbose=verbose)
        self.train()

        return metric

    def save_model(self, check_dir, name):
        with open(osp.join(check_dir, name), 'wb') as fp:
            pickle.dump(self, fp)
            
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

        if train_conf.resume is not None:
            with open(train_conf.resume, 'rb') as fp:
                resume_model = pickle.load(fp)
                self.model = resume_model.cuda()

        for epoch in range(max_epochs):
            loss_epoch = 0.
            for i, (X, y) in enumerate(loader):
                X = X.type(FloatTensor)
                y = y.type(LongTensor)
                y_pred, y_pred_aux, fea = self.model(X)
                
                mask_other = y > 0
                y_aux = LongTensor(y.shape[0]).zero_()
                y_aux[mask_other] = 1

                y_other = y[mask_other] - 1
                y_pred_other = y_pred[mask_other]
                loss1 = loss_function(y_pred_aux, y_aux)
                loss2 = loss_function(y_pred_other, y_other)
                # print(loss1)
                # print(loss2)
                loss = 16 * loss1  +  loss2
                    
                optimzer.zero_grad()
                loss.backward()
                optimzer.step()
                lr_scheduler.step(epoch+i/LEN)
                loss_epoch += loss.detach().cpu().item()
            print('epoch %d loss %.5f' % (epoch, loss_epoch / loader.__len__()))
            print('epoch %d lr   %.5f' % (epoch, lr_scheduler.get_lr()[0]))
            
            train_metric = self.eval_model(loader)
            training_acc = train_metric['OA']
            if training_acc > best_acc:
                best_model = deepcopy(self)
                self.save_model('./', conf.save_model_name)
            print('epoch %d train acc %.3f' % (epoch, training_acc))
            print('--- val ---')
            val_metric = self.eval_model(eval_loader, verbose=True)
            print('=='*10)

        return best_model


if __name__ == "__main__":
    init_seed(0)
    resnet = Resnet1D(channels=[64, 128, 256], blocks=(1, 1, 1))
    model = Model(resnet)
    
    conf = edict()
    conf.max_epochs = 200
    conf.use_cuda = True
    conf.save_model_name = 'model_st.pkl'
    conf.resume = None
    conf.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-4)
    # conf.lr_scheduler = CosineAnnealingWarmRestarts(conf.optimizer, T_0=25, T_mult=1, eta_min=1e-5) # model_dr0.5
    conf.lr_scheduler = StepLR(conf.optimizer, 25, 0.5)  # model_st
    conf.dataloader = Dataloader(Dataset(split='training', oversample=False), batch_size=512, shuffle=True)
    conf.test_dataloader = Dataloader(Dataset(split='testing'), batch_size=4096, shuffle=False)
    conf.lossf = nn.CrossEntropyLoss()
    best_model = model.train_model(conf)
    