from torch.utils.data import DataLoader as Dataloader
from sklearn.manifold import TSNE
from utils import Dataset, evaluation_metrics
import matplotlib.pyplot as plt
from main import Model
import torch
import pickle
import numpy as np
import os
from sklearn.metrics import confusion_matrix

use_cuda = True

with open('./exp/model_True_4_0', 'rb') as fp:
    model = pickle.load(fp)

loader = Dataloader(Dataset(split='testing', oversample=False), batch_size=4096)

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
model.eval() 
y_truth = []
y_pred = []
fea_list = []
for i, (X, y) in enumerate(loader):
    X = X.type(FloatTensor)
    with torch.no_grad():
        pred, fea = model(X)
    y_pred.append(pred.cpu().numpy().argmax(axis=-1))
  
    y_truth.append(y.numpy())
    fea_list.append(fea.cpu().numpy())


y_pred = np.concatenate(y_pred, axis=0)
y_truth = np.concatenate(y_truth, axis=0)
fea_all = np.concatenate(fea_list, axis=0)

array = confusion_matrix(y_truth, y_pred)

np.save('cm.npy', array)

print('-- evaluation --')
metric = evaluation_metrics(y_truth, y_pred, verbose=True)

# select0 = np.random.choice(82312, 10000)
# select = np.concatenate([select0, np.arange(82312, len(y_truth))])
# fea_all /= np.linalg.norm(fea_all, axis=-1, ord=2, keepdims=True)
# center = model.model.fcs.fc.weight.detach().cpu().numpy()
# center = center * np.linalg.norm(fea_all[select], ord=2, axis=-1).mean() / np.linalg.norm(center, ord=2, axis=-1).mean()
# to_fit = np.concatenate([center, fea_all[select]], axis=0)
# X_tsne = TSNE(n_components=2,random_state=33, n_jobs=-1).fit_transform(to_fit)
# center = X_tsne[:9]
# X_tsne = X_tsne[9:]

# ckpt_dir="exp"
# if not os.path.exists(ckpt_dir):
#     os.makedirs(ckpt_dir)
# plt.figure(figsize=(10, 5))
# plt.subplot()
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_truth[select], s=4)
# plt.scatter(center[:, 0], center[:, 1], c=list(range(9)), s=100, marker='o', facecolors='none', edgecolors='r')
# plt.savefig(os.path.join(ckpt_dir, 'tsne-model_False_4_1.svg'), dpi=120)
# plt.show()