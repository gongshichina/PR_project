import scipy.io as io
import numpy as np
import os.path as osp
import pickle
root='./data'
CLASS_NUM = 9
test_file = osp.join(root, 'test_data.mat')
train_file = osp.join(root, 'train_data.mat')
test_data, train_data = list(map(io.loadmat, [test_file, train_file]))

train_samples_num = [train_data['yidali_train'][0,i].shape[1] for i in range(CLASS_NUM)]
test_samples_num = [test_data['yidali_test'][0,i].shape[1] for i in range(CLASS_NUM)]

train_data = np.concatenate([train_data['yidali_train'][0, i] for i in range(CLASS_NUM)], axis=1).T
test_data = np.concatenate([test_data['yidali_test'][0, i] for i in range(CLASS_NUM)], axis=1).T

y_test = [[i]*test_samples_num[i] for i in range(CLASS_NUM)]
y_test = [i for j in y_test for i in j]

y_train = [[i]*train_samples_num[i] for i in range(CLASS_NUM)]
y_train = [i for j in y_train for i in j]
data = {'X_train': train_data.astype('float') / 8000.,
        'X_test': test_data.astype('float') / 8000.,
        'y_train':np.array(y_train),
        'y_test':np.array(y_test)
}

with open(osp.join(root, 'data.pkl'), 'wb') as fp:
    pickle.dump(data, fp)