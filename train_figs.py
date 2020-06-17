import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

with open('./logger.pkl', 'rb') as fp:
    logger = pickle.load(fp)
max(logger['test_acc'])

logger['']

# plt.figure()
# plt.plot(logger['train_acc'], label='train OA')
# plt.plot(logger['test_acc'], label='test OA')
# plt.legend(loc='lower right')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.title('Overall accuray')
# plt.show()

# plt.figure()
# plt.plot(logger['loss'], label='loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.title('training loss')
# plt.show()

# plt.figure()
# plt.plot(logger['lr'], label='lr')
# plt.xlabel('epoch')
# plt.ylabel('lr')
# plt.title('learning rate')
# plt.show()
