import pickle
import matplotlib.pyplot as plt
import seaborn as sns

with open('data/data.pkl', 'rb') as fp:
    data = pickle.load(fp)
X_train = data['X_train']

for i in range(103):
    x = X_train[:, i]
    plt.figure()
    plt.hist(x, 1000, density=True, color='blue')
    # plt.tight_layout()
    plt.title('Feature %d' % i)
    plt.xlabel('X')
    plt.ylabel('density')
    plt.savefig('./figs/Feature_%d.svg' % i)

y_train = data['y_train']
plt.figure()
n, bin, _ = plt.hist(y_train, density=False, color='blue')
# plt.tight_layout()
for i,j in zip(bin, n):
    plt.text(i, j+1, '%.2f' % j, ha='center', va='bottom')
plt.title('Counts of classes')
plt.xlabel('class')
plt.ylabel('count')
plt.savefig('./figs/label.png')
