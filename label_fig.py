import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
import numpy as np
with open('data/data.pkl', 'rb') as fp:
    data = pickle.load(fp)

y_train = data['y_train']
number = Counter(y_train)

s1 = pd.DataFrame({
    'class': {k:k+1 for k in range(0, 9)},
    'count': dict(number)
})

ax = sns.barplot(x="class", y="count", data=s1, palette="pastel")   #seaborn画柱状图

plt.xticks(fontsize=8)          #设置x和y轴刻度值字体大小
plt.yticks(fontsize=8)

# plt.yticks(np.arange(0, number[0]+3000, 3000))   #设置y轴标签

plt.xlabel("class", fontsize=8)  #设置x轴和y轴标签字体大小
plt.ylabel("count", fontsize=8)

x = np.arange(len(s1["class"]))    #在柱状图上插入数值
y = np.array(list(s1["count"]))
for a,b in zip(x,y):
    plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=8)
plt.title('Count of classes')
plt.savefig('./figs/label.svg')

plt.show()