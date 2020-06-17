from collections import Counter
from imblearn.over_sampling import SMOTE
from utils import Dataset

data = Dataset(split='training', oversample=False).data
X = data['X']
y = data['y']

sm = SMOTE(random_state=42, k_neighbors=5)
sm.fit_resample(X, y)