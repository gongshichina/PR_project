from utils import Dataset, evaluation_metrics
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

train_data = Dataset(split='training', oversample=False, binary=True).data
test_data = Dataset(split='testing', binary=True).data

cv_params = {'n_estimators': [400, 500, 600, 700, 800], 'learning_rate': [0.1, 0.01, 0.001]}
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
model = XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(train_data['X'], train_data['y'])
evalute_result = optimized_GBM.grid_scores_
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

# clf = AdaBoostClassifier(n_estimators=100, random_state=0)


# clf.fit()
# y_pred = clf.predict(test_data['X'])
# evaluation_metrics(test_data['y'], y_pred, verbose=True)

# y_pred = clf.predict(train_data['X'])
# evaluation_metrics(train_data['y'], y_pred, verbose=True)



