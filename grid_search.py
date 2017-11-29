import numpy as np
from numpy import genfromtxt, linalg
import sklearn
from sklearn import preprocessing, ensemble, feature_selection, model_selection
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

data_file = 'new/train_2008_back_to_basics_selected_feats.csv'
data = genfromtxt(data_file, delimiter=',')
print("Data loaded from " + data_file + '...')

# Trim the first row which contains header information
# and the first three columns which contain irrelevant data
data = data[1:, :]
x_train = data[:, 0:-1]
y_train = data[:, -1]

# Scale data to have 0 mean and unit variance
'''
scaler = sklearn.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
print("Data trimmed and rescaled.")
'''
'''
clf = ensemble.ExtraTreesClassifier()
clf = clf.fit(x_train, y_train)
model = feature_selection.SelectFromModel(clf, prefit=True)
x_train = model.transform(x_train)
print(np.shape(x_train))
'''

print("Begin training Gradient Boosting...")
rf = ensemble.GradientBoostingClassifier()
#rf.fit(x_train2, y_train)

parameter_grid = {
                 'n_estimators': [800, 1000, 1200],
                 'min_samples_leaf' : [14, 15, 16, 17],
                 #'min_samples_split' : [14, 16, 18, 20, 22],
                 #'learning_rate': [0.05],
                 #'subsample': [0.8, 0.9, 1.0]
                 #'criterion': ['gini', 'entropy']
                 }

grid_search = GridSearchCV(rf,
                           param_grid=parameter_grid,
                           verbose=1, cv=3)

print("Beginning grid search over parameters...")
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))







