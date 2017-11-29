import numpy as np
from numpy import genfromtxt, linalg
import sklearn
from sklearn import preprocessing, ensemble, feature_selection
from matplotlib import pyplot as plt
from sklearn.externals import joblib

data = genfromtxt('train_2008.csv', delimiter=',')

# Trim the first row which contains header information 
# and the first three columns which contain irrelevant data 
data = data[1:, 3:]
x_train = data[:, 0:-1]
y_train = data[:, -1]

test_data = genfromtxt('test_2008.csv', delimiter=',')
x_test = test_data[1:, 3:]



# Scale data to have 0 mean and unit variance 
scaler = sklearn.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

clf = ensemble.ExtraTreesClassifier(n_estimators=800, min_samples_leaf=5)
clf = clf.fit(x_train, y_train)
model = feature_selection.SelectFromModel(clf, prefit=True, threshold='mean')
x_train = model.transform(x_train)
x_test = model.transform(x_test)


# Train Gradient Boosting Classifier
print("Begin training Gradient Boosting...")
rf = ensemble.GradientBoostingClassifier(n_estimators=1200, min_samples_leaf=19)
rf.fit(x_train, y_train)

trng_acc = rf.score(x_train, y_train)

print("leaf size: %d" %leaf_size)
print("estimators: %d" %num_e)
print("Training accuracy: %f" %trng_acc)

prediction = rf.predict(x_test)

np.savetxt("prediction.csv", prediction)

