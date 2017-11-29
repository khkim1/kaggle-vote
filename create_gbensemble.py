import numpy as np
from numpy import genfromtxt, linalg
import sklearn
from sklearn import preprocessing, ensemble, neural_network, neighbors, feature_selection
from matplotlib import pyplot as plt
from sklearn.externals import joblib

data = genfromtxt('train_2008.csv', delimiter=',')

# Trim the first row which contains header information 
# and the first three columns which contain irrelevant data 
data = data[1:, 3:]
x_train = data[:, 0:-1]
y_train = data[:, -1]

partition = 55000

x_test = x_train[partition:, :]
y_test = y_train[partition:]
#y_test[y_test == 2] = -1

x_train = x_train[0:partition, :]
y_train = y_train[0:partition]
#y_train[y_train == 2] = -1

# Scale data to have 0 mean and unit variance 
scaler = sklearn.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

clf = ensemble.ExtraTreesClassifier()
clf = clf.fit(x_train, y_train)
model = feature_selection.SelectFromModel(clf, prefit=True)
x_train_use = model.transform(x_train)
x_test_use = model.transform(x_test)
print(np.shape(x_train_use))


gb_pred = []
max_acc = 0
# Train Gradient Boosting Classifier
for idx in range(20):

	print("Begin training Gradient Boosting...")
	clf = ensemble.GradientBoostingClassifier(n_estimators=500, min_samples_leaf=19)
	clf.fit(x_train_use, y_train)
	trng_acc = clf.score(x_train_use, y_train)
	val_acc = clf.score(x_test_use, y_test)
	if val_acc > max_acc:
		max_acc = val_acc
		max_idx = idx
	gb_pred.append(clf.predict(x_test_use))
	print("Index: %d" %idx)
	print("Training accuracy: %f" %trng_acc)
	print("Validation accuracy: %f" %val_acc)
	joblib.dump(clf, "gb" + str(idx) + ".pkl")

'''
ensemble_pred = rf1_pred + rf2_pred + ada_pred + et1_pred + et2_pred + 5*gb_pred + nn_pred
ensemble_pred[ensemble_pred >= 0] = 1
ensemble_pred[ensemble_pred < 0] = -1
wrong = np.count_nonzero(ensemble_pred - y_test)
print("Final Accuracy of ENSEMBLE: %f" %((len(y_test) - wrong) / len(y_test)))
'''






