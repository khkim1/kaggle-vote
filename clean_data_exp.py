import numpy as np
from numpy import genfromtxt, linalg
import sklearn
from sklearn import preprocessing, ensemble, neural_network, neighbors, feature_selection, tree, metrics
from matplotlib import pyplot as plt
from sklearn.externals import joblib

data = genfromtxt('new/train_2008_back_to_basics_selected_feats.csv', delimiter=',')

# Remove the feature descriptions
data = data[1:, 0:]
x_train = data[:, 0:-1]
y_train = data[:, -1]

partition = 45000

x_test = x_train[partition:, :]
y_test = y_train[partition:]
#y_test[y_test == 2] = -1

x_train = x_train[0:partition, :]
y_train = y_train[0:partition]
#y_train[y_train == 2] = -1

# Scale data to have 0 mean and unit variance 
'''
scaler = sklearn.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
'''
'''
clf = ensemble.ExtraTreesClassifier()
clf = clf.fit(x_train, y_train)
model = feature_selection.SelectFromModel(clf, prefit=True, threshold='mean')
x_train_use = model.transform(x_train)
x_test_use = model.transform(x_test)
print(np.shape(x_train_use))
'''

for sub in [0.2, 0.4, 0.6, 0.8, 1]:
	'''
	clf = ensemble.ExtraTreesClassifier()
	clf = clf.fit(x_train, y_train)
	model = feature_selection.SelectFromModel(clf, prefit=True, threshold=str(ratio)+'*mean')
	x_train_use = model.transform(x_train)
	x_test_use = model.transform(x_test)
	print(np.shape(x_train_use))


	gb_pred = []
	max_acc = 0
	'''
	x_train_use = x_train
	x_test_use = x_test

	# Train Gradient Boosting Classifier
	print("Begin training Gradient Boosting...")
	clf = ensemble.GradientBoostingClassifier(n_estimators=800, min_samples_leaf=15, 
											  learning_rate=0.1, subsample=sub)
	clf.fit(x_train_use, y_train)

	trng_acc = clf.score(x_train_use, y_train)
	val_acc = clf.score(x_test_use, y_test)

	print("Subsample: %f" %sub)
	print("Training accuracy: %f" %trng_acc)
	print("Validation accuracy: %f" %val_acc)

	'''
	pred = clf.predict_proba(x_test_use)
	labels = np.zeros((len(y_test), ))
	F1 = []
	max_acc = 0 
	for threshold in np.linspace(-0.05, 0.05, 200):
		for idx in range(len(y_test)):
			if pred[idx, 0] - pred[idx, 1] > threshold:
				labels[idx] = 1
			else:
				labels[idx] = 2
		accuracy = 1 - np.count_nonzero(labels - y_test) / len(y_test)
		if accuracy > max_acc:
			max_acc = accuracy
			max_threshold = threshold
		#print(metrics.precision_recall_fscore_support(y_test, labels))
	print("ratio: %f, threshold: %f, best accuracy: %f" %(ratio, max_threshold, max_acc))
	'''







	'''
	print("Begin training Adaboost...")
	for n_est in range(2, 10, 1):
		clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(min_samples_split=100), n_estimators=n_est)
		clf.fit(x_train_use, y_train)

		trng_acc = clf.score(x_train_use, y_train)
		val_acc = clf.score(x_test_use, y_test)
		if val_acc > max_acc:
			max_acc = val_acc
			max_idx = idx
		gb_pred.append(clf.predict(x_test_use))
		print("Index: %d" %idx)
		print("Number of estimators: %d" %n_est)
		print("Training accuracy: %f" %trng_acc)
		print("Validation accuracy: %f" %val_acc)
	'''

	#joblib.dump(clf, "gb" + str(idx) + ".pkl")



