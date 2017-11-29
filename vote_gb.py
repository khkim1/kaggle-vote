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
partition = 55000
x_test = x_train[partition:, :]
y_test = y_train[partition:]
x_train = x_train[0:partition, :]
y_train = y_train[0:partition]


# Scale data to have 0 mean and unit variance 
scaler = sklearn.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

'''
# Dimensional reduction of training feature through PCA
pcad = 150
def doPCA(x): 
	from sklearn.decomposition import PCA
	pca = PCA(n_components=pcad, svd_solver='arpack')
	pca.fit(x)
	return pca

pca = doPCA(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
'''

for ratio in np.linspace(0.5, 2, 15):
	clf = ensemble.ExtraTreesClassifier()
	clf = clf.fit(x_train, y_train)
	model = feature_selection.SelectFromModel(clf, prefit=True, threshold=(str(ratio) + '*mean'))
	x_train2 = model.transform(x_train)
	x_test2 = model.transform(x_test)
	print(np.shape(x_train))


	# Train Gradient Boosting Classifier
	print("Begin training Gradient Boosting...")
	rf = ensemble.GradientBoostingClassifier(n_estimators=1200, min_samples_leaf=19)
	rf.fit(x_train2, y_train)

	trng_acc = rf.score(x_train2, y_train)
	val_acc = rf.score(x_test2, y_test)
	print("Ratio: %f" %ratio)
	print("Training accuracy: %f" %trng_acc)
	print("Validation accuracy: %f" %val_acc)

'''
# Save trained classifier
joblib.dump(rf, "rf" + str(counter) + ".pkl")
counter += 1
'''

'''
# Train random forest classifier
print("Begin training random forest...")
rf = ensemble.RandomForestClassifier(n_estimators=800, min_samples_leaf=5)
rf.fit(x_train, y_train)


rf = ensemble.AdaBoostClassifier(n_estimators=500)
rf.fit(x_train, y_train


trng_acc = rf.score(x_train, y_train)
val_acc = rf.score(x_test, y_test)
print("Training accuracy: %f" %trng_acc)
print("Validation accuracy: %f" %val_acc)
'''
