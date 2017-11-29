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


'''
# Scale data to have 0 mean and unit variance 
scaler = sklearn.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
#x_test = scaler.transform(x_test)


# Dimensional reduction of training feature through PCA
pcad = 50
def doPCA(x): 
	from sklearn.decomposition import PCA
	pca = PCA(n_components=pcad, svd_solver='arpack')
	pca.fit(x)
	return pca

pca = doPCA(x_train)
x_train = pca.transform(x_train)
#x_test = pca.transform(x_test)
'''

clf = ensemble.ExtraTreesClassifier()
clf = clf.fit(x_train, y_train)
model = feature_selection.SelectFromModel(clf, prefit=True)
x_train = model.transform(x_train)
x_test = model.transform(x_test)
print(np.shape(x_train))


counter = 0 
for leaf_size in range(19, 3, -2):


	for num_e in range(800, 1600, 100):
		# Train random forest classifier
		print("Begin training random forest...")
		rf = ensemble.RandomForestClassifier(n_estimators=num_e, min_samples_leaf=leaf_size)
		rf.fit(x_train, y_train)

		trng_acc = rf.score(x_train, y_train)
		val_acc = rf.score(x_test, y_test)
		print("leaf size: %d" %leaf_size)
		print("estimators: %d" %num_e)
		print("Training accuracy: %f" %trng_acc)
		print("Validation accuracy: %f" %val_acc)

	'''
	# Save trained classifier
	joblib.dump(rf, "rf" + str(counter) + ".pkl")
	counter += 1
	'''


# Train random forest classifier
print("Begin training random forest...")
rf = ensemble.RandomForestClassifier(n_estimators=800, min_samples_leaf=5)
rf.fit(x_train, y_train)


'''
rf = ensemble.AdaBoostClassifier(n_estimators=500)
rf.fit(x_train, y_train


trng_acc = rf.score(x_train, y_train)
val_acc = rf.score(x_test, y_test)
print("Training accuracy: %f" %trng_acc)
print("Validation accuracy: %f" %val_acc)
'''
