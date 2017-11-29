import numpy as np
from numpy import genfromtxt, linalg
import sklearn
from sklearn import preprocessing, svm, ensemble, feature_selection
from matplotlib import pyplot as plt


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

clf = ensemble.ExtraTreesClassifier()
clf = clf.fit(x_train, y_train)
model = feature_selection.SelectFromModel(clf, prefit=True, threshold='mean')
x_train = model.transform(x_train)
x_test = model.transform(x_test)


clf = svm.SVC()
clf.fit(x_train, y_train)

trng_acc = clf.score(x_train, y_train)
val_acc = clf.score(x_test, y_test)
print("Training accuracy: %f" %trng_acc)
print("Validation accuracy: %f" %val_acc)

'''
plt.plot(x_train[y_train == 1, 0], x_train[y_train == 1, 1], 'r.')
plt.plot(x_train[y_train == 2, 0], x_train[y_train == 2, 1], 'b.')
plt.show()
'''






