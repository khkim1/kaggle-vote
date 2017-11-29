import numpy as np
from numpy import genfromtxt, linalg
import sklearn
from sklearn import preprocessing, neural_network, feature_selection, ensemble
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
pcad = 50
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
model = feature_selection.SelectFromModel(clf, prefit=True)
x_train = model.transform(x_train)
x_test = model.transform(x_test)

nn = neural_network.MLPClassifier(hidden_layer_sizes=(500, 500, 500), early_stopping=True, )
nn.fit(x_train, y_train)

trng_acc = nn.score(x_train, y_train)
val_acc = nn.score(x_test, y_test)
print("Training accuracy: %f" %trng_acc)
print("Validation accuracy: %f" %val_acc)









