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
y_test[y_test == 2] = -1

x_train = x_train[0:partition, :]
y_train = y_train[0:partition]
y_train[y_train == 2] = -1

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

clf = ensemble.ExtraTreesClassifier(n_estimators=800, min_samples_leaf=5)
clf = clf.fit(x_train, y_train)
model = feature_selection.SelectFromModel(clf, prefit=True)
x_train = model.transform(x_train)
x_test = model.transform(x_test)
print(np.shape(x_train))

# Train random forest classifier with gini-impurity
print("Begin training random forest with gini...")
clf = ensemble.RandomForestClassifier(n_estimators=800, min_samples_leaf=5)
clf.fit(x_train, y_train)
trng_acc = clf.score(x_train, y_train)
val_acc = clf.score(x_test, y_test)
rf1_pred = clf.predict(x_test)
print("Training accuracy: %f" %trng_acc)
print("Validation accuracy: %f" %val_acc)
joblib.dump(clf, "rf_gini.pkl")

# Train random forest classifier with entropy-impurity
print("Begin training random forest with entropy...")
clf = ensemble.RandomForestClassifier(n_estimators=800, min_samples_leaf=5, criterion='entropy')
clf.fit(x_train, y_train)
trng_acc = clf.score(x_train, y_train)
val_acc = clf.score(x_test, y_test)
rf2_pred = clf.predict(x_test)
print("Training accuracy: %f" %trng_acc)
print("Validation accuracy: %f" %val_acc)
joblib.dump(clf, "rf_entropy.pkl")

# Train adaboost classifier
print("Begin AdaBoost...")
clf = ensemble.AdaBoostClassifier(n_estimators=1000)
clf.fit(x_train, y_train)
trng_acc = clf.score(x_train, y_train)
val_acc = clf.score(x_test, y_test)
ada_pred = clf.predict(x_test)
print("Training accuracy: %f" %trng_acc)
print("Validation accuracy: %f" %val_acc)
joblib.dump(clf, "ada.pkl")

# Train Extra trees classifier with gini
print("Begin training extra trees...")
clf = ensemble.ExtraTreesClassifier(n_estimators=800, min_samples_leaf=5)
clf.fit(x_train, y_train)
trng_acc = clf.score(x_train, y_train)
val_acc = clf.score(x_test, y_test)
et1_pred = clf.predict(x_test)
print("Training accuracy: %f" %trng_acc)
print("Validation accuracy: %f" %val_acc)
joblib.dump(clf, "extratrees.pkl")

# Train Extra trees classifier with entropy
print("Begin training extra trees...")
clf = ensemble.ExtraTreesClassifier(n_estimators=800, min_samples_leaf=5, criterion='entropy')
clf.fit(x_train, y_train)
trng_acc = clf.score(x_train, y_train)
val_acc = clf.score(x_test, y_test)
et2_pred = clf.predict(x_test)
print("Training accuracy: %f" %trng_acc)
print("Validation accuracy: %f" %val_acc)
joblib.dump(clf, "extratrees_gini.pkl")

# Train Gradient Boosting Classifier
print("Begin training Gradient Boosting...")
clf = ensemble.GradientBoostingClassifier(n_estimators=1200, min_samples_leaf=19)
clf.fit(x_train, y_train)
trng_acc = clf.score(x_train, y_train)
val_acc = clf.score(x_test, y_test)
gb_pred = clf.predict(x_test)
print("Training accuracy: %f" %trng_acc)
print("Validation accuracy: %f" %val_acc)
joblib.dump(clf, "gb.pkl")

# Train Neural Network 
# Training neural network 
print("Begin training Neural Network...")
nn = neural_network.MLPClassifier(hidden_layer_sizes=(500, 500, 500), early_stopping=True)
nn.fit(x_train, y_train)
trng_acc = nn.score(x_train, y_train)
val_acc = nn.score(x_test, y_test)
nn_pred = nn.predict(x_test)
print("Training accuracy: %f" %trng_acc)
print("Validation accuracy: %f" %val_acc)
joblib.dump(nn, "nn.pkl")


ensemble_pred = rf1_pred + rf2_pred + ada_pred + et1_pred + et2_pred + 5*gb_pred + nn_pred
ensemble_pred[ensemble_pred >= 0] = 1
ensemble_pred[ensemble_pred < 0] = -1
wrong = np.count_nonzero(ensemble_pred - y_test)
print("Final Accuracy of ENSEMBLE: %f" %((len(y_test) - wrong) / len(y_test)))







