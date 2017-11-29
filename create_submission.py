import numpy as np
from numpy import genfromtxt, linalg
import sklearn
from sklearn import preprocessing, ensemble, feature_selection
from matplotlib import pyplot as plt
from sklearn.externals import joblib

data = genfromtxt('new/', delimiter=',')

# Remove the feature descriptions
data = data[1:, 3:]
x_train = data[:, 0:-1]
print(np.shape(x_train))
#y_train = data[:, -1]
y_train = data[:, -1]

test_data = genfromtxt('rawdata/2008_data_feat_eng_all_feats_fixed/test_data_feat_eng_mean_fill_raw.csv', delimiter=',')
x_test = test_data[1:, 3:]
print(np.shape(x_test))



# Scale data to have 0 mean and unit variance 
'''
scaler = sklearn.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
'''

clf = ensemble.ExtraTreesClassifier(n_estimators=500, min_samples_leaf=10)
clf = clf.fit(x_train, y_train)
model = feature_selection.SelectFromModel(clf, prefit=True, threshold='2.66667*mean')
x_train = model.transform(x_train)
x_test = model.transform(x_test)


# Train Gradient Boosting Classifier
print("Begin training Gradient Boosting...")
rf = ensemble.GradientBoostingClassifier(n_estimators=500, min_samples_leaf=15)
rf.fit(x_train, y_train)

trng_acc = rf.score(x_train, y_train)

print("Training accuracy: %f" %trng_acc)

labels = rf.predict(x_test)

'''
pred = rf.predict_proba(x_test)
labels = np.zeros((len(x_test), ))

for idx in range(len(x_test)):
	if pred[idx, 0] - pred[idx, 1] > -0.026382:
		labels[idx] = 1
	else:
		labels[idx] = 2
'''
np.savetxt("prediction.csv", labels)

