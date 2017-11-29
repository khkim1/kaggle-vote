import numpy as np
from numpy import genfromtxt, linalg
import sklearn
from sklearn import preprocessing, ensemble
from matplotlib import pyplot as plt
from sklearn.externals import joblib

# Helper function to calculate ensemble accuracy 
def model_score(y_test, ensemble):
	num_models = len(ensemble)
	num_val = len(y_test)

	prediction = np.zeros((num_val, ))
	for idx in range(num_models):
		prediction += ensemble[idx]

	prediction[prediction >= 0] = 1
	prediction[prediction < 0] = -1

	classification = prediction - y_test 
	num_wrong = np.count_nonzero(classification)
	return 1 - (num_wrong / len(y_test))

#------------------------------------------------------------------------------------------------

#Load the data
print("Loading data")
data = genfromtxt('train_2008.csv', delimiter=',')


# Trim the first row which contains header information 
# and the first three columns which contain irrelevant data 
data = data[1:, 3:]
x_train = data[:, 0:-1]
y_train = data[:, -1]

partition = 45000
x_test = x_train[partition:, :]
y_test = y_train[partition:]
y_test[y_test == 2] = -1

x_train = x_train[0:partition, :]
y_train = y_train[0:partition]


# Scale data to have 0 mean and unit variance 
scaler = sklearn.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Dimensional reduction of training feature through PCA
pcad = 50
def doPCA(x): 
	from sklearn.decomposition import PCA
	pca = PCA(n_components=pcad)
	pca.fit(x)
	return pca

pca = doPCA(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)


#Load pretrained models
print("Loading ensemble library")
file_pre = 'rf'
model_list = []
ensemble = []
prediction_list = []

for i in range(30):
	clf = joblib.load(file_pre + str(i) + ".pkl")
	prediction = clf.predict(x_test)
	print("score of model %d: %f" %(i, clf.score(x_test, y_test)))
	prediction[prediction == 2] = -1
	model_list.append(clf)
	prediction_list.append(prediction)

# Perform ensemble selection with replacement 
epoch = 0 
ensemble.append(prediction_list[0])
ensemble.append(prediction_list[1])
ensemble.append(prediction_list[2])
ensemble.append(prediction_list[3])

print("Perform ensemble selection")
while True: 
	epoch += 1
	max_val_acc = 0
	max_val_idx = 0

	for idx in range(len(model_list)):
		ensemble.append(prediction_list[idx])
		val_acc = model_score(y_test, ensemble)
		if val_acc > max_val_acc:
			max_idx = idx
			max_val_acc = val_acc
		del ensemble[-1]
	ensemble.append(prediction_list[max_idx])
	ensemble_acc = max_val_acc
	print("Ensemble perfomance on epoch %d: %f" %(epoch, ensemble_acc))

