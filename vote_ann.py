import numpy as np 
import tensorflow as tf 
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from numpy import genfromtxt, linalg
import sklearn
from sklearn import preprocessing, ensemble, feature_selection
from matplotlib import pyplot as plt

data = genfromtxt('train_2008.csv', delimiter=',')

# Trim the first row which contains header information 
# and the first three columns which contain irrelevant data 
data = data[1:, 3:]
x_train = data[:, 0:-1]
y_train = data[:, -1]
y_train[y_train==2] = 0
y_train = np_utils.to_categorical(y_train)


# Scale data to have 0 mean and unit variance 
scaler = sklearn.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)

clf = ensemble.ExtraTreesClassifier()
clf = clf.fit(x_train, y_train)
model = feature_selection.SelectFromModel(clf, prefit=True)
x_train = model.transform(x_train)

# First layer 
model = Sequential()
model.add(Dense(500, input_dim=np.shape(x_train)[1]))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Second layer 
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Second layer 
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Second layer 
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Second layer 
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Second layer 
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Final softmax layer
model.add(Dense(2))
model.add(Activation('softmax'))

## Printing a summary of the layers and weights in your model
model.summary()

# Initialize training metrics 
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

while True: 
	fit = model.fit(x_train, y_train, batch_size=512, nb_epoch=1, verbose=1, shuffle=True, show_accuracy=True, 
		validation_split=0.2)
	'''
	print("Test accuracy: %f" %score[1])
	if score[1] > 0.983:
		break
	'''


'''
## Printing the accuracy of our model, according to the loss function specified in model.compile above
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
'''

