from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
import pandas as pd
from matplotlib import pyplot


np.random.seed(3)

# number of wine classes
classifications = 4

# load dataset
dataset = np.loadtxt('dataset.csv', delimiter=",")

# split dataset into sets for testing and training
X = dataset[:,1:4]
Y = dataset[:,0:1]

# standardize the data attributes
standardized_X = preprocessing.scale(X)

# split data for train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)

# normalize proces
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)

# convert output values to one-hot
y_train = keras.utils.to_categorical(y_train-1, classifications)
y_test = keras.utils.to_categorical(y_test-1, classifications)

# creating model
model = Sequential()
# input layer
model.add(Dense(20, input_dim=3, activation='relu'))
# hidden layer
model.add(Dense(45, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(35, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))

# output layer
model.add(Dense(classifications, activation='softmax'))
# compile and fit model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=25, epochs=50, validation_data=(x_test, y_test))

print(y_test)
print(x_test)
# plot train and validation loss
#pyplot.plot(history.history['loss'])
#pyplot.plot(history.history['val_loss'])
#pyplot.title('model train vs validation loss')
#pyplot.ylabel('loss')
#pyplot.xlabel('epoch')
#pyplot.legend(['train', 'validation'], loc='upper right')
#pyplot.show()

# plot train and validation accuracy
#pyplot.plot(history.history['accuracy'])
#pyplot.plot(history.history['val_accuracy'])
#pyplot.title('model train vs validation accuracy')
#pyplot.ylabel('accuracy')
#pyplot.xlabel('epoch')
#pyplot.legend(['train', 'validation'], loc='upper right')
#pyplot.show()

