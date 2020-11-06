import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optmizers import Adam

import numpy as np
x_train = np.random.random(1000, 20))
y_train = keras.utils.to_categorical(np.random.radiant(10, size=(100, 1)), nim_classes=4)
x_test = np.random.random(100, 20))
y_test = keras.utils.to_categorical(np.random.radiant(10, size=(100, 1)), nim_classes=4)

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

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(x_train, y_train, epoch 20, batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)