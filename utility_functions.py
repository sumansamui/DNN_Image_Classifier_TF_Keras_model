import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os

import config



def download_mnist_fashion():

	fashion_mnist = keras.datasets.fashion_mnist

	(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()



	X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0

	y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

	X_test = X_test/255.0


	os.makedirs(config.path_to_data,exist_ok=True)
	np.save(os.path.join(config.path_to_data,'X_train.npy'),X_train)
	np.save(os.path.join(config.path_to_data,'y_train.npy'),y_train)
	np.save(os.path.join(config.path_to_data,'X_valid.npy'),X_valid)
	np.save(os.path.join(config.path_to_data,'y_valid.npy'),y_valid)
	np.save(os.path.join(config.path_to_data,'X_test.npy'),X_test)
	np.save(os.path.join(config.path_to_data,'y_test.npy'),y_test)

	print('saving data....complete!')


def load_dataset():

	X_train = np.load(os.path.join(config.path_to_data,'X_train.npy')) 
	y_train = np.load(os.path.join(config.path_to_data,'y_train.npy'))

	X_valid = np.load(os.path.join(config.path_to_data,'X_valid.npy'))
	y_valid = np.load(os.path.join(config.path_to_data,'y_valid.npy'))
	
	X_test = np.load(os.path.join(config.path_to_data,'X_test.npy'))
	y_test = np.load(os.path.join(config.path_to_data,'y_test.npy'))

	return X_train, y_train, X_valid, y_valid, X_test, y_test


# Define a simple sequential model
def create_model():
	model = keras.models.Sequential([
		keras.layers.Flatten(input_shape=[28,28]),
		keras.layers.Dense(300,activation='relu'),
		keras.layers.Dense(100,activation='relu'),
		keras.layers.Dense(10,activation='softmax')
		])
	model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return model


