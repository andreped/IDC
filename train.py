import numpy as np
import matplotlib.pyplot as plt
import os
from numpy.random import shuffle
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D,\
Dropout, Flatten, Dense
from tensorflow.python.keras.models import Sequential
import cv2
from tqdm import tqdm
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback
import h5py


def make_set(test_dir, data_path, N):
	test_set = []
	for d1 in test_dir:
		tmp = []
		for d2 in os.listdir(data_path + d1):
			tmp_class = []
			for d3 in os.listdir(data_path + d1 + "/" + d2):
				tmp_class.append(data_path + d1 + "/" + d2 + "/" + d3)
			shuffle(tmp_class)
			tmp.append(tmp_class[:N])
		for t in tmp:
			for n in t:
				test_set.append(n)
	return test_set


def fill_set_with_images(test_set, input_shape):
	X_test = np.zeros((len(test_set),) + input_shape)
	Y_test = np.zeros(len(test_set))
	for i, t in tqdm(enumerate(test_set), "Image: ", total=len(test_set)):
		tmps = np.zeros(input_shape)
		tmp = cv2.cvtColor(cv2.imread(test_set[i]), cv2.COLOR_BGR2RGB).astype(np.float32)
		tmps[:tmp.shape[0], :tmp.shape[1]] = tmp
		X_test[i] = tmps / 255
		Y_test[i] = int(t.split("/")[-2])
	return X_test, Y_test

# name for model
name = "model_IDC_classification_17_07"

# paths
data_path = "C:/Users/andrep/Documents/Projects/DP/IDC/data/"
save_model_path = "C:/Users/andrep/Documents/Projects/DP/IDC/output/saved_models/"
history_path = "C:/Users/andrep/Documents/Projects/DP/IDC/output/history/"
datasets_path = "C:/Users/andrep/Documents/Projects/DP/IDC/output/datasets/"

pats = []
for d1 in os.listdir(data_path):
	pats.append(d1)

tot = len(pats)
N_test = int(np.round(tot*0.2))

shuffle(pats)

test_dir = pats[:N_test]
val_dir = pats[N_test:int(2*N_test)]
train_dir = pats[int(2*N_test):]

# number of samples from each
N = 10

# number of classes
nb_classes = 2

# image size
input_shape = (50, 50, 3)

# make sets
test_set = make_set(test_dir, data_path, N)
val_set = make_set(val_dir, data_path, N)
train_set = make_set(val_dir, data_path, N)

# make data split, and store images in arrays
X_test, Y_test = fill_set_with_images(test_set, input_shape)
X_val, Y_val = fill_set_with_images(val_set, input_shape)
X_train, Y_train = fill_set_with_images(train_set, input_shape)

# one-hot encode
Y_train = to_categorical(Y_train, num_classes=nb_classes)
Y_val = to_categorical(Y_val, num_classes=nb_classes)
Y_test = to_categorical(Y_test, num_classes=nb_classes)

# save generated datasets
f = h5py.File(datasets_path + "dataset_" + name + '.h5', 'w')
f.create_dataset("X_train", data=X_train, compression="gzip", compression_opts=4)
f.create_dataset("Y_train", data=Y_train, compression="gzip", compression_opts=4)
f.create_dataset("X_val", data=X_val, compression="gzip", compression_opts=4)
f.create_dataset("Y_val", data=Y_val, compression="gzip", compression_opts=4)
f.create_dataset("X_test", data=X_test, compression="gzip", compression_opts=4)
f.create_dataset("Y_test", data=Y_test, compression="gzip", compression_opts=4)
f.close()


# define model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))


model.compile(
	loss='binary_crossentropy',
	optimizer='adam',
	metrics=['accuracy']
	)

datagen = ImageDataGenerator(
	featurewise_center=False,
	samplewise_center=False,
	featurewise_std_normalization=False,
	samplewise_std_normalization=False,
	zca_whitening=False,
	rotation_range=20,
	width_shift_range=0.2,
	height_shift_range=0.1,
	horizontal_flip=True,
	vertical_flip=True
	)

batch_size = 32
epochs = 100

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_epoch_end(self, batch, logs={}):
		self.losses.append([np.float32(logs.get('loss')), 
			np.float32(logs.get('val_loss')),
			np.float32(logs.get('acc')),
			np.float32(logs.get('val_acc'))])

		# save history:
		f = h5py.File(history_path+'/history_' + name + '.h5', 'w')
		f.create_dataset("history", data=self.losses, compression="gzip", compression_opts=4)
		f.close()

history_log = LossHistory()

save_best = ModelCheckpoint(
	save_model_path + '/name' + '.h5',
	monitor='val_acc',
	verbose=0,
	save_best_only=True,
	save_weights_only=False,
	mode='auto',
	period=1
)

history = model.fit_generator(
	datagen.flow(X_train, Y_train, batch_size=batch_size),
	steps_per_epoch=len(X_train)/batch_size,
	epochs = epochs,
	validation_data = [X_val, Y_val],
	callbacks=[save_best, history_log]
	)


