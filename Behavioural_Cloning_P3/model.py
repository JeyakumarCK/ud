import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.optimizers import Adam

#---------------------------------------------------#
# Define Processing Parameters
#---------------------------------------------------#
CSV_FILE_NAME = 'driving_log.csv'
BATCH_SIZE = 32
EPOCH_COUNT = 4
LEARN_RATE = 0.0001
SAMPLES_PER_EPOCH = int((20000//BATCH_SIZE)*BATCH_SIZE)
ROWS, COLS, DEPTH = 64, 64, 3
IMG_SIZE = (ROWS, COLS)
INPUT_SHAPE = (ROWS, COLS, DEPTH)
TRAINING_SPLIT = 0.8
WEIGHTS_FILE_NAME = 'model.h5'
MODEL_FILE_NAME = 'model.json'

#---------------------------------------------------#
# Flip Image Horizontally
#---------------------------------------------------#
def flipImage(image):
	image1 = cv2.flip(image, 1)
	return image1

#---------------------------------------------------#
# Remove the sky portion in the top and car hood 
# portion in the bottom by cropping & resize it 
# defined size or 64x64 if not mentioned
#---------------------------------------------------#
def cropAndResize(image, img_size=(64, 64)):
	image1 = image[55:135, :, :]
	image1 = cv2.resize(image1, img_size)
	return image1

#---------------------------------------------------#
# Adjust the brightness of Image by a random factor
#---------------------------------------------------#
def adjustBrightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    adjustFactor = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*adjustFactor
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1

#---------------------------------------------------#
# Normalize the intensity of the image
#---------------------------------------------------#
def normalizeImage(image):
    image1 = image/255.0 - 0.5
    return image1

#---------------------------------------------------#
# High Level Function to read image from the given
# dataframe record & perform the image preprocessings
#---------------------------------------------------#
def readImageWithLabel(df_row):
	#---------------------------------------------------#
	# Randomly pick Center/Left/Right image
	#---------------------------------------------------#
	options = ['CenterImg', 'LeftImg', 'RightImg']
	imgToRead = random.choice(options)
	steer = df_row['SteerAngle']
	if imgToRead == 'LeftImg':
		steer += 0.25
	elif imgToRead == 'RightImg':
		steer -= 0.25

	#---------------------------------------------------#
	# Read the image, crop the sky & bonnet & resize it
	#---------------------------------------------------#
	image = ndimage.imread(df_row[imgToRead].strip())
	image = cropAndResize(image, IMG_SIZE)

	#---------------------------------------------------#
	# Flip 50% of image
	#---------------------------------------------------#
	if random.random() > 0.5:
		image = flipImage(image)
		steer = -1*steer

	#---------------------------------------------------#
	# Adjust brightness of 80% of images
	#---------------------------------------------------#
	if random.random() > 0.2:
		image = adjustBrightness(image)

	#---------------------------------------------------#
	# Normalize the intensity of images
	#---------------------------------------------------#
	image = normalizeImage(image)

	return image, steer

#---------------------------------------------------#
# Generator to feed images of required batch size 
# to model for training/validation
#---------------------------------------------------#
def imageDataGenerator(df, batch_size=32):
	batches_per_epoch = df.shape[0] // batch_size
	batch_counter = 0
	while True:
		start_idx = batch_counter * batch_size
		end_idx = start_idx + batch_size - 1

		X_batch = np.zeros((batch_size, ROWS, COLS, DEPTH), dtype=np.float32)
		y_batch = np.zeros((batch_size,), dtype=np.float32)

		i = 0
		for index, row in df.loc[start_idx:end_idx].iterrows():
			X_batch[i], y_batch[i] = readImageWithLabel(row)
			i += 1

		batch_counter += 1
		if batch_counter == batches_per_epoch-1:
			# Reset Batch Counter
			batch_counter = 0
		yield X_batch, y_batch

#---------------------------------------------------#
# Define the Neural Network Model
#---------------------------------------------------#
def getModel():
    model = Sequential()
    
    #Layer-1 Convolution layer - from 3 Dimention to 24 dimension
    model.add(Convolution2D(24, 4, 4, subsample=(2, 2), border_mode='same', input_shape=INPUT_SHAPE ))
    model.add(ELU())
    model.add(MaxPooling2D((3, 3)))
    
    #Layer-2 Convolution layer - from 24 Dimention to 36 dimension
    model.add(Convolution2D(36, 3, 3, border_mode='valid'))
    model.add(ELU())
    model.add(Dropout(0.3))
    model.add(AveragePooling2D((2, 2)))
    
    #Layer-3 Convolution layer - from 36 Dimention to 48 dimension
    model.add(Convolution2D(48, 4, 4, border_mode='same'))
    model.add(ELU())
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((2, 2)))
    
    #Layer-4 Convolution layer - from 48 Dimention to 64 dimension
    model.add(Convolution2D(64, 2, 2, border_mode='valid'))
    model.add(ELU())
    model.add(Dropout(0.3))
    model.add(AveragePooling2D((1, 1)))
    
    #Layer-5 Flatten the output
    model.add(Flatten())
    
    #Layer-6 Fully Connected Layer
    model.add(Dense(512))
    model.add(Dropout(0.2))
    model.add(ELU())
    
    #Layer-7 Fully Connected Layer
    model.add(Dense(64))
    model.add(ELU())
    
    #Layer-8 Fully Connected Layer, no activation as it is a regression model
    model.add(Dense(1))
    adm = Adam(lr=LEARN_RATE)
    model.compile(optimizer=adm, loss='mean_squared_error', metrics=['accuracy'])
    return model


if __name__ == "__main__":
	
	#---------------------------------------------------#
	# Create DataFrame & Split for Training & Validation
	#---------------------------------------------------#
	df_a = pd.read_csv(CSV_FILE_NAME, header=0, usecols=[0, 1, 2, 3], names=['CenterImg', 'LeftImg', 'RightImg', 'SteerAngle'])
	print('df_a record count', df_a.shape[0])
	train_rows_count = int(df_a.shape[0]*TRAINING_SPLIT)
	df_a = df_a.sample(frac=1).reset_index(drop=True)
	df_t = df_a.loc[0:train_rows_count-1]
	df_v = df_a.loc[train_rows_count:]
	df_v = df_v.reset_index(drop=True)
	print('df_t record count', df_t.shape[0])
	print('df_v record count', df_v.shape[0])
	df_a = None

	#---------------------------------------------------#
	# Create Model & Train the model using generators
	#---------------------------------------------------#
	model = getModel()
	trainGen = imageDataGenerator(df_t, BATCH_SIZE)
	validGen = imageDataGenerator(df_v, BATCH_SIZE)
	print('Training the model')
	model.fit_generator(trainGen, samples_per_epoch = SAMPLES_PER_EPOCH, nb_epoch = EPOCH_COUNT, 
		validation_data = validGen, nb_val_samples = 1000)
	print('Model trained successfully')
	
	#---------------------------------------------------#
	# Save Model & Weights to json and h5 files
	#---------------------------------------------------#
	print('Saving weights to', WEIGHTS_FILE_NAME)
	model.save_weights(WEIGHTS_FILE_NAME)

	print('Saving model to', MODEL_FILE_NAME)
	with open(MODEL_FILE_NAME, 'w') as f:
	   f.write(model.to_json())
	print('#---------------------------------------------------#')
	