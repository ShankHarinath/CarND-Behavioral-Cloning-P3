import csv
import os
import os
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Tensorflow imports
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Conv2D, Cropping2D, Flatten, Lambda, Reshape, MaxPooling2D, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.metrics import categorical_accuracy, mean_squared_error
from tensorflow.python.keras.callbacks import BaseLogger, ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras import backend as K

# Input and output directories
data_dir = os.getcwd() + "/data/"
output_dir = os.getcwd() + "/out/"
log_dir = os.getcwd() + "/logs/"


# Read recorded data
angle_delta = 0.25

def preprocess_img(line):
    images = []
    measurements = []
    center_source_path = line[0]
    left_source_path = line[1]
    right_source_path = line[2]
    center_filename = center_source_path.split('/')[-1]
    left_filename = left_source_path.split('/')[-1]
    right_filename = right_source_path.split('/')[-1]
    for source_path, delta in [(center_source_path, 0), (left_source_path, angle_delta), (right_source_path, -angle_delta)]:
        filename = source_path.split('/')[-1]
        current_path = data_dir + 'IMG/' + filename
        measurement = float(line[3].strip()) + delta
        # Read image
        image = cv2.imread(current_path)
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Flip random images
        flip = np.random.randint(2)
        if flip == 0:
            image = cv2.flip(image, 1)
            measurement = -measurement
        images.append(image)
        measurements.append(measurement)
    return images, measurements

def generator(lines, batch_size):
    num_samples = len(lines)
    while True:
        shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_line_items = lines[offset:offset + batch_size]
            images = []
            measurements = []
            for line in batch_line_items:
                i, m = preprocess_img(line)
                images.extend(i)
                measurements.extend(m)
            x = np.array(images)
            y = np.array(measurements)
            yield x, y

lines = []
with open(data_dir + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

lines = shuffle(lines[1:])
train_data, validation_data = train_test_split(lines, test_size=0.2)

input_shape = (160, 320, 3)
gpu_list = ["/gpu:%d" % i for i in range(8)]

def get_model():
    gpus = iter(gpu_list)
    model = Sequential()
    
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25), (0,0))))
    
    with tf.device(next(gpus)):
        # Layer to learn suitable color scheme
        model.add(Conv2D(3, (1, 1), activation='relu'))
        model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
        
    with tf.device(next(gpus)):
        model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
        model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    
    with tf.device(next(gpus)):
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
    
    with tf.device(next(gpus)):
        model.add(Flatten())
        model.add(Dense(100))
        model.add(LeakyReLU())
        model.add(Dense(50))
        model.add(LeakyReLU())
        model.add(Dense(10))
        model.add(Dense(1))
    return model

model = get_model()

# Training loss and hyperparameters
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Optimizer
optimizer = Adam(lr=LEARNING_RATE, decay=1e-6)

# Training metrics
metrics=[mean_squared_error]

# Training loss
loss = 'mse'

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

train_generator = generator(train_data, BATCH_SIZE)
validation_generator = generator(validation_data, BATCH_SIZE)

# Training callbacks
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')
tensorboard = TensorBoard(log_dir='./logs', batch_size=BATCH_SIZE, write_graph=True)
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
filepath = output_dir + "model-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoints = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
callbacks = [tensorboard, early_stopper, checkpoints, reduce_lr_on_plateau]

# Model train and save
model.fit_generator(train_generator, steps_per_epoch=len(train_data) / BATCH_SIZE, verbose=1, validation_data=validation_generator, validation_steps = len(validation_data) / BATCH_SIZE, callbacks=callbacks, epochs=EPOCHS)
model.save(output_dir + "model.h5")

