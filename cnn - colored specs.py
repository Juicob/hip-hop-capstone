# %%
import os
import sys
import datetime
import itertools
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objs as go
import plotly.express as px
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.random import set_seed
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, auc, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics

import visualkeras
from collections import defaultdict

set_seed(42)
np.random.seed(42)

# %%
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# %%
def plot_results(results):
  """Function to convert a models results into a dataframe and plot them to show the both the accuracy and validation accuracy, as well as the loss and validation loss over epochs.

  Args:
      results_dataframe (dataframe): 
  """

  results_dataframe = pd.DataFrame(results)

  fig = px.line(results_dataframe, x=results_dataframe.index, y=["accuracy","val_accuracy"])
  fig.update_layout(title='Accuracy and Validation Accuracy over Epochs',
                    xaxis_title='Epoch',
                    yaxis_title='Percentage',
                )
  fig.update_traces(mode='lines+markers')
  fig.show()

  fig = px.line(results_dataframe, x=results_dataframe.index, y=['loss','val_loss'])
  fig.update_layout(title='Loss and Validation Loss over Epochs',
                    xaxis_title='Epoch',
                    yaxis_title='Loss'
                )
  fig.update_traces(mode='lines+markers')
  fig.show()

def plotImages(images_arr, labels_arr):
    # labels_arr = ['Normal: 0' if label == 0 else 'Pneumonia: 1' for label in labels_arr]
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    # axes = axes.flatten()
    for img, label, ax in zip(images_arr, 
                              labels_arr, 
                              axes):
        ax.imshow(img)
        ax.set_title(label, size=18)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def evaluate_results(model): 
    labels = ['East','South', 'Mid-West', 'West']
    predictions = model.predict(X_test).argmax(axis=1)
    cm = metrics.confusion_matrix(y_test.argmax(axis=1), 
                                    predictions,
                                    normalize="pred")

    ax = sns.heatmap(cm, cmap='Blues',annot=True,square=True)
    ax.set(xlabel='Predicted Class',ylabel='True Class')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    print(metrics.classification_report(y_test.argmax(axis=1), predictions))

# def prepare_confusion_matrix(model):

#     predictions = model.predict(X_test).round()
#     cm = confusion_matrix(y_test, predictions)
#     return cm, predictions
    # test_generator.class_indices
  
# %%
# labels = {'East': 0, 'Mid-West': 1, 'South': 2, 'West': 3}
# for value in list(labels.values()):
#     print(list(labels.keys())[value])
# %% [markdown]
## Setting up paths

# %%

train_east = os.path.join('../capstone-data/music/TRAIN/East')
train_west = os.path.join('../capstone-data/music/TRAIN/West')
train_mid_west = os.path.join('../capstone-data/music/TRAIN/Mid-west')
train_south = os.path.join('../capstone-data/music/TRAIN/South')


# test_east = os.path.join('../capstone-data/music/TEST/East')
# test_west = os.path.join('../capstone-data/music/TEST/West')
# test_mid_west = os.path.join('../capstone-data/music/TEST/Mid-west')
# test_south = os.path.join('../project_image_data/TEST/South')

# val_normal = os.path.join('../project_image_data/val/NORMAL')
# val_pneumonia = os.path.join('../project_image_data/val/PNEUMONIA')

all_paths = [train_east, train_west, train_mid_west, train_south]

# %% [markdown]
## Show number of files for each class in each folder
# %%

# for path in all_paths:
#     print(f'{path} has  {len(os.listdir(path))}  files')
for path in all_paths:
    png_count = len(glob.glob(os.path.join(path, '**/*.png'), recursive=True))
    print(path,':', png_count)    
# %%

# %% [markdown]
## Normalize image size and load into generator
# %%
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)

# test_val_datagen = ImageDataGenerator(rescale=1./255,
#                                       validation_split=0.2)



# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory('../capstone-data/music - Copy/TRAIN/',  # Source dir for training images
                                                  target_size=(64, 64),  # All images will be resized to 150x150
                                                  batch_size=4273, #2086
                                                #   color_mode='grayscale',
                                                  # Since we use categorical_crossentropy loss, we need binary labels
                                                  class_mode='categorical',
                                                  subset='training',
                                                  shuffle=True)

# val_generator = test_val_datagen.flow_from_directory('../capstone-data/music/TRAIN/', # This is th source dir for validation images
#                                                  target_size=(64, 64),  # All images will be resized to 150x150
#                                                  batch_size=852, #435
#                                                  color_mode='grayscale',
#                                                  # Since we use categorical_crossentropy loss, we need binary labels
#                                                  class_mode='categorical',
#                                                  subset='validation',
#                                                  shuffle=True)

# test_generator = test_val_datagen.flow_from_directory('../capstone-data/music/TRAIN/', # This is th source dir for validation images
#                                                  target_size=(64, 64),  # All images will be resized to 150x150
#                                                  batch_size=4273, #1752
#                                                  color_mode='grayscale',
#                                                  # Since we use categorical_crossentropy loss, we need binary labels
#                                                  class_mode='categorical',
#                                                 #  subset='training',
#                                                  shuffle=False)       

X_all,y_all = next(train_generator)
# X_test,y_test = next(test_generator)
# X_val,y_val = next(val_generator)


# %%
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=42, shuffle=True)
# %%
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# %%
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=42, shuffle=True) 
# %%

print(X_test.shape)
print(y_test.shape)
print(X_val.shape)
print(y_val.shape)
# %% [markdown]
## Show pictures


# %%
plotImages(X_train, y_train)
# print(y_train[:10])
# print(X_train[0].shape)
print(train_generator.class_indices)
# %%
## Verify labels
# %%
# %% [markdown]
## Begin modeling
# %%
%%time
earlystop = tf.keras.callbacks.EarlyStopping(patience=10, verbose=True)
Adam_32_32_32_D3_64 = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(64, 64, 3)),
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(64, 64, 3)),
    # tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    # tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    # tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    # tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    # tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dense(32, activation='relu'),
    # Only 3 output neurond. It will contain a value from 0-2
    tf.keras.layers.Dense(4, activation='softmax')
])

Adam_32_32_32_D3_64.summary()
Adam_32_32_32_D3_64.compile(loss='categorical_crossentropy',
              optimizer="Adam",
              metrics=['accuracy',tf.keras.metrics.Recall()])

checkpoint = ModelCheckpoint(filepath=r'../capstone-data/checkpoints/Adam_32_32_32_D3_64_', verbose=0, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=f'../capstone-data/logs/Adam_32_32_32_D3_64_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
history1 = Adam_32_32_32_D3_64.fit(
      X_train, 
      y_train,
    #   steps_per_epoch=8,  
      batch_size=64,
      epochs=200, #epochs=15
      verbose=0,
      callbacks=[earlystop, tensorboard, checkpoint],
      validation_data=(X_val, y_val))

plot_results(history1.history)
evaluate_results(Adam_32_32_32_D3_64)

# %%
'''predictions = RMSprop_32_64.predict(x=test_generator, verbose=2)
np.round(predictions)
cm = confusion_matrix(y_true=test_generator.classes, y_pred=np.argmax(predictions, axis=-1))'''
# %%
'''from sklearn import metrics
print(metrics.classification_report(y_test, predictions.round()))'''
# %%
%%time
Adam_32_32_128_D32 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

Adam_32_32_128_D32.summary()
Adam_32_32_128_D32.compile(loss='categorical_crossentropy',
              optimizer="Adam",
              metrics=['accuracy',tf.keras.metrics.Recall()])

checkpoint = ModelCheckpoint(filepath=r'../capstone-data/checkpoints/Adam_32_32_128_D32', verbose=0, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=f'../capstone-data/logs/Adam_32_32_128_D32{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
history2 = Adam_32_32_128_D32.fit(
      X_train, 
      y_train,
    #   steps_per_epoch=8,  
      batch_size=128,
      epochs=200, #epochs=15
      verbose=0,
      callbacks=[earlystop, tensorboard, checkpoint],
      validation_data=(X_val, y_val))

plot_results(history2.history)
evaluate_results(Adam_32_32_128_D32)
# %%

# %%

# %%
%%time
Adam_32_32_128_D16 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

Adam_32_32_128_D16.summary()
Adam_32_32_128_D16.compile(loss='categorical_crossentropy',
              optimizer="Adam",
              metrics=['accuracy',tf.keras.metrics.Recall()])

checkpoint = ModelCheckpoint(filepath=r'../capstone-data/checkpoints/Adam_32_32_128_D16', verbose=0, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=f'../capstone-data/logs/Adam_32_32_128_D16{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
history3 = Adam_32_32_128_D16.fit(
      X_train, 
      y_train,
    #   steps_per_epoch=8,  
      batch_size=128,
      epochs=200, #epochs=15
      verbose=0,
      callbacks=[earlystop, tensorboard, checkpoint],
      validation_data=(X_val, y_val))

plot_results(history3.history)
evaluate_results(Adam_32_32_128_D16)
# %%

# %%
train_generator = train_datagen.flow_from_directory('../capstone-data/music - Copy/TRAIN/',  # Source dir for training images
                                                  target_size=(256, 256),  # All images will be resized to 150x150
                                                  batch_size=4273, #2086
                                                #   color_mode='grayscale',
                                                  # Since we use categorical_crossentropy loss, we need binary labels
                                                  class_mode='categorical',
                                                  subset='training',
                                                  shuffle=True)
X_all,y_all = next(train_generator)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=42, shuffle=True)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=42, shuffle=True) 
# %%
%%time
Adam256_32_32_32_D3_64 = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    # tf.keras.layers.Conv2D(32, (7,7), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.Conv2D(32, (7,7), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(3,3),
    # The second convolution
    tf.keras.layers.Conv2D(32, (7,7), activation='relu'),
    # tf.keras.layers.Conv2D(32, (7,7), activation='relu'),
    # tf.keras.layers.Conv2D(32, (7,7), activation='relu'),
    tf.keras.layers.MaxPooling2D(3,3),
    # tf.keras.layers.Conv2D(32, (7,7), activation='relu'),
    tf.keras.layers.Conv2D(32, (7,7), activation='relu'),
    tf.keras.layers.MaxPooling2D(3,3),

    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dense(64, activation='relu'),
    # Only 3 output neurond. It will contain a value from 0-2
    tf.keras.layers.Dense(4, activation='softmax')
])

Adam256_32_32_32_D3_64.summary()
Adam256_32_32_32_D3_64.compile(loss='categorical_crossentropy',
              optimizer="Adam",
              metrics=['accuracy',tf.keras.metrics.Recall()])

checkpoint = ModelCheckpoint(filepath=r'../capstone-data/checkpoints/Adam_32_32_32_D3_64_', verbose=0, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=f'../capstone-data/logs/Adam_32_32_32_D3_64_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
history1 = Adam256_32_32_32_D3_64.fit(
      X_train, 
      y_train,
    #   steps_per_epoch=8,  
      batch_size=128,
      epochs=200, #epochs=15
      verbose=0,
      callbacks=[earlystop, tensorboard, checkpoint],
      validation_data=(X_val, y_val))

plot_results(history1.history)
evaluate_results(Adam256_32_32_32_D3_64)
# %%
# %%
Adam_32_64_128_256 = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    # tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1
    tf.keras.layers.Dense(4, activation='softmax')
])
Adam_32_64_128_256.summary()
Adam_32_64_128_256.compile(loss='categorical_crossentropy',
              optimizer="Adam",
              metrics=['accuracy',tf.keras.metrics.Recall()])

# %%
%%time
checkpoint = ModelCheckpoint(filepath=r'../capstone-data/checkpoints/Adam_32_64_128_256', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=f'../capstone-data/logs/Adam_32_64_128_256{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
history5 = Adam_32_64_128_256.fit(
      X_train, 
      y_train,
    #   steps_per_epoch=8,  
      batch_size=128,
      epochs=60, #epochs=15
      verbose=1,
      callbacks=[earlystop, tensorboard, checkpoint],
      validation_data=(X_val, y_val))

# %%
plot_results(history5.history)
evaluate_results(Adam_32_64_128_256)
# %%
Adam_32_64_64_64 = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    # 512 neuron hidden layer
  
    # tf.keras.layers.Dense(32, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1
    tf.keras.layers.Dense(4, activation='softmax')
])
Adam_32_64_64_64.summary()
Adam_32_64_64_64.compile(loss='categorical_crossentropy',
              optimizer="Adam",
              metrics=['accuracy',tf.keras.metrics.Recall()])

# %%
%%time
checkpoint = ModelCheckpoint(filepath=r'../capstone-data/checkpoints/Adam_32_64_64_64_', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=f'../capstone-data/logs/Adam_32_64_64_64_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
history6 = Adam_32_64_64_64.fit(
      X_train, 
      y_train,
    #   steps_per_epoch=8,  
      batch_size=128,
      epochs=60, #epochs=15
      verbose=1,
      callbacks=[earlystop, tensorboard, checkpoint], 
      validation_data=(X_val, y_val))

# %%
plot_results(history6.history)
evaluate_results(Adam_32_64_64_64)
# %%
earlystop = tf.keras.callbacks.EarlyStopping(patience=5, verbose=True)
Adam_32_64_64_64_P5 = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),


    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    # tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1
    tf.keras.layers.Dense(4, activation='softmax')
])
Adam_32_64_64_64_P5.summary()
Adam_32_64_64_64_P5.compile(loss='categorical_crossentropy',
              optimizer="Adam",
              metrics=['accuracy',tf.keras.metrics.Recall()])

# %%
%%time
checkpoint = ModelCheckpoint(filepath=r'../capstone-data/checkpoints/Adam_32_64_64_64_P5_', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=f'../capstone-data/logs/Adam_32_64_64_64_P5_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
history7 = Adam_32_64_64_64_P5.fit(
      X_train, 
      y_train,
    #   steps_per_epoch=8,  
      batch_size=128,
      epochs=60, #epochs=15
      verbose=1,
      callbacks=[tensorboard, earlystop, checkpoint],
      validation_data=(X_val, y_val))

# %%
plot_results(history7.history)
evaluate_results(Adam_32_64_64_64_P5)
# %%
earlystop = tf.keras.callbacks.EarlyStopping(patience=3, verbose=True)
Adam_32_32_32_32 = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),


    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    # tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1
    tf.keras.layers.Dense(4, activation='softmax')
])
Adam_32_32_32_32.summary()
Adam_32_32_32_32.compile(loss='categorical_crossentropy',
              optimizer="Adam",
              metrics=['accuracy',tf.keras.metrics.Recall()])

# %%
%%time
checkpoint = ModelCheckpoint(filepath=r'../capstone-data/checkpoints/Adam_32_32_32_32_', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=f'../capstone-data/logs/Adam_32_32_32_32_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
history8 = Adam_32_32_32_32.fit(
      X_train, 
      y_train,
    #   steps_per_epoch=8,  
      batch_size=128,
      epochs=60, #epochs=15
      verbose=1,
      callbacks=[tensorboard, earlystop, checkpoint],
      validation_data=(X_val, y_val))

# %%
plot_results(history8.history)
evaluate_results(Adam_32_32_32_32)
# %% [markdown]
### Use LIME to show 'feature' selection in a sense
# %%
Adam_32_32_32_32_2D = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),


    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(64, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1
    tf.keras.layers.Dense(4, activation='softmax')
])
# %%
Adam_32_32_32_32_2D.summary()
# %%
Adam_32_32_32_32_2D.compile(loss='categorical_crossentropy',
              optimizer="Adam",
              metrics=['accuracy',tf.keras.metrics.Recall()])

# %%
%%time
checkpoint = ModelCheckpoint(filepath=r'../capstone-data/checkpoints/Adam_32_32_32_32_2D_', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=f'../capstone-data/logs/Adam_32_32_32_32_2D_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
history9 = Adam_32_32_32_32_2D.fit(
      X_train, 
      y_train,
      batch_size=128,
    #   steps_per_epoch=8,  
      epochs=60, #epochs=15
      verbose=1,
      callbacks=[earlystop, tensorboard, checkpoint],
      validation_data=(X_val, y_val))

# %%
plot_results(history9.history)
evaluate_results(Adam_32_32_32_32_2D)
# %%
# visualkeras.layered_view(Adam_32_32_32_32, to_file='network_visual.png').show()
# %% [markdown]
# After iterating over and running all the models I viewed the results of each as well as their compared accuracy/loss graphs via tensorboard. I found that the "Adam_32_32_32_32" model performed the best and stored the model within a separate filepath to ensure it doesn't get written over or compromised.
# %%
best_model = tf.keras.models.load_model('Adam_32_32_32_32__best')

# %%
# earlystop = tf.keras.callbacks.EarlyStopping(patience=3, verbose=True)
# checkpoint = ModelCheckpoint(filepath=r'../capstone-data/checkpoints/testloadd.hdf5', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
# tensorboard = TensorBoard(log_dir=f'../capstone-data/logs/Adam_32_64_64_64_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
# test_HISTORY = best_model.fit(
#       X_train, 
#       y_train,
#     #   steps_per_epoch=8,  
#       batch_size=128,
#       epochs=100, #epochs=15
#       verbose=1,
#       callbacks=[earlystop],
#       validation_data=(X_val, y_val))
# %%
evaluate_results(best_model)
# %% [markdown]
## Best Model
# I selected this as the best model because it had the highest accuracy at 88%. While this model does not have the highest recall which is what I would like to maximize, I believe sacrificing 4 percentage points of correctly identifying pneumonia was worth the 25% increase of correctly identifying normal lungs. 
# %% [markdown]
## Creating network topology visual 
# %%
color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = 'rgb(128,191,183)'
color_map[ZeroPadding2D]['fill'] = 'gray'
color_map[Dropout]['fill'] = 'pink'
color_map[MaxPooling2D]['fill'] = 'rgb(11,135,161)'
color_map[Dense]['fill'] = '#a25d71'#'#928d6d'
color_map[Flatten]['fill'] = '#6d7292' #'#6d7292'

visualkeras.layered_view(best_model, color_map=color_map)
# %%
# visualkeras.layered_view(best_model, to_file='network_visual.png', color_map=color_map).show()

# %% [markdown]
## Create ROC Curve
# %%
preds = best_model.predict(X_test)
fpr, tpr, thresh = roc_curve(y_test, preds)

# Calculate the ROC (Reciever Operating Characteristic) AUC (Area Under the Curve)
rocauc = auc(fpr, tpr)
print('Train ROC AUC Score: ', rocauc)
# %%
fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
#     marker=dict(color='fa7f72'),

    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain', tickvals=[0,0.25,0.5,0.75,1])

fig.show()

# %% [markdown]
# ROC curve restates that to get a 95% true positive, we would need to be comfortable accepting a 25% false positive rate.
# %% [markdown]
## Recommendations

# Based on my results throughout this process there are a few things I'd recommend if you were to go about building a model to predict pneumonia. 

# * More data
#   * While there were about 5 thousands images total given within the dataset. I believe that better results could be gathered if there was more data for the model to train on. CNN's will always benefit from more data as long as there are enough resources available to process it effectively.
# * Transfer Learning
#   * Using transfer learning from a model that was pre-trained on x-rays with and without pneumonia would yield better results in less time and with less resources. You'll spend less time iterating on network variations as the pre-trained model will already have done most of the work. Fine tuning will take considerably less time and resources and yield just as good if not better results from the right model.
# * Resources
#   * Before you decide to get into CNN's, be sure you have the appropriate resources available to you. Training a CNN over thousands of images is going to take a long time and a lot your computer's resources if you don't have sufficient RAM or GPU. If you don't have the hardware available to you, there are cloud services such as Google Colab that offer access GPU processers virtually through the cloud at low to no cost. 

# %% [markdown]
## Leaving below code for review
# %%
from img_classification import teachable_machine_classification
from PIL import Image, ImageOps
image = Image.open(r'D:\Python_Projects\flatiron\class-materials\phase04\project_image_data\test\NORMAL\IM-0006-0001.jpeg')
weights_file = "./Adam_32_32_32_32__best"

label = teachable_machine_classification(image, weights_file)
print(label)
if label == 0:
    print("The image has pneumonia")
else:
    print("The image is healthy")
# %%
import lime
from lime import lime_image
from lime import lime_base

from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
# %%
# i = np.random.choice(range(len(y_train)))
label = y_train[24]
img = X_train[24]
# %%
pred = Adam_32_64_64_64.predict(np.array([img]))
pred_class = int(pred.round())
# %%
# labels = train_generator.class_indices
print(f"Image 24 = {label}: Pneumonia")
print(f"Model Predicts {pred_class}")
array_to_img(img)

# array_to_img(img)
# %%
explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(img, Adam_32_64_64_64.predict, top_labels=2,
                                         hide_color=None, num_samples=2000)

# %%