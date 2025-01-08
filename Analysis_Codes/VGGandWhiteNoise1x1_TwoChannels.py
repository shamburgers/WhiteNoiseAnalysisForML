#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Here, we just copy and pasted the preamble on the given notebook in colab

import h5py
import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential, Model
from keras.initializers import TruncatedNormal
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import scipy.special as sp
from functools import partial


# In[2]:


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[3]:


import h5py

electron = h5py.File("/bighome/rgsohal/Electron.hdf5", "r")
photon = h5py.File("/bighome/rgsohal/Photon.hdf5", "r")


# In[4]:


X = np.concatenate([electron['X'], photon['X']])
y = np.concatenate([electron['y'], photon['y']])


# In[5]:


#Split data


# In[6]:


X_train, X_val1, y_train , y_val1 = train_test_split(X,y,train_size = 0.8, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_val1, y_val1, train_size = 0.5, random_state = 42)
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

# In[100]:

from tensorflow.math.special import bessel_j0, bessel_j1
from tensorflow.math import sin, cos

class WhiteNoiseConvolution2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernels=None,
                 kernel_initializer=None,
                 strides = 1,
                 activation="relu",
                 **kwargs):
        super(WhiteNoiseConvolution2D, self).__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.filters = filters
        self.strides = strides
        self.kernels = kernels
        self.kernel_initializer = kernel_initializer
        self.initializer = tf.keras.initializers.RandomUniform(minval=1, maxval=5, seed = 300)
        self.constraint = tf.keras.constraints.MinMaxNorm(min_value = -0.4, 
                                                          max_value = 1.0, rate = 1.0)


    def build(self, input_shape):    
       
        kernel_shape = self.kernels[0], self.kernels[1], input_shape[-1], self.filters 

        self.kernel = self.add_weight(
            shape=kernel_shape,
#             initializer=self.initializer,
            initializer = self.kernel_initializer,
	    constraint = self.constraint,
            trainable=True
        )
    def call(self, inputs): 
        kernel_h = self.kernels[0]
        kernel_w = self.kernels[1]
        kernel_in_ch = tf.shape(self.kernel)[2]
        kernel_out_ch = tf.shape(self.kernel)[3]

        patches = tf.image.extract_patches(images = inputs,
                                             sizes=[1, kernel_h, kernel_w, 1],
                                             strides=[1, self.strides[0], self.strides[1], 1],
                                             rates=[1, 1, 1, 1],   # dilated conv
                                             padding="SAME")
        
#         print("Patches Shape:", patches.shape)
        batch = tf.shape(inputs)[0]
        img_h = patches.shape[1]
        img_w = patches.shape[2]
        patches = tf.reshape(patches, [batch, 1, img_h*img_w, kernel_h*kernel_w*kernel_in_ch])
        
        kernel = tf.reshape(self.kernel, [1, 1, kernel_h*kernel_w*kernel_in_ch, kernel_out_ch])
        kernel = tf.transpose(kernel, [0, 3, 1, 2])
        kernel = tf.tile(kernel, [batch,1,1,1])

        patches = tf.math.divide( bessel_j1(patches), (patches + 1) )
        mul = tf.multiply(patches, kernel)
        result = tf.reduce_sum(mul, 3)

        result = tf.reshape(result, [batch, img_h, img_w, kernel_out_ch])
#         print("--------------------------------------------------")
        return result

    def get_config(self):
        config = super(WhiteNoiseConvolution2D, self).get_config()
        config.update({
            "filters":self.filters,
            "kernels":self.kernels,
            "kernel_initializer":self.kernel_initializer,
            })
        return config

# In[103]:


input_shape = (32, 32, 2)
inputs = Input(shape = input_shape, batch_size = None)
# model1 = tf.keras.model1s.Sequential()
# model1 = GaussianNoise(0.01)(inputs)
model1 = Conv2D(name="conv1", filters=32, kernel_size = (1,1), strides = (1,1))(inputs)
model1 = Activation("relu")(model1)
model1 = Conv2D(name="conv2", filters=32, kernel_size = (1,1), strides = (1,1))(model1)
model1 = Activation("relu")(model1)
model1 = MaxPooling2D(pool_size=3, strides=2, padding="SAME")(model1)
model1 = Conv2D(name="conv3", filters=64, kernel_size = (1,1), strides = (1,1))(model1)
model1 = Activation("relu")(model1)
model1 = Conv2D(name="conv4", filters=64, kernel_size = (1,1), strides = (1,1))(model1)
model1 = Activation("relu")(model1)
model1 = MaxPooling2D(pool_size=3, strides=2, padding="SAME")(model1)
model1 = Flatten()(model1)
model1 = Dense(256, activation='relu', kernel_initializer='TruncatedNormal')(model1)
model1 = Dropout(0.2)(model1)
model1 = Dense(256, activation='relu', kernel_initializer='TruncatedNormal')(model1)
model1 = Dropout(0.2)(model1)
output = Dense(1, activation='sigmoid')(model1)
model1 = Model(inputs = inputs, outputs = output)
model1.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=6, min_lr=1.e-6)
early_stopping = EarlyStopping(monitor="val_loss", patience = 25, restore_best_weights=True)
callbacks = [reduce_lr, early_stopping]
model1.summary()

# In[104]:


history1 = model1.fit(X_train, y_train,\
        batch_size=64,\
        epochs=100,\
        validation_data=(X_val, y_val),\
        callbacks=callbacks)


# In[86]:


# Evaluate on test set
score = model1.evaluate(X_test, y_test, verbose=1)
print('\nTest loss / accuracy: %0.4f / %0.4f'%(score[0], score[1]))
y_pred = model1.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print('Test ROC AUC (VGG 1x1 Kernel):', roc_auc)


# In[87]:


plt.plot([0, 1], [0, 1], 'k--')
#plt.legend(loc=2, prop={'size': 15})
plt.plot(fpr, tpr, label='(ROC-AUC (VGG 1x1 Kernel)= {:.6f})'.format(roc_auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
plt.savefig("/bighome/rgsohal/VGG1x1_TwoChannels.png", dpi=1200)


input_shape = (32, 32, 2)
inputs = Input(shape = input_shape, batch_size = None)
# model2 = tf.keras.model2s.Sequential()
# model2 = GaussianNoise(0.01)(inputs)
model2 = WhiteNoiseConvolution2D(name="conv1", filters=32, kernels = (1,1), strides = (1,1))(inputs)
model2 = Activation("relu")(model2)
model2 = WhiteNoiseConvolution2D(name="conv2", filters=32, kernels = (1,1), strides = (1,1))(model2)
model2 = Activation("relu")(model2)
model2 = MaxPooling2D(pool_size=3, strides=2, padding="SAME")(model2)
model2 = WhiteNoiseConvolution2D(name="conv3", filters=64, kernels = (1,1), strides = (1,1))(model2)
model2 = Activation("relu")(model2)
model2 = WhiteNoiseConvolution2D(name="conv4", filters=64, kernels = (1,1), strides = (1,1))(model2)
model2 = Activation("relu")(model2)
model2 = MaxPooling2D(pool_size=3, strides=2, padding="SAME")(model2)
model2 = Flatten()(model2)
model2 = Dense(256, activation='relu', kernel_initializer='TruncatedNormal')(model2)
model2 = Dropout(0.2)(model2)
model2 = Dense(256, activation='relu', kernel_initializer='TruncatedNormal')(model2)
model2 = Dropout(0.2)(model2)
output = Dense(1, activation='sigmoid')(model2)
model2 = Model(inputs = inputs, outputs = output)
model2.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=6, min_lr=1.e-6)
early_stopping = EarlyStopping(monitor="val_loss", patience = 25, restore_best_weights=True)
callbacks = [reduce_lr, early_stopping]
model2.summary()


history2 = model2.fit(X_train, y_train,\
        batch_size=64,\
        epochs=100,\
        validation_data=(X_val, y_val),\
        callbacks=callbacks)



# Evaluate on test set
score = model2.evaluate(X_test, y_test, verbose=1)
print('\nTest loss / accuracy: %0.4f / %0.4f'%(score[0], score[1]))
y_pred = model2.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print('Test ROC AUC (White Noise 1x1 Kernel):', roc_auc)


# In[87]:


plt.plot([0, 1], [0, 1], 'k--')
#plt.legend(loc=2, prop={'size': 15})
plt.plot(fpr, tpr, label='(ROC-AUC (White Noise 1x1 Kernel)= {:.6f})'.format(roc_auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
plt.savefig("/bighome/rgsohal/WhiteNoise1x1_TwoChannels.png", dpi=1200)



