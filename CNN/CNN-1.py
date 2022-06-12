import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping

(x_train,y_train),(x_test,y_test) = mnist.load_data()



# transform into binary class 
y_cat_test  = to_categorical(y_test ,num_classes=10)
y_cat_train = to_categorical(y_train,num_classes=10)


# remap values
x_train = x_test/255
x_test  = x_test/255
# batch size, width, height, channels
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] , x_train.shape[2],1)
x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1] , x_test.shape[2],1)

#print(y_cat_train.shape)
#print(x_train.shape)
#print(' ')
#print(y_cat_test.shape)
#print(x_test.shape)
# init model
model = Sequential()
# add convolutional layer
model.add(Conv2D(filters=32,kernel_size=(4,4),strides=(1,1),input_shape=(28,28,1),activation='relu'))
# add pooling layer
model.add(MaxPool2D(pool_size=(2,2)))
# layer between CNN and ANN
model.add(Flatten())
# ANN
model.add(Dense(128,activation='relu'))
# output : softmax -> multiclass probleme
model.add(Dense(10,activation='softmax'))
# compile it
model.compile(loss='catergorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# create early stopping function
early_stop = EarlyStopping(monitor='val_loss',patience=1)
# launch model
model.fit(x_train,y_cat_train,epochs=10,validation_data=(x_test,y_cat_test),callbacks=early_stop)




