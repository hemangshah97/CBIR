from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from PIL import Image
from numpy import *
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

path1='D:\project\input_data'
path2='D:\project\input_data_resized'

listing=os.listdir(path1)
num_samples=size(listing)
print (num_samples)

img_rows,img_cols=256,256

for file in listing:
    im=Image.open(path1 + '\\' + file)
    img=im.resize((img_rows,img_cols))
    gray=img.convert('L')
    gray.save(path2 + '\\' + file,"JPEG")
    
imlist=os.listdir(path2)

im1=array(Image.open('input_data_resized'+'\\'+imlist[0]))
m,n=im1.shape[0:2]
imnbr=len(imlist)

immatrix = array([array(Image.open('input_data_resized'+'\\'+im2)).flatten() for im2 in imlist],'f')

label=np.ones((num_samples,),dtype=int)
label[0:100]=0
label[100:200]=1
label[200:300]=2
label[300:400]=3
label[400:500]=4
label[500:600]=5
label[600:700]=6
label[700:800]=7
label[800:900]=8
label[900:1000]=9

data,label = shuffle(immatrix,label,random_state=2)
train_data=[data,label]
img=immatrix[550].reshape(img_rows,img_cols)
plt.imshow(img,cmap='gray')
print (train_data[0].shape)
print (train_data[1].shape)

batch_size = 32
nb_classes=10
nb_epoch=20
img_channels=1
nb_filters=32
nb_pool=2
nb_conv=3

(X,y)=(train_data[0],train_data[1])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=4)
X_train=X_train.reshape(X_train.shape[0],1,img_rows,img_cols)
X_test=X_test.reshape(X_test.shape[0],1,img_rows,img_cols)

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:',X_train.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')

Y_train=np_utils.to_categorical(y_train,nb_classes)
Y_test=np_utils.to_categorical(y_test,nb_classes)

i=100
plt.imshow(X_train[i,0], interpolation='nearest')
print("label : ",Y_train[i,:])

model = Sequential()

model.add(Convolution2D(nb_filters,nb_conv,nb_conv,border_mode='valid',input_shape=(img_rows,img_cols,1)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters,nb_conv,nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

model.fit(X_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(X_test,Y_test))
model.fit(X_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_split=0.1)

score=model.evaluate(X_test,Y_test,show_accuracy=True,verbose=0)
print('Test score =',score[0])
print('Test accuracy =',score[1])
print(model.predice_classes(X_test[1:5]))
print(Y_test[1:5])