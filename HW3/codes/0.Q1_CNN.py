#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 10:24:05 2020

@author: zzhajun
"""
# =============================================================================
# 1.a
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
#loading dataset
from keras.datasets import mnist 
(train_X , train_Y), (test_X , test_Y) = mnist.load_data ()
#printing out the shape
print(np.shape(train_X))
print(np.shape(test_X))
#visualize images
fig=plt.gcf()
fig.set_size_inches(6,7)
num=10
idx=0
for i in range(0,num):
    ax=plt.subplot(5,5,i+1)
    ax.imshow(train_X[idx].reshape(28,28),cmap='binary')
    title='label='+str(train_Y[idx])
            
    ax.set_title(title,fontsize=10)
    ax.set_xticks([]);ax.set_yticks([])
    idx+=1
plt.show()

fig=plt.gcf()
fig.set_size_inches(6,7)
num=10
idx=0
for i in range(0,num):
    ax=plt.subplot(5,5,i+1)
    ax.imshow(test_X[idx].reshape(28,28),cmap='binary')
    title='label='+str(test_Y[idx])
            
    ax.set_title(title,fontsize=10)
    ax.set_xticks([]);ax.set_yticks([])
    idx+=1
plt.show()


# =============================================================================
# 1.b
# =============================================================================
from keras.utils import np_utils
#reshape
train_X=train_X.reshape(60000,28,28,1).astype('float32')
test_X=test_X.reshape(10000,28,28,1).astype('float32')
#scale the pixel values
X_train_normalize=train_X/255
X_test_normalize=test_X/255
#one-hot
train_Y_onehot=np_utils.to_categorical(train_Y)
test_Y_onehot=np_utils.to_categorical(test_Y)
#visualize
fig=plt.gcf()
fig.set_size_inches(6,7)
num=4
idx=0
for i in range(0,num):
    ax=plt.subplot(2,2,i+1)
    ax.imshow(X_train_normalize[idx].reshape(28,28),cmap='binary')
    title='label='+str(train_Y_onehot[idx])
            
    ax.set_title(title,fontsize=10)
    ax.set_xticks([]);ax.set_yticks([])
    idx+=1
plt.show()

fig=plt.gcf()
fig.set_size_inches(6,7)
num=4
idx=0
for i in range(0,num):
    ax=plt.subplot(2,2,i+1)
    ax.imshow(X_test_normalize[idx].reshape(28,28),cmap='binary')
    title='label='+str(test_Y_onehot[idx])
            
    ax.set_title(title,fontsize=10)
    ax.set_xticks([]);ax.set_yticks([])
    idx+=1
plt.show()

# =============================================================================
# 1.c
# =============================================================================


from keras.models import Sequential 
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D
from keras.layers import Dense 
from keras.layers import Flatten 
from keras.optimizers import SGD

def create_cnn ():
    # define using Sequential 
    model = Sequential () 
    # Convolution layer 
    model.add(
            Conv2D (32, (3, 3),
            activation='relu',
            kernel_initializer='he_uniform',
            input_shape =(28, 28, 1))
            ) 
    # Maxpooling layer 
    model.add(MaxPooling2D ((2, 2))) 
    # Flatten output 
    model.add(Flatten ()) 
    # Dense layer of 100 neurons 
    model.add(
            Dense (100,
            activation='relu',
            kernel_initializer='he_uniform')
            ) 
    model.add(Dense (10, activation='softmax')) 
    # initialize optimizer 
    opt = SGD(lr=0.01 , momentum =0.9) 
    # compile model 
    model.compile(
            optimizer=opt ,
            loss='categorical_crossentropy',
            metrics =['accuracy']
            ) 
    print(model.layers)
    return model
create_cnn()

# =============================================================================
# 1.d
# =============================================================================

model=create_cnn()
model.fit(X_train_normalize,train_Y_onehot, batch_size =32, epochs =10, validation_split =0.1)
score = model.evaluate(X_test_normalize, test_Y_onehot, verbose =0)
print(score)

# =============================================================================
# 1.e.1
# =============================================================================

from keras.datasets import mnist 
(train_X , train_Y), (test_X , test_Y) = mnist.load_data ()
#reshape
train_X=train_X.reshape(60000,28,28,1).astype('float32')
test_X=test_X.reshape(10000,28,28,1).astype('float32')
#scale the pixel values
X_train_normalize=train_X/255
X_test_normalize=test_X/255
#one-hot
train_Y_onehot=np_utils.to_categorical(train_Y)
test_Y_onehot=np_utils.to_categorical(test_Y)
def create_cnn ():
    # define using Sequential 
    model = Sequential () 
    # Convolution layer 
    model.add(
            Conv2D (32, (3, 3),
            activation='relu',
            kernel_initializer='he_uniform',
            input_shape =(28, 28, 1))
            ) 
    #print(model.layers)
    # Maxpooling layer 
    model.add(MaxPooling2D ((2, 2))) 
    #print(model.layers)
    # Flatten output 
    model.add(Flatten ()) 
    #print(model.layers)
    # Dense layer of 100 neurons 
    model.add(
            Dense (100,
            activation='relu',
            kernel_initializer='he_uniform')
            ) 
    #print(model.layers)
    model.add(Dense (10, activation='softmax')) 
    #print(model.layers)
    # initialize optimizer 
    opt = SGD(lr=0.01 , momentum =0.9) 
    # compile model 
    model.compile(
            optimizer=opt ,
            loss='categorical_crossentropy',
            metrics =['accuracy']
            ) 
    #print(model.layers)
    return model
model_1=create_cnn()
epoch_history_1 = model_1.fit(X_train_normalize , train_Y_onehot, batch_size =32, epochs =50, validation_split =0.1)
print(epoch_history_1.history['accuracy']) 
print(epoch_history_1.history['val_accuracy'])
a_1=epoch_history_1.history['accuracy']
b_1=epoch_history_1.history['val_accuracy']
acc_1=[]
val_acc_1=[]
for i in range(50):
    if (i+1)%10==0:
        acc_1.append(a_1[i])
        val_acc_1.append(b_1[i])
x=[10,20,30,40,50]
def show_epoch_history(epoch_history,train,validation):

    plt.plot(x,acc_1)
    plt.plot(x,val_acc_1)
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

show_epoch_history(epoch_history_1,'accuracy','val_accuracy')    
'''
def show_epoch_history(epoch_history,train,validation):

    plt.plot(a_1)
    plt.plot(b_1)
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

show_epoch_history(epoch_history,'accuracy','val_accuracy')   
'''
# print validation and training accuracy over epochs 

# =============================================================================
# 1.e.2
# =============================================================================

from keras.layers import Dropout
from keras.datasets import mnist 
(train_X , train_Y), (test_X , test_Y) = mnist.load_data ()
#reshape
train_X=train_X.reshape(60000,28,28,1).astype('float32')
test_X=test_X.reshape(10000,28,28,1).astype('float32')
#scale the pixel values
X_train_normalize=train_X/255
X_test_normalize=test_X/255
#one-hot
train_Y_onehot=np_utils.to_categorical(train_Y)
test_Y_onehot=np_utils.to_categorical(test_Y)
def create_cnn ():
    # define using Sequential 
    model = Sequential () 
    # Convolution layer 
    model.add(
            Conv2D (32, (3, 3),
            activation='relu',
            kernel_initializer='he_uniform',
            input_shape =(28, 28, 1))
            ) 
    #print(model.layers)
    # Maxpooling layer 
    model.add(MaxPooling2D ((2, 2))) 
    #print(model.layers)
    # Flatten output 
    model.add(Flatten ()) 
    #print(model.layers)
    model.add(Dropout (0.5))
    # Dense layer of 100 neurons 
    model.add(
            Dense (100,
            activation='relu',
            kernel_initializer='he_uniform')
            ) 
    #print(model.layers)
    model.add(Dense (10, activation='softmax')) 
    #print(model.layers)
    # initialize optimizer 
    opt = SGD(lr=0.01 , momentum =0.9) 
    # compile model 
    model.compile(
            optimizer=opt ,
            loss='categorical_crossentropy',
            metrics =['accuracy']
            ) 
    #print(model.layers)
    return model
model_2=create_cnn()
epoch_history_2 = model_2.fit(X_train_normalize , train_Y_onehot, batch_size =32, epochs =50, validation_split =0.1)
print(epoch_history_2.history['accuracy']) 
print(epoch_history_2.history['val_accuracy'])
a_2=epoch_history_2.history['accuracy']
b_2=epoch_history_2.history['val_accuracy']
acc_2=[]
val_acc_2=[]
for i in range(50):
    if (i+1)%10==0:
        acc_2.append(a_2[i])
        val_acc_2.append(b_2[i])
x=[10,20,30,40,50]
def show_epoch_history(epoch_history,train,validation):

    plt.plot(x,acc_2)
    plt.plot(x,val_acc_2)
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

show_epoch_history(epoch_history_2,'accuracy','val_accuracy')    
'''
def show_epoch_history_2(epoch_history,train,validation):

    plt.plot(epoch_history_2.history['accuracy'])
    plt.plot(epoch_history_2.history['val_accuracy'])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

show_epoch_history_2(epoch_history_2,'accuracy','val_accuracy')  
'''  
# =============================================================================
# 1.e.3
# =============================================================================

from keras.datasets import mnist 
(train_X , train_Y), (test_X , test_Y) = mnist.load_data ()
#reshape
train_X=train_X.reshape(60000,28,28,1).astype('float32')
test_X=test_X.reshape(10000,28,28,1).astype('float32')
#scale the pixel values
X_train_normalize=train_X/255
X_test_normalize=test_X/255
#one-hot
train_Y_onehot=np_utils.to_categorical(train_Y)
test_Y_onehot=np_utils.to_categorical(test_Y)
def create_cnn ():
    # define using Sequential 
    model = Sequential () 
    # Convolution layer 
    model.add(
            Conv2D (32, (3, 3),
            activation='relu',
            kernel_initializer='he_uniform',
            input_shape =(28, 28, 1))
            ) 
    # Maxpooling layer 
    model.add(MaxPooling2D ((2, 2))) 
    # Add another convolution layer
    model.add(
            Conv2D(64,(3,3),
            activation='relu',
            kernel_initializer='he_uniform',
            input_shape =(28, 28, 1))
            )
    model.add(MaxPooling2D((2,2)))
    # Flatten output 
    model.add(Flatten ()) 
    #print(model.layers)
    model.add(Dropout (0.5))
    # Dense layer of 100 neurons 
    model.add(
            Dense (100,
            activation='relu',
            kernel_initializer='he_uniform')
            ) 
    #print(model.layers)
    model.add(Dense (10, activation='softmax')) 
    #print(model.layers)
    # initialize optimizer 
    opt = SGD(lr=0.01 , momentum =0.9) 
    # compile model 
    model.compile(
            optimizer=opt ,
            loss='categorical_crossentropy',
            metrics =['accuracy']
            ) 
    #print(model.layers)
    return model
model=create_cnn()
model.fit(X_train_normalize,train_Y_onehot, batch_size =32, epochs =10, validation_split =0.1)
score = model.evaluate(X_test_normalize, test_Y_onehot, verbose =0)
print(score)

# =============================================================================
# 1.e.4
# =============================================================================
#0.001 learning rate
from keras.datasets import mnist 
(train_X , train_Y), (test_X , test_Y) = mnist.load_data ()
#reshape
train_X=train_X.reshape(60000,28,28,1).astype('float32')
test_X=test_X.reshape(10000,28,28,1).astype('float32')
#scale the pixel values
X_train_normalize=train_X/255
X_test_normalize=test_X/255
#one-hot
train_Y_onehot=np_utils.to_categorical(train_Y)
test_Y_onehot=np_utils.to_categorical(test_Y)
def create_cnn ():
    # define using Sequential 
    model = Sequential () 
    # Convolution layer 
    model.add(
            Conv2D (32, (3, 3),
            activation='relu',
            kernel_initializer='he_uniform',
            input_shape =(28, 28, 1))
            ) 
    # Maxpooling layer 
    model.add(MaxPooling2D ((2, 2))) 
    # Add another convolution layer
    model.add(
            Conv2D(64,(3,3),
            activation='relu',
            kernel_initializer='he_uniform',
            input_shape =(28, 28, 1))
            )
    model.add(MaxPooling2D((2,2)))
    # Flatten output 
    model.add(Flatten ()) 
    #print(model.layers)
    model.add(Dropout (0.5))
    # Dense layer of 100 neurons 
    model.add(
            Dense (100,
            activation='relu',
            kernel_initializer='he_uniform')
            ) 
    #print(model.layers)
    model.add(Dense (10, activation='softmax')) 
    #print(model.layers)
    # initialize optimizer 
    opt = SGD(lr=0.001 , momentum =0.9) 
    # compile model 
    model.compile(
            optimizer=opt ,
            loss='categorical_crossentropy',
            metrics =['accuracy']
            ) 
    #print(model.layers)
    return model
model=create_cnn()
model.fit(X_train_normalize,train_Y_onehot, batch_size =32, epochs =10, validation_split =0.1)
score = model.evaluate(X_test_normalize, test_Y_onehot, verbose =0)
print(score)

#0.1 learning rate
from keras.datasets import mnist 
(train_X , train_Y), (test_X , test_Y) = mnist.load_data ()
#reshape
train_X=train_X.reshape(60000,28,28,1).astype('float32')
test_X=test_X.reshape(10000,28,28,1).astype('float32')
#scale the pixel values
X_train_normalize=train_X/255
X_test_normalize=test_X/255
#one-hot
train_Y_onehot=np_utils.to_categorical(train_Y)
test_Y_onehot=np_utils.to_categorical(test_Y)
def create_cnn ():
    # define using Sequential 
    model = Sequential () 
    # Convolution layer 
    model.add(
            Conv2D (32, (3, 3),
            activation='relu',
            kernel_initializer='he_uniform',
            input_shape =(28, 28, 1))
            ) 
    # Maxpooling layer 
    model.add(MaxPooling2D ((2, 2))) 
    # Add another convolution layer
    model.add(
            Conv2D(64,(3,3),
            activation='relu',
            kernel_initializer='he_uniform',
            input_shape =(28, 28, 1))
            )
    model.add(MaxPooling2D((2,2)))
    # Flatten output 
    model.add(Flatten ()) 
    #print(model.layers)
    model.add(Dropout (0.5))
    # Dense layer of 100 neurons 
    model.add(
            Dense (100,
            activation='relu',
            kernel_initializer='he_uniform')
            ) 
    #print(model.layers)
    model.add(Dense (10, activation='softmax')) 
    #print(model.layers)
    # initialize optimizer 
    opt = SGD(lr=0.1 , momentum =0.9) 
    # compile model 
    model.compile(
            optimizer=opt ,
            loss='categorical_crossentropy',
            metrics =['accuracy']
            ) 
    #print(model.layers)
    return model
model=create_cnn()
model.fit(X_train_normalize,train_Y_onehot, batch_size =32, epochs =10, validation_split =0.1)
score = model.evaluate(X_test_normalize, test_Y_onehot, verbose =0)
print(score)