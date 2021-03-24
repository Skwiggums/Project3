import sys
import random
import numpy as np
import pandas as pd
import PIL as pl#pillow's module name isn't pillow, it's PIL, took me an hour to figure that out
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers

##pulling in test and training data and combining into tensorflow datasets
val_labels=pd.read_csv("fairface_label_val.csv")
val_labels=val_labels.to_numpy()#keras can work with datasets or arrays but not both. Converting to array
train_labels=pd.read_csv("fairface_label_train.csv")
train_labels=train_labels.to_numpy()#keras can work with datasets or arrays but not both. Converting to array
train_labels=train_labels[:,0:4]
val_labels=val_labels[:,0:4]#removing service test category since tensorflow doesn't like bools

#giving string entries numerical values will correspond to index in label array
#age 0=0-2, 1=10-19, 2=20-29, 3=3-9, 4=30-39, 5=40-49, 6=50-59, 7=60-69, 8=more than 70
#gender 0=male, 1=female
#race 0=Black, 1=East Asian, 2=Indian, 3=Latino_Hispanic, 4=Middle Eastern, 5= Southeast Asian, 6=White
agelabels=np.unique(train_labels[:,1])
genderlabels=np.unique(train_labels[:,2])
racelabels=np.unique(train_labels[:,3])
labels=[agelabels,genderlabels,racelabels]#all possible categories to be classified

class labelvalues:
    def _init_(self):
        print('init')
    #encode array should be one of the category label arrays, data array should be one of the
    #label arrays used in training or validation
    def encode(self,dataarray,encodearray):
        self.dataarray=dataarray
        self.encodearray=encodearray
        for i in range(np.size(self.encodearray)):
            indices=(self.dataarray[:]==self.encodearray[i])
            self.dataarray[indices]=i
        return self.dataarray
    def extract(self,dataarray,encodearray):
        self.dataarray=dataarray
        self.encodearray=encodearray
        for i in range(np.size(self.encodearray)):
            indices=(self.dataarray[:]==i)
            self.dataarray[indices]=self.encodearray[i]
        return self.dataarray
#a=(train_labels[:,2]=='Male')
#train_labels[a,2]=1


train_images = [] #got this from stack exchange:https://stackoverflow.com/questions/50557468/load-all-images-from-a-folder-using-pil
for f in glob.iglob("C:/Users/vwilso17/Documents/Grad School/COSC525Deeplearning/code/projects/project3/train/*"):
    train_images.append(np.asarray(pl.Image.open(f)))
train_images = np.array(train_images)

val_images = [] #got this from stack exchange:https://stackoverflow.com/questions/50557468/load-all-images-from-a-folder-using-pil
for f in glob.iglob("C:/Users/vwilso17/Documents/Grad School/COSC525Deeplearning/code/projects/project3/val/*"):
    val_images.append(np.asarray(pl.Image.open(f)))
val_images = np.array(val_images)

#train_dataset=tf.data.Dataset.from_tensor_slices((train_images, train_labels))#not using currently
#val_dataset=tf.data.Dataset.from_tensor_slices((val_images, val_labels))#not using currently

train_images=train_images.reshape(86744,1024)
train_images=train_images.astype('float32')
val_images=val_images.reshape(10954,1024)
val_images=val_images.astype('float32')


## Task 1, FCNN
class networks:
    def __init__(self):
        print('init')#never prints

    def FCNN(self,lr=0.01,lrd=0,mom=0.0,tasknum=1,xtrain=train_images,ytrain=train_labels,xval=val_images,yval=val_labels,categories=labels):#lr=learning rate, lrd=learning rate decay, mom=momentum,tasknum=what to classify
        self.lr=lr
        self.lrd=lrd
        self.mom=mom
        self.tasknum=tasknum
        self.xtrain=xtrain
        self.ytrain=ytrain
        self.xval=xval
        self.yval=yval
        self.categories=labels

        #network set-up 
        model=keras.Sequential()
        model.add(keras.Input(shape=(1024,)))
        model.add(layers.Dense(1024, activation="tanh"))
        model.add(layers.Dense(512, activation="sigmoid"))
        model.add(layers.Dense(100, activation="relu"))
        if self.tasknum==0:#classify based on age
            model.add(layers.Dense(9,activation="softmax"))
            ytrain=ytrain[:,self.tasknum+1]
            encode=labelvalues()
            ytrain=encode.encode(ytrain,categories[0])
            ytrain=ytrain.astype('float32')
        elif self.tasknum==1:#classify based on gender
            model.add(layers.Dense(2,activation="softmax"))
            ytrain=ytrain[:,self.tasknum+1]
            encode=labelvalues()
            ytrain=encode.encode(ytrain,categories[1])
            ytrain=ytrain.astype('float32')
        else:#classify based on race
            model.add(layers.Dense(7,activation="softmax"))
            ytrain=ytrain[:,self.tasknum+1]
            encode=labelvalues()
            ytrain=encode.encode(ytrain,categories[2])
            ytrain=ytrain.astype('float32')
        model.summary()
    
        model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=self.lr,momentum=self.mom),#default lr=0.01, default mom=0.0
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalCrossentropy()],
        )

        #training

        model.fit(xtrain,ytrain,epochs=3)
        #validation/classification

    #plotting/final results summary
networktest=networks()
FCNNtest=networktest.FCNN()
print('test')


## Task 2, CNN

## Task 3, homebrew CNN

## Task 4, multitask CNN

## Task 5, Variational Auto-Encoder
