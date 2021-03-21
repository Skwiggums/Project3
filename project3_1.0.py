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
train_labels=train_labels[:,0:3]
val_labels=val_labels[:,0:3]#removing service test category since tensorflow doesn't like bools

train_images = [] #got this from stack exchange:https://stackoverflow.com/questions/50557468/load-all-images-from-a-folder-using-pil
for f in glob.iglob("C:/Users/vwilso17/Documents/Grad School/COSC525Deeplearning/code/projects/project3/train/*"):
    train_images.append(np.asarray(pl.Image.open(f)))
train_images = np.array(train_images)

val_images = [] #got this from stack exchange:https://stackoverflow.com/questions/50557468/load-all-images-from-a-folder-using-pil
for f in glob.iglob("C:/Users/vwilso17/Documents/Grad School/COSC525Deeplearning/code/projects/project3/val/*"):
    val_images.append(np.asarray(pl.Image.open(f)))
val_images = np.array(val_images)

train_dataset=tf.data.Dataset.from_tensor_slices((train_images, train_labels))
val_dataset=tf.data.Dataset.from_tensor_slices((val_images, val_labels))

## Task 1, FCNN
class networks:
    def __init__(self):
        print('init')#never prints

    def FCNN(self,lr=0.01,lrd=0,mom=0.0,tasknum=0):#lr=learning rate, lrd=learning rate decay, mom=momentum,tasknum=what to classify
        self.lr=lr
        self.lrd=lrd
        self.mom=mom
        self.tasknum=tasknum
    
        #network set-up 
        model=keras.Sequential()
        model.add(keras.Input(shape=(32,)))
        model.add(layers.Dense(1024, activation="tanh"))
        model.add(layers.Dense(512, activation="sigmoid"))
        model.add(layers.Dense(100, activation="relu"))
        if self.tasknum==0:#classify based on gender
            model.add(layers.Dense(2,activation="softmax"))
        elif self.tasknum==1:#classify based on race
            model.add(layers.Dense(7,activation="softmax"))
        else:#classify based on age
            model.add(layers.Dense(9,activation="softmax"))
        model.summary()
    
        model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=self.lr,momentum=self.mom),#default lr=0.01, default mom=0.0
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalCrossentropy()],
        )

        #training
        model.fit(train_dataset,epochs=3)
        #validation/classification

    #plotting/final results summary
networktest=networks()
FCNNtest=networktest.FCNN()
print('test')
