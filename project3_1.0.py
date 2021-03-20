import sys
import random
import numpy as np
import pandas as pd
import PIL as pl#pillow's module name isn't pillow, it's PIL, took me an hour to figure that out
import glob
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers

##pulling in test and training data
val_labels=pd.read_csv("fairface_label_val.csv")
train_labels=pd.read_csv("fairface_label_train.csv")

train_images = [] #got this from stack exchange:https://stackoverflow.com/questions/50557468/load-all-images-from-a-folder-using-pil
for f in glob.iglob("C:/Users/vwilso17/Documents/Grad School/COSC525Deeplearning/code/projects/project3/train/*"):
    train_images.append(np.asarray(pl.Image.open(f)))
train_images = np.array(train_images)

val_images = [] #got this from stack exchange:https://stackoverflow.com/questions/50557468/load-all-images-from-a-folder-using-pil
for f in glob.iglob("C:/Users/vwilso17/Documents/Grad School/COSC525Deeplearning/code/projects/project3/val/*"):
    val_images.append(np.asarray(pl.Image.open(f)))
val_images = np.array(val_images)

## Task 1, FCNN
#def FCNN(lr,lrd,mom):#lr=learning rate, lrd=learning rate decay, mom=momentum
    #self.lr=lr
    #self.lrd=lrd
    #self.mom=mom

model=keras.Sequential()
model.add(keras.Input(shape=(32,)))
model.add(layers.Dense(1024, activation="tanh"))
model.add(layers.Dense(512, activation="sigmoid"))
model.add(layers.Dense(100, activation="relu"))
model.summary()
## Task 2, CNN

## Task 3, homebrew CNN

## Task 4, multitask CNN

## Task 5, Variational Auto-Encoder



