import sys
import random
import numpy as np
import pandas as pd
import PIL as pl#pillow's module name isn't pillow, it's PIL, took me an hour to figure that out
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers

##pulling in test and training data
val_labels=pd.read_csv("fairface_label_val.csv")
train_labels=pd.read_csv("fairface_label_train.csv")

## Task 1, FCNN
FCNN=keras.Sequential()
FCNN.add(keras.Input(shape=(32, 32, 1)))
FCNN.add(layers.Dense(1024, activation="tanh"))
FCNN.add(layers.Dense(512, activation="sigmoid"))
FCNN.add(layers.Dense(100, activation="relu"))

## Task 2, CNN

## Task 3, homebrew CNN

## Task 4, multitask CNN

## Task 5, Variational Auto-Encoder

