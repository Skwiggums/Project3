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
import matplotlib.pyplot as plt

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

#pulling in image data
train_images = [] #got this from stack exchange:https://stackoverflow.com/questions/50557468/load-all-images-from-a-folder-using-pil
for f in glob.iglob("C:/Users/vwilso17/Documents/Grad School/COSC525Deeplearning/code/projects/project3/train/*"):
    train_images.append(np.asarray(pl.Image.open(f)))
train_images = np.array(train_images)
train_extremes=[min(train_images.reshape(86744*1024)),max(train_images.reshape(86744*1024))]#min-max scaling
train_images=(train_images-train_extremes[1])/(train_extremes[1]-train_extremes[0])

val_images = [] #got this from stack exchange:https://stackoverflow.com/questions/50557468/load-all-images-from-a-folder-using-pil
for f in glob.iglob("C:/Users/vwilso17/Documents/Grad School/COSC525Deeplearning/code/projects/project3/val/*"):
    val_images.append(np.asarray(pl.Image.open(f)))
val_images = np.array(val_images)
val_extremes=[min(val_images.reshape(10954*1024)),max(val_images.reshape(10954*1024))]#min-max scaling
val_images=(val_images-val_extremes[1])/(val_extremes[1]-val_extremes[0])

#reshaping image data by flattening 32x32 image data to 1x1024. Converting to float for keras
train_images=train_images.reshape(86744,1024)
train_images=train_images.astype('float32')
val_images=val_images.reshape(10954,1024)
val_images=val_images.astype('float32')


## Task 1, FCNN
class networks:
    def __init__(self,tasknum=0,xtrain=train_images,ytrain=train_labels,xval=val_images,yval=val_labels,categories=labels):
        print('init')#never prints
        self.tasknum=tasknum
        self.xtrain=xtrain
        self.ytrain=ytrain
        self.xval=xval
        self.yval=yval
        self.categories=labels

        #converting strings to numeric so Keras can use labels
        self.ytrain=self.ytrain[:,self.tasknum+1]
        self.yval=self.yval[:,self.tasknum+1]
        encode=labelvalues()
        self.ytrain=encode.encode(self.ytrain,categories[self.tasknum])
        self.ytrain=self.ytrain.astype('float32')
        self.yval=encode.encode(self.yval,categories[self.tasknum])
        self.yval=self.yval.astype('float32')

        #creating test data from training data
        self.xtest=self.xtrain[-100:]#since we're grabbing test data from pre-normalized training data, test data is also normalized
        self.ytest=self.ytrain[-100:]
        self.xtrain=self.xtrain[:-100]
        self.ytrain=self.ytrain[:-100]
    
    def fcnn(self,lr=0.5,lrd=0,mom=0.0):#lr=learning rate, lrd=learning rate decay, mom=momentum,tasknum=what to classify
        self.lr=lr
        self.lrd=lrd
        self.mom=mom

        #network set-up 
        model=keras.Sequential()
        model.add(keras.Input(shape=(1024,)))
        model.add(layers.Dense(1024, activation="tanh"))
        model.add(layers.Dense(512, activation="sigmoid"))
        model.add(layers.Dense(100, activation="relu"))
        if self.tasknum==0:#classify based on age
            model.add(layers.Dense(9,activation="softmax"))
        elif self.tasknum==1:#classify based on gender
            model.add(layers.Dense(2,activation="softmax"))
        else:#classify based on race
            model.add(layers.Dense(7,activation="softmax"))
        model.summary()
    
        model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=self.lr,momentum=self.mom),#default lr=0.01, default mom=0.0
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
        
        #training and validating
        print('training and validating')
        history=model.fit(self.xtrain,self.ytrain,epochs=5,validation_data=(self.xval,self.yval))#if you don't specify a batch size, it uses 32 for mini-batch GD
        #testing
        print('testing')
        results=model.evaluate(self.xtest,self.ytest)
        print('predicting')
        #predicting
        predictions = model.predict(self.xtest[:3])
        
        #plotting/final results summary, from https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
        plt.plot(history.history['sparse_categorical_accuracy'])
        plt.plot(history.history['val_sparse_categorical_accuracy'])
        if self.tasknum==0:#classify based on age
            plt.title('model accuracy for age classification')
        elif self.tasknum==1:#classify based on gender
            plt.title('model accuracy for gender classification')
        else:#classify based on race
            plt.title('model accuracy for race classification')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        if self.tasknum==0:#classify based on age
            plt.title('model loss for age classification')
        elif self.tasknum==1:#classify based on gender
            plt.title('model loss for gender classification')
        else:#classify based on race
            plt.title('model loss for race classification')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        trainingloss=history.history['loss']
        validationloss=history.history['val_loss']

        print('final accuracy is %f' %results[1])

        predictions = np.argmax(predictions, axis=1)
        

        return [trainingloss,validationloss,history,results,confmatrix]


    def cnn(self):
        print('init')
    def cnnhomebrew(self):
        print('init')
    def cnntwo(self):
        print('init')
    def vae(self):
        print('init')

NETWORKS=networks()

# Task 1, FCNN
FCNN=NETWORKS.fcnn()
## Task 2, CNN
CNN=NETWORKS.cnn()
## Task 3, homebrew CNN
CNNHOMEBREW=NETWORKS.cnnhomebrew()
## Task 4, multitask CNN
CNNTWO=NETWORKS.cnntwo()
## Task 5, Variational Auto-Encoder
VAE=NETWORKS.vae()
