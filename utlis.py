from importlib.resources import files

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from keras import Sequential
from keras.src.layers import Flatten, Dense, BatchNormalization, Conv2D, Dropout
from keras.src.optimizers import Adam
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random


def getName(filePath):
    return filePath.split("\\")[-1]

def importDataInfo(path):
    columns=['Center','Left','Right','Steering','Throttle','Break','Speed']# columns of our data
    # 'Center','Left','Right' as input and
    # 'Steering' as output
    data=pd.read_csv(os.path.join(path,'driving_log.csv'),names=columns)# we can join at the end of directory 'driving_log.csv'
    #print(data.head())
    #print(data['Center'][0])
    #print(getName(data['Center'][0]))
    data['Center']=data['Center'].apply(getName)
    #print(data.shape[0])#row no needed 11383 data points
    return data


def balanceData(data, display=True):
    nBins = 31
    samplesPerBin = 1000
    hist, bins = np.histogram(data['Steering'], nBins)

    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.05)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.show()

    removeIndexList = []
    for i in range(nBins):
        binDataList = np.where((data['Steering'] >= bins[i]) & (data['Steering'] <= bins[i + 1]))[0]
        binDataList = shuffle(binDataList)
        if len(binDataList) > samplesPerBin:
            removeIndexList.extend(binDataList[samplesPerBin:])

    print(f'Removed Images: {len(removeIndexList)}')
    data.drop(data.index[removeIndexList], inplace=True)
    print(f'Remaining Images: {len(data)}')

    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        plt.bar(center, hist, width=0.05)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.show()


def loadData(path,data):
    imagePath=[]
    steering=[]
    for i in range(len(data)):
        indexedData=data.iloc[i]
        #print(indexedData)
        imagePath.append(os.path.join(path,'IMG',indexedData[0]))
        #print(imagePath)
        steering.append(float(indexedData[3]))
        #print(steering)
    imagePath=np.asarray(imagePath)
    steering=np.asarray(steering)
    return imagePath,steering


def augmentImage(imgPath, steering):
    img = mpimg.imread(imgPath)

    ### PAN ###
    if random.uniform(0, 1) < 0.5:
        pan = iaa.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)})
        img = pan.augment_image(img)

    ### ZOOM ###
    if random.uniform(0, 1) < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)

    ### BRIGHTNESS ###
    if random.uniform(0, 1) < 0.5:
        brightness = iaa.Multiply((0.4, 1.3))
        img = brightness.augment_image(img)

    ### FLIP ###
    if random.uniform(0, 1) < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering

    return img, steering


#imgRe, st=augmentIMage(',0)
#plt.imshow(imgRe)
#plt.show()
def preProcessing(img):
    img=img[60:135,:,:] # crop img for road only
    img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)# WE DO color change to see the borderline
    img=cv2.GaussianBlur(img,(3,3),0)
    img=cv2.resize(img,(200,66)) #nvdia did that
    img=img/255 #normalization
    return img
#imgRe=preProcessing(mpimg.imread(''))
#plt.imshow(imgRe)
#plt.show()

def batchGen(imagePath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
        for i in range(batchSize):
            index = random.randint(0, len(imagePath) - 1)
            if trainFlag:
                img, steering = augmentImage(imagePath[index], steeringList[index])
            else:
                img = mpimg.imread(imagePath[index])
                steering = steeringList[index]
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch), np.asarray(steeringBatch))


def createModel():
    model = Sequential()

    model.add(Conv2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), (1, 1), activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), (1, 1), activation='elu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(100, activation='elu', kernel_regularizer='l2'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu', kernel_regularizer='l2'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu', kernel_regularizer='l2'))
    model.add(Dense(1))  # Output layer

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

    return model
