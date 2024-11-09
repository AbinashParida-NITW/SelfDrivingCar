import matplotlib.pyplot as plt
from keras.src.callbacks import EarlyStopping

from utlis import *
from sklearn.model_selection import train_test_split

########
#STEP 1 (IMPORT DATA )
path= '../myData'
data=importDataInfo(path)
########

######## STEP 2 (DATA Visualization and Normalization)
balanceData(data,display=True)
#############STEP 3 (convert pandas data to numpy array.)
imagePath, steering=loadData(path,data)
#print(imagePath[0],steering[0],sep='\n')
###################
#STEP 4 (SPLIT DATA)
xTrain,xVal,yTrain,yVal=train_test_split(imagePath,steering,test_size=0.2,random_state=42)
print(f'Total Training Images:{len(xTrain)}\nTotal Testing Images:{len(xVal)}')
####### STEP 5( DATA AUGMENTATION)
#### Step 6 (Data preprocessing)
#####step 7 (Model creation)
model=createModel()
model.summary()
#### step 8 (Training)
# Set up EarlyStopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model using the batch generator
history = model.fit(
    batchGen(xTrain, yTrain, batchSize=100, trainFlag=True),
    steps_per_epoch=200,  # Define steps per epoch based on your dataset
    epochs=100,  # You can set more epochs, but EarlyStopping will stop early if no improvement
    validation_data=batchGen(xVal, yVal, batchSize=100, trainFlag=False),
    validation_steps=50,  # Define validation steps
    callbacks=[early_stop],  # Adding EarlyStopping callback here
    verbose=1
)#xTrain=image of road,
 #(yTrain = steering angle as output). so our model predict the steering angle according to the input xTrain image to move.
###### Step 9 (save and Plotting)
model.save('model1.h5')
print("Model Saved")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()