#%%
from keras.models import Sequential

from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
import os
import tensorflow as tf
#%%
# Visualise using matplotlib

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


img = mpimg.imread("Data/Train/ISIC_0024306.jpg")
print(img)
imgplot = plt.imshow(img)

plt.colorbar()
imgplot = plt.imshow(img)
lum_img = img[:, :, 0]
plt.imshow(lum_img)

#%%

df=pd.read_csv('ISIC2018_Task3_Training_GroundTruth.csv', dtype=str)
df['image'] = df['image']+'.jpg'

IMG_HEIGHT,IMG_WIDTH = 224,224



datagen=ImageDataGenerator(rescale=1./255,validation_split=0.2,
                           featurewise_center=True,featurewise_std_normalization=True,
                           shear_range=0.2,zoom_range=0.2,
                           rotation_range=10,width_shift_range=0.1,
                           height_shift_range=0.1)

train_generator=datagen.flow_from_dataframe(dataframe=df, directory="Data/Train", 
                                            x_col="image", y_col=['MEL','NV','BCC','AKIEC','BKL','DF','VASC'], 
                                            class_mode="raw", 
                                            target_size=(IMG_HEIGHT,IMG_WIDTH), batch_size=100,
                                           subset='training')

valid_generator=datagen.flow_from_dataframe(dataframe=df, directory="Data/Train", 
                                            x_col="image", y_col=['MEL','NV','BCC','AKIEC','BKL','DF','VASC'], 
                                            class_mode="raw", 
                                            target_size=(IMG_HEIGHT,IMG_WIDTH), batch_size=100,
                                           subset='validation')
#%%

input_shape = (224, 224, 3)
num_classes = 7


#%%
# CNN architecture

model=Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu', padding='same',input_shape=input_shape))
model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,kernel_size=(3, 3), activation='relu',padding = 'Same'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64,kernel_size=(3, 3), activation='relu',padding = 'Same'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

#%%
model.summary()

#%%
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#%%

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

#stopper = EarlyStopping(monitor='val_loss', mode='min', patience=4)

#tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #log_dir='results/tf_logs', profile_batch=5)

#%%
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size


hist=model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=50)

#%%


model.save('cancer1.h5')
	                       
#%%	  
#Curve                       

accuracy = hist.history['acc']
val_accuracy = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#%%



#%%

