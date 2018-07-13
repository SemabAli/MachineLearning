
# coding: utf-8

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Activation,Dropout
from keras import utils
from keras.callbacks import TensorBoard
from keras.preprocessing.image import  ImageDataGenerator

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""



#Activation vs activations

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))


model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])




test_datagen=ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)



train_generator = train_datagen.flow_from_directory(
        './TrainF',                      # this is the target directory
        target_size=(150, 150),            # all images will be resized to 150x150
        batch_size=5,
        class_mode='categorical')




validation_geneator=test_datagen.flow_from_directory("./ValF",
                                                     target_size=(150,150),
                                                     batch_size=5,
                                                     class_mode='categorical')



model.fit_generator(
        train_generator,
        steps_per_epoch=500,    #batch_size,
        epochs=2,
        validation_data=validation_geneator,
        validation_steps=10 )    #batch_size)

model.save_weights('my_model_weights.h5')


# In[1]:


print "hello"


# In[ ]:




