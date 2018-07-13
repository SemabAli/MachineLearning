
# coding: utf-8

# In[24]:


# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Activation,Dropout
from keras import utils
from keras.callbacks import TensorBoard
from keras.preprocessing.image import  ImageDataGenerator

from keras.applications.resnet50 import preprocess_input
import keras



width=200
height=200


pre_model = keras.applications.ResNet50(include_top=False,weights='imagenet',input_shape=(width,height,3))
print "done"


# In[6]:


#Activation vs activations


# In[25]:



model=Sequential()
model.add(pre_model)

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(3))    # 3 classes
model.add(Activation('softmax'))


# model = Model(inputs=pre_model.input, outputs=top_model(prev_model.output))



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
        target_size=(width, height),            # all images will be resized to 150x150
       , batch_size=5
        class_mode='categorical')




validation_geneator=test_datagen.flow_from_directory("./ValF",
                                                     target_size=(width,height),
                                                     batch_size=5,
                                                     class_mode='categorical')



model.fit_generator(
        train_generator,
        steps_per_epoch=500,    #batch_size,
        epochs=2,
        validation_data=validation_geneator,
        validation_steps=10 )    #batch_size)

model.save_weights('my_model_weights.h5')


# In[ ]:




