
# coding: utf-8

# In[23]:


from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,MaxPool2D,Flatten
import keras.utils
from keras.datasets import mnist
from keras import activations
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""



(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)


#sns.distplot(y_test)

sns.distplot(y_train)
plt.show()


# In[24]:


model=Sequential()
model.add(Conv2D(32,(2,2),input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D())


model=Sequential()
model.add(Conv2D(64,(2,2),input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D())


model.add(Flatten())

y_train_labels=keras.utils.to_categorical(y_train,num_classes=10)
y_test_labels=keras.utils.to_categorical(y_test,num_classes=10)

model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train_labels,batch_size=60,nb_epoch=5,validation_data=(x_test,y_test_labels))



# In[3]:


# test=x_test[0].reshape(1,28,28,1)
# print model.predict(test)



# plt.imshow(x_test[0])
# plt.show()


# In[ ]:




