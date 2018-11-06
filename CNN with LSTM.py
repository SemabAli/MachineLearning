#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from keras.utils import OrderedEnqueuer
pd.set_option('display.max_columns', 100)
from sklearn.cross_validation import train_test_split


# In[7]:


data=pd.read_csv("cleaned_data.csv",index_col=0,names=["tweet","label"],skiprows=1)
data.head()
print data.label.unique()


# In[8]:


data=data.dropna()
tweets=data["tweet"].tolist()
labels=data["label"].tolist()


# In[9]:


stop_words = stopwords.words('english')
parsed=[]
try:
    for sen in tweets:
         parsed.append([txt for txt in word_tokenize(sen.lower()) if txt not in stop_words])
except:
    print sen


# In[10]:


print parsed[2]


# In[11]:


model=Word2Vec(parsed,size=50,min_count=1)
model.train(parsed,total_examples=len(data),epochs=10)


# In[12]:


weights=model.wv.syn0

vocab_size,embedding_size=weights.shape
print("vocab is ",vocab_size)
print("embedding size ",embedding_size)


# In[13]:


from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,LSTM,Embedding,Activation,Conv1D,Dropout
from keras.activations import sigmoid,softmax
from keras.preprocessing.sequence import pad_sequences
import seaborn as sns


# In[14]:



vocab={}
counter=1
max=0
for txt in parsed:
    for word in txt:
        if word in vocab.keys():
            pass
        else:
            vocab[word]=counter
            counter+=1   


# In[15]:


train_x=[]

for txt in parsed:
    temp=[]
    for word in txt:
        temp.append(vocab[word])
    train_x.append(temp)    


# In[16]:


sns.distplot(labels)


# In[17]:


train_x=np.array([np.array(xi) for xi in train_x])

train_x=pad_sequences(train_x,maxlen=29,padding='post')
train_y=to_categorical(labels)



# In[18]:


from keras import backend as K
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[5]:


model_k=Sequential()
model_k.add(Embedding(input_dim=29,output_dim=40))

model_k.add(Conv1D(32,kernel_size=(3)))
model_k.add(Activation('relu'))

model_k.add(LSTM(400,return_sequences=True))
model_k.add(LSTM(200))

model_k.add(Dense(32))
model_k.add(Activation('relu'))
model_k.add(Dropout(0.2))

model_k.add(Dense(3))
model_k.add(Activation('softmax'))

model_k.compile(loss='categorical_crossentropy',metrics=[f1],optimizer="adam")
model_k.summary()
model_k.fit(train_x,train_y,epochs=20,batch_size=32)


# In[145]:


sns.distplot(labels,rug=True, kde=True, norm_hist=False)
temp=(data[data["label"]==2])
print temp.count()


# In[21]:


def wordtoinx(word):
    return model.wv.vocab[word].index
train_xx=[]
for sen in parsed:
    temp=[]
    for i,word in enumerate(sen):
        temp.append(wordtoinx(word))
    train_xx.append(temp)   
        
train_xx=pad_sequences(train_xx,maxlen=29,padding='post')



# In[25]:


model_k=Sequential()
model_k.add(Embedding(input_dim=18590,output_dim=50,weights=[my_weights],trainable=False))

model_k.add(Conv1D(32,kernel_size=(3)))
model_k.add(Activation('relu'))

model_k.add(LSTM(400,return_sequences=True))
model_k.add(LSTM(200))

model_k.add(Dense(32))
model_k.add(Activation('relu'))
model_k.add(Dropout(0.2))

model_k.add(Dense(3))
model_k.add(Activation('softmax'))

model_k.compile(loss='categorical_crossentropy',metrics=[f1],optimizer="adam")
model_k.summary()
model_k.fit(Train_X,Train_Y,epochs=50,batch_size=32,validation_data=(Test_X,Test_Y))


# In[22]:


my_weights=model.wv.syn0


# In[23]:


Train_X,Test_X,Train_Y,Test_Y=train_test_split(train_xx,train_y,test_size=0.2)


# In[77]:


tst=""

tst=word_tokenize(tst)
tst=[word for word in tst if word not in stop_words]
i=0
temp=np.zeros((1,29))
print tst
for t in tst:
        print t
        temp[0,i]=wordtoinx(t)
        i=i+1

pred= model_k.predict_classes(temp)
print pred


# In[ ]:




