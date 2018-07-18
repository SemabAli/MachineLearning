
# coding: utf-8

# In[726]:



import pandas as pd
from  sklearn.decomposition import PCA
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import make_scorer

from sklearn.metrics import mean_squared_error
from math import sqrt


# In[727]:


data=pd.read_csv("/../../data/Training_Data_Claims.csv",nrows=5675143,thousands=",")


# In[728]:


data.drop(["gender","submission_type","adj_write_off_type","details","reject_type","reject_amount","eob_code","era_category_code","era_adjustment_code","era_rejection_category_code"],axis=1 ,inplace=True)
print data.columns
data.dropna(inplace=True)

print data.amount_approved


# In[729]:


data.amount_approved=data.amount_approved/data.units

# std=data.amount_approved.std()
# mean=data.amount_approved.mean()
# lower=mean-std
# upper=mean+std
# data=data[(data.amount_approved > lower) & (data.amount_approved < upper)]


P = np.percentile(data.amount_approved, [5, 90])
data = data[(data.amount_approved > P[0]) & (data.amount_approved < P[1])]





# In[730]:


# data2=data.loc[:,["procedure_code"]]   
# data=data.drop(["procedure_code"],axis=1)  
# data3 = data.join(data2)
#patient_account

data.amount_approved=data.amount_approved/data.units
data=data.drop(["paid_proc_code","bill_date","dos","doe","dos_from","dos_to","claim_payments_date_entry","claim_no",
                "amount",
               "payment_no",u'amount_paid',"claim_payments_Units","units", "pri_sec_oth_type", "pos",
               "payment_type","inspayer_id", "insname_id","payment_source","insgroup_id"],axis=1)


data['procedure_code'] = le.fit_transform(data['procedure_code'])
data['inspayer_state'] = le.fit_transform(data['inspayer_state'])

print data.columns

print data.dtypes


# In[731]:




correlation = data.corr(method='pearson')
print correlation


# In[732]:


label=data.loc[:,"amount_approved"]
data.drop(["amount_approved"],axis=1,inplace=True)

data.isnull().sum()

X_train, X_test, y_train, y_test = train_test_split(
   data, label, test_size=0.30, random_state=42)






# In[ ]:



params1={'n_estimators':100, 'criterion':'mse', 'max_depth':5, 'min_samples_split':2, 
         'min_samples_leaf':5,
         'max_features':'auto', 'n_jobs':5}
         
         
gb1=RandomForestRegressor()
gb1.fit(X_train,y_train)

results = pd.DataFrame()
results["actual"]=y_test
results["predictions"] = gb1.predict(X_test)
results["dif"] = (results["actual"] - results["predictions"]).abs()
print (results['dif'].mean())



# gbr_scores = cross_val_score(gb, X_train, y_train, cv=3)
# print gbr_scores  


# In[ ]:



params = {'alpha' :0.9 ,'n_estimators': 400, 'max_depth': 4, 'min_samples_split': 3,
          'learning_rate': 0.1, 'loss': 'ls'}

gb=GradientBoostingRegressor(**params)
gb.fit(X_train,y_train)


# In[ ]:


y_pred=gb.predict(X_test)

rms = sqrt(mean_squared_error(y_test, y_pred))



# In[ ]:


results = pd.DataFrame()
results["actual"]=y_test
results["predictions"] = gb.predict(X_test)
results["dif"] = (results["actual"] - results["predictions"]).abs()
print (results['dif'].mean())


# In[ ]:


for key in gb.feature_importances_:
    print key
print X_train.columns    


# In[ ]:


myarray1=np.arange(0,10)
myarray=np.array([1,5,10,100,5])

print myarray.std()
print myarray1.std()


# In[ ]:




