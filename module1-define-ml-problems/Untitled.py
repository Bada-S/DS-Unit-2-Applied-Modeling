#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install matplotlib


# In[2]:


import pandas as pd
import numpy as np
import matplotlib as plt


# In[3]:


df = pd.read_csv(r"C:\Users\james\AppData\Local\Temp\Temp1_56584_396578_bundle_archive.zip\bad-drivers.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


pip install pandas_profiling


# In[7]:


import pandas_profiling
df.profile_report()


# In[8]:


#No missing values, no zeros-- really no wrangling to do


# In[9]:


from sklearn.model_selection import train_test_split
train, val = train_test_split(df, train_size=.8, test_size=.2, random_state=42)
print(train.shape, val.shape, df.shape)
assert train.shape[0] + val.shape[0] == df.shape[0]


# In[10]:


target = 'Number of drivers involved in fatal collisions per billion miles'

X_train, y_train = train.drop(target, axis=1), train[target]
X_val, y_val = val.drop(target, axis=1), val[target]


# In[11]:


#Establish baseline accuracy
from sklearn.metrics import mean_absolute_error
print('Training MAE:', mean_absolute_error(y_train, [y_train.mean()]*len(y_train)))
print('Validation MAE:', mean_absolute_error(y_val, [y_val.mean()]*len(y_val)))


# In[12]:


from sklearn.impute import SimpleImputer
from category_encoders import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

model_rf = make_pipeline(
            OrdinalEncoder(),
            RandomForestRegressor()
)
model_rf.fit(X_train,y_train)


# In[13]:


print('Training accuracy:', model_rf.score(X_train,y_train))
print('validation accuracy:', model_rf.score(X_val,y_val))


# In[14]:


df2 = pd.read_csv(r'C:\Users\james\AppData\Local\Temp\Temp1_199387_1319582_bundle_archive.zip\US_Accidents_June20.csv', parse_dates=True, infer_datetime_format=True)


# In[15]:


df2.head()


# In[16]:


df2.drop('ID', axis=1)


# In[18]:


df2.corr()


# In[21]:


df2.isnull().sum()


# In[26]:


def wrangle(X):
    #convert booleans to int
    x['Amenity','Bump', 'Crossing', 'Give_way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_calming',
     'Traffic_signal', 'Turning_Loop'].astype(int, inplace=True)


# In[17]:


df2.info()


# In[ ]:




