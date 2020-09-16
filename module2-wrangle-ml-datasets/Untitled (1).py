#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib as plt


# In[19]:


df = pd.read_csv(r'C:\Users\james\AppData\Local\Temp\Temp1_199387_1319582_bundle_archive.zip\US_Accidents_June20.csv')

#Putting Times in datetime format
df['Start_Time']= pd.to_datetime(df['Start_Time'])
df['End_Time']= pd.to_datetime(df['End_Time'])
df['year'] = df['Start_Time'].dt.year
df['month'] = df['Start_Time'].dt.month
df['day'] = df['Start_Time'].dt.day
df['hour'] = df['Start_Time'].dt.hour

df['traffic_disruption(min)'] = round((df['End_Time'] - df['Start_Time'])/np.timedelta64(1,'m'))
df.drop(['Start_Time', 'End_Time'], axis=1, inplace=True)


# In[4]:


df.head()


# In[5]:


df.corr()


# In[6]:


df.isnull().sum()


# In[7]:


df.nunique()


# In[8]:


unique_cats = df.select_dtypes('object').nunique()
unique_cats


# In[9]:


#High cardinality categories
high_card = [col for col in unique_cats.index if unique_cats[col]>127]
high_card


# In[10]:


def wrangle(X):
    #Drop high cardinality columns
    X.drop(high_card, axis=1, inplace=True)
    
    #Drop more columns- country and turning loop only have 1 unique value
    X.drop(['Turning_Loop', 'Country'], axis=1, inplace=True)
    
    #Set negative time values to NaN
    df[df['traffic_disruption(min)']<=0] = np.nan
    
    return X


# In[11]:


df = wrangle(df)


# In[12]:


df.info()


# In[20]:


df.head()


# In[14]:


df.info()


# In[21]:


target = 'Severity'

y = df[target]
X = df.drop(target, axis=1)


# In[22]:


y.value_counts(normalize=True)


# In[23]:


#Baseline accuracy
baseline_acc = y.value_counts(normalize=True).max()
print('Baseline accuracy:',baseline_acc)


# In[25]:


from sklearn.model_selection import train_test_split
train, val = train_test_split(df, test_size=.2, random_state=42)
X_train, y_train = train.drop(target, axis=1), train[target]
X_val, y_val = val.drop(target, axis=1), val[target]


# In[26]:


print(train.shape, val.shape, df.shape)


# Building gradientboost

# In[27]:


from sklearn.impute import SimpleImputer
from category_encoders import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier


# In[28]:


model = make_pipeline(
        OrdinalEncoder(),
        SimpleImputer(),
        LogisticRegression
)


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


print("Training accuracy:" model2.score(X_train,y_train))
print("Validation accuracy:" model2.score(X_val,y_val))


# In[ ]:


feature_imp = model.named_steps['xgbclassifier'].feature_importances_
features= X_train.columns


# In[ ]:


pd.series(feature_imp, features).sort_values(ascending=False).head(5).plot(kind='barh')


# In[ ]:


1+1


# # Drop-Column Importance

# In[ ]:




