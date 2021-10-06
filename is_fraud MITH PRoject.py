#!/usr/bin/env python
# coding: utf-8

# # Problem Statement: 
# 
# Detecting fraud for transactions in a payment gateway
# 
# A new disruptive payment gateway start-up, ‘IndAvenue’, has started gaining traction due to its extremely low processing fees for handling online vendors’ digital payments. This strategy has led to very low costs of acquiring new vendors.
# 
# Unfortunately, due to the cheap processing fees, the company was not able to build and deploy a robust and fast fraud detection system. Consequently, a lot of the vendors have accumulated significant economic burden due to handling fraudulent transactions on their platforms. This has resulted in a significant number of current clients leaving IndAvenue’s payment gateway platform for more expensive yet reliable payment gateway companies. 
# 
# The company’s data engineers curated a dataset that they believe follows the real world distribution of transactions on their payment gateway. The company hired Insofe and provided it with the dataset, to create a fast and robust AI based model that can detect and prevent fraudulent transactions on its payment gateway.
# 
# They have provided you with the dataset that has the `is_fraud` column, which encodes the information whether a transaction was fraudulent or not.
# 
#  
# 
# In this hackathon, you will now have to use this curated data to create a machine learning model that will be able to predict the `is_fraud` column.

# In[1]:


#import all the required packages and libraries
import numpy as np
import pandas as pd

import datetime

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit

from xgboost import XGBClassifier

# !pip install imblearn
from imblearn.over_sampling import SMOTE

from IPython.display import Image

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#read the train and test data
traindt=pd.read_csv("train_data-1599717478076.csv")
testdt=pd.read_csv("test_data-1599717650029.csv")


# In[3]:


traindt.head()            #view of few observations of training


# In[4]:


testdt.head()           #view of few observations of testing


# In[5]:


traindt.shape       #shape of training data


# In[6]:


testdt.shape       #shape of testing data


# In[7]:


traindt.columns   #columns in training


# In[8]:


testdt.columns         #columns in testing


# In[9]:


traindt.dtypes         #datatypes of attributes


# In[10]:


testdt.dtypes     #datatypes of attributes


# In[11]:


traindt.nunique()       #checking unique  values


# In[12]:


testdt.nunique()            #checking unique  values


# droping user_id and country

# In[13]:


traindt=traindt.drop(["user_id","country"],axis=1)      


# In[14]:


testdt=testdt.drop(["user_id","country"],axis=1)


# In[15]:


traindt.shape


# In[16]:


testdt.shape


# In[17]:


traindt.describe(include='all')       #description of train data


# In[18]:


testdt.describe(include='all')         #description of test data


# In[19]:


#converting the variables to categorical in training and testing
cat_cols=['payment_method', 'partner_id',
       'partner_category','device_type','partner_pricing_category','is_fraud']


# In[20]:


traindt[cat_cols]=traindt[cat_cols].astype('category')


# In[21]:


cat_cols1=['payment_method', 'partner_id',
       'partner_category','device_type','partner_pricing_category']


# In[22]:


testdt[cat_cols1]=testdt[cat_cols1].astype('category')


# In[23]:


traindt.dtypes     #datatypes of variables


# In[24]:


testdt.dtypes         #datatypes of variables


# Here we are dividing transaction_initiation to hour,date,month,weekday for both test and traing datasets

# In[25]:


traindt['transaction_DateTime'] = pd.to_datetime(traindt['transaction_initiation'])
traindt['transaction_date'] = traindt['transaction_DateTime'].dt.date
traindt['transaction_month']=traindt['transaction_DateTime'].dt.month
traindt['transaction_hour'] = traindt['transaction_DateTime'].dt.hour
traindt['transaction_weekday']=traindt['transaction_DateTime'].dt.weekday
traindt['transaction_hour']=traindt['transaction_DateTime'].dt.hour


# In[26]:


testdt['transaction_DateTime'] = pd.to_datetime(testdt['transaction_initiation'])
testdt['transaction_date'] = testdt['transaction_DateTime'].dt.date
testdt['transaction_month']=testdt['transaction_DateTime'].dt.month
testdt['transaction_hour'] = testdt['transaction_DateTime'].dt.hour
testdt['transaction_weekday']=testdt['transaction_DateTime'].dt.weekday
testdt['transaction_hour']=testdt['transaction_DateTime'].dt.hour


# In[27]:


traindt.head()


# In[28]:


testdt.head()


# In[29]:


traindt.shape


# In[30]:


testdt.shape


# In[31]:


#dropping the unnecessary variables for test and training datasets
traindt=traindt.drop(['transaction_initiation','transaction_DateTime','transaction_date'],axis=1)


# In[32]:


testdt=testdt.drop(['transaction_initiation','transaction_DateTime','transaction_date'],axis=1)


# In[33]:


traindt.shape


# In[34]:


testdt.shape


# Do label encoding for necessary categorical variables

# In[35]:


lb_encoder=preprocessing.LabelEncoder()


# In[36]:


traindt['payment_method']=lb_encoder.fit_transform(traindt['payment_method'])
traindt['partner_id']=lb_encoder.fit_transform(traindt['partner_id'])
traindt['partner_category']=lb_encoder.fit_transform(traindt['partner_category'])
traindt['device_type']=lb_encoder.fit_transform(traindt['device_type'])
traindt['partner_pricing_category']=lb_encoder.fit_transform(traindt['partner_pricing_category'])


# In[37]:


traindt.head()


# In[38]:


traindt.shape


# In[39]:


testdt['payment_method']=lb_encoder.fit_transform(testdt['payment_method'])
testdt['partner_id']=lb_encoder.fit_transform(testdt['partner_id'])
testdt['partner_category']=lb_encoder.fit_transform(testdt['partner_category'])
testdt['device_type']=lb_encoder.fit_transform(testdt['device_type'])
testdt['partner_pricing_category']=lb_encoder.fit_transform(testdt['partner_pricing_category'])


# In[40]:


testdt.shape


# In[69]:


dt=traindt['payment_method']


# In[71]:


traindt['payment_method'].value_counts()


# In[42]:


#checking the counts of 0's and 1 in is_fraud
traindt.is_fraud.value_counts()


# In[43]:


traindt.is_fraud.value_counts(normalize=True)*100


# In[44]:


X = traindt.drop('is_fraud', axis=1)  #splitting xtrain,xtest,y_train,y_test for modeling
y = traindt['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123) 


# In[45]:


print(pd.value_counts(y_train)/y_train.count() * 100)      #check whether they are evenly sampled based on is_fraud

print(pd.value_counts(y_test) /y_test.count() * 100)


# In[46]:


X_train=X_train.drop(['transaction_number'],axis=1)   #drop transaction_number for xtrain and x_test


# In[47]:


X_test=X_test.drop(['transaction_number'],axis=1)


# In[48]:


X_train.shape


# In[49]:


X_test.shape


# In[50]:


import xgboost


# In[51]:


clf_XGB = XGBClassifier()
clf_XGB.fit(X_train, y_train)      #fit the xgboost model


# In[52]:


ytrain_pred=clf_XGB.predict(X_train)         #give xtrain to build model

print(f'training accuracy score:{accuracy_score(y_train,ytrain_pred)}')
print(f'\nconfusion matrix:\n{confusion_matrix(y_train,ytrain_pred)}')
print(f'\n\nClassification Report :\n{classification_report(y_train,ytrain_pred)}')


# In[53]:


ytest_pred=clf_XGB.predict(X_test)          #now x-test 

print(f'training accuracy score:{accuracy_score(y_test,ytest_pred)}')
print(f'\nconfusion matrix:\n{confusion_matrix(y_test,ytest_pred)}')
print(f'\n\nClassification Report :\n{classification_report(y_test,ytest_pred)}')


# In[54]:


testdt.columns


# In[55]:


X_test.head()


# In[56]:


test_transation_number=testdt['transaction_number']  
#assign transaction _number oftest data to a variable inorder to map the predection is_fraud


# In[57]:


test_transation_number


# In[58]:


testdt=testdt.drop(['transaction_number'],axis=1) #now drop the variable


# In[59]:


testdt.head()


# In[60]:


y_pred_testdt=clf_XGB.predict(testdt)   #fit the model with the orginal test dataset


# In[61]:


y_pred_testdt


# In[62]:


datapred=pd.DataFrame(y_pred_testdt,columns=['is_fraud']) #creating new dataframe


# In[63]:


datapred


# In[64]:


sample=pd.concat([test_transation_number,datapred],axis=1)     #add the transaction_number to the new dataset


# In[65]:


sample


# In[66]:


sample['is_fraud'].value_counts()


# In[67]:


sample.to_csv('sample_submission.csv')


# In[68]:


pwd()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




