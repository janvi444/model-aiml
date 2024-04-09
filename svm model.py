#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing all required libraries

import pandas as pd


# In[2]:


df=pd.read_csv("Titanic-Dataset.csv",)
df.head()


# In[3]:


df.info()


# In[4]:


df=df.drop(["PassengerId","Name","Cabin","Ticket","Embarked","Fare"],axis=1)


# In[5]:


y=df["Survived"]
y


# In[6]:


X=df.drop("Survived",axis=1)


# In[7]:


X


# In[8]:


list(df["Age"].value_counts())


# In[9]:


X["Age"]=X["Age"].fillna(X["Age"].mode())


# In[10]:


X.info()


# In[11]:


X["Age"]


# In[12]:


X["Age"]=X["Age"].fillna(X["Age"].mode()[0])


# In[13]:


X["Age"]   # Now we can see nan values we changed with mode values of ages


# In[14]:


X=pd.get_dummies(X,columns=["Sex"],drop_first=True)
X


# In[15]:


type(X)


# In[16]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=30,random_state=43,stratify=y)


# In[17]:


X_train


# In[18]:


from sklearn.preprocessing import MinMaxScaler


# In[19]:


scaleing=MinMaxScaler()


# In[20]:


X_train=scaleing.fit_transform(X_train)


# In[21]:


X_test=scaleing.transform(X_test)


# In[22]:


from sklearn.svm import SVC


# In[23]:


model=SVC(kernel="poly",degree=5)  #kernel="poly",degree=5,gamma="auto"


# In[24]:


model.fit(X_train,y_train)


# In[25]:


model.score(X_train,y_train)


# In[26]:


X.head()


# In[27]:


data=pd.DataFrame({"Pclass":[1],"Age":[25],"SibSp":[2],"Parch":[2],"Sex_male":[0]})
data


# In[28]:


scaled_data=scaleing.transform(data)
scaled_data


# In[29]:


model.predict(scaled_data)


# In[ ]:




