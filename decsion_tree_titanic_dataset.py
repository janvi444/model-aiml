#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Importing our dataset from csv file
import pandas as pd
df=pd.read_csv("Titanic-Dataset.csv",)
#Preprocessing our data
df['Age'].fillna(df['Age'].mean(),inplace=True)
df.replace({'Sex':{'male': 1,'female':0}},inplace=True)
df['Cabin']=df.Cabin.fillna('G10')
df.replace({'Survived':{1:'Yes',0:'No'}},inplace=True)


# In[5]:


#importing relevant libraries
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#features extraction
x=df.drop(["Survived", "Name", "Ticket","Cabin", "Embarked"], axis=1)
y= df["Survived"]
#splitting data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25,random_state=0)


# In[6]:


model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train) #fitting our model
y_pred=model.predict(x_test) # evaluating our model
print("score:{}".format(accuracy_score(y_test, y_pred)))


# In[8]:


df.describe


# In[11]:


import matplotlib.pyplot as plt
from sklearn import tree

fig = plt.figure(figsize=(100, 100))
tree.plot_tree(model,
               feature_names=x.columns.values.tolist(),
               class_names=df.Survived.unique().tolist(),  # Convert array to list
               filled=True)
fig.savefig("tree1.png")


# In[ ]:




