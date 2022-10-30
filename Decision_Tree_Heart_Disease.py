#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns


# In[3]:


hear_disease_df = pd.read_csv("/Users/ahmetokur/Desktop/Datasets/HeartDisease.csv")


# In[4]:


hear_disease_df


# In[5]:


hear_disease_df.info()


# In[6]:


hear_disease_df.isnull().sum()


# In[7]:


hear_disease_df.columns


# In[8]:


features = ['age', 'gender', 'chest_pain', 'rest_bps', 'cholestrol',
       'fasting_blood_sugar', 'rest_ecg', 'thalach', 'exer_angina', 'old_peak',
       'slope', 'ca', 'thalassemia']


# In[9]:


x = hear_disease_df[features]


# In[10]:


y = hear_disease_df.target


# In[11]:


x_train, x_test,y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[12]:


from sklearn.tree import DecisionTreeClassifier


# In[13]:


tree_classifier = DecisionTreeClassifier()


# In[14]:


tree_classifier.fit(x_train, y_train)


# In[15]:


tree_classifier.predict(x_test)
tree_classifier.score(x_test, y_test)# Accuracy score in classifications


# In[16]:


y_pred = tree_classifier.predict(x_test)
y_pred


# In[17]:


test_case = tree_classifier.predict([[45,1, 3, 146, 230, 1, 1, 180, 1, 2.2, 0, 0, 2]])


# In[18]:


test_case


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Evaluataing the Model

# In[20]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred, y_test)


# In[21]:


from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
confusion_matrix = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

confusion_matrix.plot()
plt.show()


# In[22]:


#ACCURACY
from sklearn.metrics import accuracy_score
accuracy_score(y_pred, y_test)


# In[27]:


#precision
from sklearn.metrics import precision_score
precision_score(y_pred, y_test) # average=None


# In[29]:


# Recall
from sklearn.metrics import recall_score
recall_score(y_pred, y_test, average=None)


# In[30]:


#f1 score
from sklearn.metrics import f1_score
f1_score(y_pred, y_test)


# In[ ]:




