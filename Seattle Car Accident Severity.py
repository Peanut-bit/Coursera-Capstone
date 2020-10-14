#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

import datetime as dt

import warnings
warnings.filterwarnings('ignore')


# In[4]:


df=pd.read_csv('https://raw.githubusercontent.com/siv-26/Coursera_Capstone/master/Data-Collisions.csv')
df.head(5)


# In[5]:


df.info()


# ## 1. Select Parameters & Create Dataframe

# In[6]:


list(df.columns.values.tolist()) 


# In[7]:


data=df[['SEVERITYCODE', 'X','Y', 'LOCATION','ADDRTYPE', 'JUNCTIONTYPE', 'WEATHER', 'ROADCOND','LIGHTCOND']]
data.head()


# In[8]:


data.info()


# ## 2. Cleaning Dataset

# In[9]:


data['SEVERITYCODE'].value_counts()


# In[10]:


data['ADDRTYPE'].value_counts()


# In[11]:


data['JUNCTIONTYPE'].value_counts()


# In[12]:


data['WEATHER'].value_counts()


# In[13]:


data['ROADCOND'].value_counts()


# In[14]:


data['LIGHTCOND'].value_counts()


# In[15]:


print(data.isnull().sum(axis=0))


# In[16]:


newdata=data.dropna(how='any')


# In[17]:


print(newdata.isnull().sum(axis=0))


# In[18]:


newdata = newdata[newdata.JUNCTIONTYPE!= 'Unknown']


# In[19]:


newdata = newdata[newdata.WEATHER!= 'Unknown']


# In[20]:


newdata = newdata[newdata.ROADCOND!= 'Unknown']


# In[21]:


newdata = newdata[newdata.LIGHTCOND!= 'Unknown']


# In[22]:


newdata.info()


# ## 3. Data visualization and pre-processing

# In[24]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[25]:


import seaborn as sns

bins = np.linspace(newdata.SEVERITYCODE.min(), newdata.SEVERITYCODE.max(), 10)
g = sns.FacetGrid(newdata, col="ADDRTYPE", hue="LIGHTCOND", palette="Set1", col_wrap=5)
g.map(plt.hist, 'SEVERITYCODE', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[39]:


import folium


# In[40]:


# Seattle latitude and longitude values
latitude = 47.608013
longitude = -122.335167


# In[41]:


Seattle_map = folium.Map(location=[latitude, longitude], zoom_start=12)


# In[42]:


from folium import plugins

# instantiate a mark cluster object for the incidents in the dataframe
incidents = plugins.MarkerCluster().add_to(Seattle_map)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, in zip(newdata.Y, newdata.X, newdata.LOCATION):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(incidents)

# display map
Seattle_map


# ### Convert Categorical features to numerical values

# In[23]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


newdata.groupby(['ADDRTYPE'])['SEVERITYCODE'].value_counts(normalize=True)


# 73.18 % of accidents happend in Block has severity of 1 while 26.82% has severity of 2

# 56.17% of accidents happend in Block has severity of 1 while 43.83% has severity of 2 

# In[25]:


newdata.groupby(['WEATHER'])['SEVERITYCODE'].value_counts(normalize=True)


# ### Downsample

# In[26]:


from sklearn.utils import resample


# In[27]:


newdata['SEVERITYCODE'].value_counts()


# In[28]:


severity_majority = newdata[newdata['SEVERITYCODE'] == 1]
severity_minority = newdata[newdata['SEVERITYCODE'] == 2]


# In[29]:


severity_minority_downsampled = resample(severity_majority, n_samples=54690, replace=False, random_state=100)


# In[30]:


newdata_downsampled=pd.concat([severity_minority_downsampled,severity_minority])


# In[31]:


newdata_downsampled['SEVERITYCODE'].value_counts()


# In[32]:


Feature1 = pd.concat([pd.get_dummies(newdata_downsampled['WEATHER'])], axis=1)
Feature1.head(5)


# In[33]:


Feature2 = pd.concat([Feature1,pd.get_dummies(newdata_downsampled['JUNCTIONTYPE'])], axis=1)
Feature2.head(5)


# In[34]:


Feature3 = pd.concat([Feature2,pd.get_dummies(newdata_downsampled['ROADCOND'])], axis=1)
Feature3.head(5)


# In[35]:


Feature4 = pd.concat([Feature3,pd.get_dummies(newdata_downsampled['LIGHTCOND'])], axis=1)
Feature4.head(5)


# In[36]:


X = Feature4
X[0:5]


# In[37]:


y = newdata_downsampled['SEVERITYCODE'].values
y[0:5]


# In[38]:


X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:3]


# ## 4.Classification

# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# ### Support Vector Machine

# In[40]:


from sklearn import svm
SVM_model = svm.SVC(kernel='rbf')
SVM_model.fit(X_train, y_train) 


# In[41]:


yhat = SVM_model.predict(X_test)
yhat [0:5]


# In[42]:


from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted') 


# In[43]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)


# ### Logistic Regression

# In[44]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# In[45]:


yhat = LR.predict(X_test)
yhat


# In[46]:


yhat_prob = LR.predict_proba(X_test)
yhat_prob


# In[47]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)


# In[48]:


from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)


# In[49]:


from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted') 


# ### KNN Model

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
Ks=10
mean_acc=np.zeros((Ks-1))
std_acc=np.zeros((Ks-1))
ConfustionMx=[];
for n in range(1,Ks):
    
    #Train Model and Predict  
    kNN_model = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    yhat = kNN_model.predict(X_test)
    
    
    mean_acc[n-1]=np.mean(yhat==y_test);
    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
mean_acc

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[ ]:




