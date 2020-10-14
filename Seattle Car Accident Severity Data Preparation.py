#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

import datetime as dt

import warnings
warnings.filterwarnings('ignore')


# In[53]:


df=pd.read_csv('https://raw.githubusercontent.com/siv-26/Coursera_Capstone/master/Data-Collisions.csv')
df.head(5)


# In[54]:


df.info()


# ## 1. Select Parameters & Create Dataframe

# In[55]:


list(df.columns.values.tolist()) 


# In[56]:


data=df[['SEVERITYCODE', 'X','Y', 'LOCATION','ADDRTYPE', 'JUNCTIONTYPE', 'WEATHER', 'ROADCOND','LIGHTCOND']]
data.head()


# In[57]:


data.info()


# ## 2. Cleaning Dataset

# In[58]:


data['SEVERITYCODE'].value_counts()


# In[59]:


data['ADDRTYPE'].value_counts()


# In[61]:


data['JUNCTIONTYPE'].value_counts()


# In[62]:


data['WEATHER'].value_counts()


# In[63]:


data['ROADCOND'].value_counts()


# In[64]:


data['LIGHTCOND'].value_counts()


# In[65]:


print(data.isnull().sum(axis=0))


# In[66]:


newdata=data.dropna(how='any')


# In[67]:


print(newdata.isnull().sum(axis=0))


# In[68]:


newdata = newdata[newdata.JUNCTIONTYPE!= 'Unknown']


# In[69]:


newdata = newdata[newdata.WEATHER!= 'Unknown']


# In[70]:


newdata = newdata[newdata.ROADCOND!= 'Unknown']


# In[71]:


newdata = newdata[newdata.LIGHTCOND!= 'Unknown']


# In[72]:


newdata.info()


# ## 3. Data visualization and pre-processing

# In[73]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[92]:


import seaborn as sns

bins = np.linspace(newdata.SEVERITYCODE.min(), newdata.SEVERITYCODE.max(), 10)
g = sns.FacetGrid(newdata, col="ADDRTYPE", hue="LIGHTCOND", palette="Set1", col_wrap=5)
g.map(plt.hist, 'SEVERITYCODE', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[93]:


bins = np.linspace(newdata.SEVERITYCODE.min(), newdata.SEVERITYCODE.max(), 10)
g = sns.FacetGrid(newdata, col="ADDRTYPE", hue="JUNCTIONTYPE", palette="Set1", col_wrap=5)
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

# In[94]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[95]:


newdata.groupby(['ADDRTYPE'])['SEVERITYCODE'].value_counts(normalize=True)


# 73.18 % of accidents happend in Block has severity of 1 while 26.82% has severity of 2

# 56.17% of accidents happend in Block has severity of 1 while 43.83% has severity of 2 

# In[97]:


newdata.groupby(['WEATHER'])['SEVERITYCODE'].value_counts(normalize=True)


# In[155]:


Feature1 = pd.concat([pd.get_dummies(newdata['WEATHER'])], axis=1)
Feature1.head(5)


# In[157]:


Feature2 = pd.concat([Feature1,pd.get_dummies(newdata['JUNCTIONTYPE'])], axis=1)
Feature2.head(5)


# In[158]:


Feature3 = pd.concat([Feature2,pd.get_dummies(newdata['ROADCOND'])], axis=1)
Feature3.head(5)


# In[159]:


Feature4 = pd.concat([Feature3,pd.get_dummies(newdata['LIGHTCOND'])], axis=1)
Feature4.head(5)


# In[160]:


X = Feature4
X[0:5]


# In[162]:


Y = newdata['SEVERITYCODE'].values
Y[0:5]


# In[163]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# ## 4.Classification

# In[ ]:




