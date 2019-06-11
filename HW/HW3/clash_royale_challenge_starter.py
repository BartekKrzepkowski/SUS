#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Load necessary packages

import os
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


DATA_PATH = "data"


# In[17]:


# Read data and present
train = pd.read_csv(os.path.join(DATA_PATH, 'trainingData.csv'))
valid = pd.read_csv(os.path.join(DATA_PATH, 'validationData.csv'))
train.head()


# In[18]:


valid.head()


# In[11]:


# Helper functions to preprocess data to bag-of-cards format

def unnest(df, col):
    unnested = (df.apply(lambda x: pd.Series(x[col]), axis=1)
                .stack()
                .reset_index(level=1, drop=True))
    unnested.name = col
    return df.drop(col, axis=1).join(unnested)

def to_bag_of_cards(df):
    df['ind'] = np.arange(df.shape[0]) + 1
    df_orig = df.copy()
    df['deck'] = df['deck'].apply(lambda d: d.split(';'))
    df = unnest(df, 'deck')
    df['value'] = 1
    df_bag = df.pivot(index='ind', columns='deck', values='value')
    df_bag[df_bag.isna()] = 0
    df_bag = df_bag.astype('int')
    return pd.concat([df_orig.set_index('ind'), df_bag], axis=1)


# In[12]:


train = to_bag_of_cards(train)
valid = to_bag_of_cards(valid)
train.head()


# In[83]:


valid.head()


# In[84]:


# Sort data by number of games played

train = train.sort_values('nofGames', ascending=False)
valid = valid.sort_values('nofGames', ascending=False)


# In[85]:


# Specify example model fitting function and R squared metric

from sklearn.svm import SVR

def R2(x, y):
    return 1 - np.sum(np.square(x - y)) / np.sum(np.square(y - np.mean(y)))

def fit_svm(data):
    svr = SVR(kernel='rbf', gamma=1.0/90, C=1.0, epsilon=0.02, shrinking=False)
    svr.fit(data.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate'], axis=1), data['winRate'])
    return svr

sizes = (np.arange(10) + 6) * 100


# In[86]:


# Fit and predict on models of various training sizes

fit_list = list(map(lambda size: fit_svm(train.iloc[1:size]), sizes))
pred_list = list(map(lambda fit: fit.predict(valid.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate'], axis=1)),
                     fit_list))


# In[89]:


# Calculate R squared scores

r2 = list(map(lambda p: R2(p, valid['winRate']), pred_list))
r2


# In[90]:


_ = plt.plot(sizes, r2)


# In[91]:


np.mean(r2)


# In[92]:


# Save hyperparameteres and selected indices in submission format

with open('example_sub_python.txt', 'a') as f:
    for size in sizes:
        ind_text = ','.join(list(map(str, train.index.values[:size])))
        text = ';'.join(['0.02', '1.0', str(1.0 / 90), ind_text])
        f.write(text + '\n')


# In[ ]:




