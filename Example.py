
# coding: utf-8

# In[14]:


# JELSR Example
# use the famous iris dataset to show the functions constructed.
# d is number of features and n is number of observations.

import numpy as np
import JELSR 
import pandas as pd

df = pd.read_csv("iris.csv")

# First get all the features.

df_X = df.drop(["variety"],1)

# get the d by n matrix X

X = df_X.values.transpose()

# Get the W and Y matrix

W,Y = JELSR.JELSR(X,2)

# Feature selection

Selected = JELSR.feature_selection(X,2,2)
print("JELSR features selected are:", [df_X.columns[e] for e in Selected] )

