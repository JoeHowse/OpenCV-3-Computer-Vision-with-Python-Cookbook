#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


data = np.load('../data/stereo/case1/stereo.npy').item()
E = data['E']


# In[3]:


R1, R2, T = cv2.decomposeEssentialMat(E)


# In[4]:


print('Rotation 1:')
print(R1)
print('Rotation 2:')
print(R2)
print('Translation:')
print(T)


# In[ ]:




