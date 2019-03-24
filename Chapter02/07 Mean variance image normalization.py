#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


image = cv2.imread('../data/Lena.png').astype(np.float32) / 255


# In[3]:


image -= image.mean()
image /= image.std()


# In[ ]:




