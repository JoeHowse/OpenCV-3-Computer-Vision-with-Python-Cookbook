#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np


# In[3]:


A = np.random.randn(10, 10)


# In[4]:


w, u, v_t = cv2.SVDecomp(A)


# In[9]:


w


# In[5]:


np.linalg.matrix_rank(A)


# In[6]:


rank = 5


# In[7]:


w[rank:,0] = 0
B = u @ np.diag(w[:,0]) @ v_t


# In[8]:


print('Rank before:', np.linalg.matrix_rank(A))
print('Rank after:', np.linalg.matrix_rank(B))


# In[11]:


print('Norm before:', cv2.norm(A))
print('Norm after:', cv2.norm(B))


# In[ ]:




