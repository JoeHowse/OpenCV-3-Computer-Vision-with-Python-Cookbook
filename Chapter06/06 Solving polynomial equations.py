#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np


# In[3]:


N = 4
coeffs = np.random.randn(N+1,1)
retval, roots = cv2.solvePoly(coeffs)


# In[15]:


for i in range(N):
    print('Root', roots[i],'residual:', 
          np.abs(np.polyval(coeffs[::-1], roots[i][0][0]+1j*roots[i][0][1])))


# In[ ]:




