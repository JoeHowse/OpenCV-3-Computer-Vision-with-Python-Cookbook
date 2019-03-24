#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[106]:


m = 10
n = 10
constrains_mat = np.random.randn(m, n+1)
weights = np.random.randn(1, n)


# In[107]:


solution = np.array((n, 1), np.float32)
res = cv2.solveLP(weights, constrains_mat, solution)

if res == cv2.SOLVELP_SINGLE:
    print('The problem has the one solution')
elif res == cv2.SOLVELP_MULTI:
    print('The problem has the multiple solutions')
elif res == cv2.SOLVELP_UNBOUNDED:
    print('The solution is unbounded')
elif res == cv2.SOLVELP_UNFEASIBLE:
    print('The problem doesnt\'t have any solutions')


# In[ ]:




