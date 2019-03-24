#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[75]:


data = np.load('../data/stereo/case1/stereo.npy').item()
Kl, Kr, Dl, Dr, left_pts, right_pts, E_from_stereo, F_from_stereo =     data['Kl'], data['Kr'], data['Dl'], data['Dr'],     data['left_pts'], data['right_pts'], data['E'], data['F']


# In[76]:


left_pts = np.vstack(left_pts)
right_pts = np.vstack(right_pts)


# In[77]:


left_pts = cv2.undistortPoints(left_pts, Kl, Dl, P=Kl)
right_pts = cv2.undistortPoints(right_pts, Kr, Dr, P=Kr)


# In[83]:


F, mask = cv2.findFundamentalMat(left_pts, right_pts, cv2.FM_LMEDS)


# In[79]:


F


# In[80]:


E = Kr.T @ F @ Kl


# In[84]:


print('Fundamental matrix:')
print(F)
print('Essential matrix:')
print(E)


# In[ ]:




