#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[49]:


K = np.array([[560,0,320],[0,560,240],[0,0,1]],dtype=np.float32)
rvec = np.array([0.1, 0.2, 0.3], np.float32)
R, _ = cv2.Rodrigues(rvec)
print(R)


# In[50]:


H = K @ R @ np.linalg.inv(K)


# In[51]:


H /= H[2, 2]


# In[52]:


print(H)


# In[53]:


H += np.random.randn(3,3)*0.0001


# In[69]:


np.save('rotational_homography.npy', {'H': H, 'K': K})


# In[70]:


data = np.load('../data/rotational_homography.npy').item()
H, K = data['H'], data['K']


# In[71]:


H


# In[72]:


H_ = np.linalg.inv(K) @ H @ K


# In[75]:


w, u, vt = cv2.SVDecomp(H_)
R = u @ vt
if cv2.determinant(R) < 0:
    R *= 1


# In[76]:


rvec = cv2.Rodrigues(R)[0]


# In[77]:


print('Rotation vector:')
print(rvec)


# In[ ]:




