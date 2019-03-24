#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


grey = cv2.imread('../data/Lena.png', 0)
cv2.imshow('original grey', grey)
cv2.waitKey()
cv2.destroyAllWindows()


# In[3]:


hist, bins = np.histogram(grey, 256, [0, 255])


# In[5]:


plt.fill(hist)
plt.xlabel('pixel value')
plt.show()

