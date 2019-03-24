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


grey_eq = cv2.equalizeHist(grey)


# In[4]:


hist, bins = np.histogram(grey_eq, 256, [0, 255])
plt.fill_between(range(256), hist, 0)
plt.xlabel('pixel value')
plt.show()


# In[5]:


cv2.imshow('equalized grey', grey_eq)
cv2.waitKey()
cv2.destroyAllWindows()


# In[6]:


color = cv2.imread('../data/Lena.png')
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)


# In[8]:


hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
color_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('original color', color)


# In[9]:


cv2.imshow('original color', color)


# In[10]:


cv2.imshow('equalized color', color_eq)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:




