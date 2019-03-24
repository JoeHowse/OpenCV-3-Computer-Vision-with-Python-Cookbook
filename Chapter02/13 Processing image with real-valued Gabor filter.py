#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


image = cv2.imread('../data/Lena.png', 0).astype(np.float32) / 255


# In[3]:


kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, cv2.CV_32F)
kernel /= math.sqrt((kernel * kernel).sum())


# In[4]:


filtered = cv2.filter2D(image, -1, kernel)


# In[5]:


plt.figure(figsize=(8,3))
plt.subplot(131)
plt.axis('off')
plt.title('image')
plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.title('kernel')
plt.imshow(kernel, cmap='gray')
plt.subplot(133)
plt.axis('off')
plt.title('filtered')
plt.imshow(filtered, cmap='gray')
plt.tight_layout()
plt.show()


# In[ ]:




