#!/usr/bin/env python
# coding: utf-8

# In[13]:


import cv2
import numpy as np

image = cv2.imread('../data/Lena.png')
print('Shape:', image.shape)
print('Data type:', image.dtype)

cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()


# In[4]:


image = image.astype(np.float32) / 255
print('Shape:', image.shape)
print('Data type:', image.dtype)


# In[ ]:


cv2.imshow('image', np.clip(image*2, 0, 1))
cv2.waitKey()
cv2.destroyAllWindows()


# In[6]:


image = (image * 255).astype(np.uint8)
print('Shape:', image.shape)
print('Data type:', image.dtype)

cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()

