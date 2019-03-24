#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


image = cv2.imread('../data/Lena.png', 0).astype(np.float32) / 255


# In[3]:


gamma = 0.5
corrected_image = np.power(image, gamma)


# In[6]:


cv2.imshow('image', image)
cv2.imshow('corrected_image', corrected_image)
cv2.waitKey()


# In[21]:


cv2.imwrite('/tmp/image.png', image*255)
cv2.imwrite('/tmp/corrected_image.png', corrected_image*255)


# In[7]:


cv2.destroyAllWindows()


# In[ ]:




