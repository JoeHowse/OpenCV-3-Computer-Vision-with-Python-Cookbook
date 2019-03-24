#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2, numpy as np
image = cv2.imread('../data/Lena.png').astype(np.float32) / 255
print('Shape:', image.shape)


# In[2]:


image[:, :, [0, 2]] = image[:, :, [2, 0]]
cv2.imshow('blue_and_red_swapped', image)
cv2.waitKey()
cv2.destroyAllWindows()


# In[3]:


image[:, :, [0, 2]] = image[:, :, [2, 0]]
image[:, :, 0] =  (image[:, :, 0] * 0.9).clip(0, 1)                          
image[:, :, 1] =  (image[:, :, 1] * 1.1).clip(0, 1)
cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:




