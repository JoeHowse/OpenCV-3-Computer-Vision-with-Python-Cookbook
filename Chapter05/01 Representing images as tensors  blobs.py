#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[5]:


image_bgr = cv2.imread('../data/Lena.png', cv2.IMREAD_COLOR)
print(image_bgr.shape)


# In[6]:


image_bgr_float = image_bgr.astype(np.float32)
image_rgb = image_bgr_float[..., ::-1]
tensor_chw = np.transpose(image_rgb, (2, 0, 1))
tensor_nchw = tensor_chw[np.newaxis, ...]

print(tensor_nchw.shape)


# In[ ]:




