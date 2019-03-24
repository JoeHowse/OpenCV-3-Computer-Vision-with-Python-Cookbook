#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'auto')


# In[6]:


image = np.full((480, 640), 255, np.uint8)
cv2.circle(image, (320, 240), 100, 0)


# In[7]:


distmap = cv2.distanceTransform(image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


# In[9]:


plt.figure()
plt.imshow(distmap, cmap='gray')
plt.show()


# In[ ]:




