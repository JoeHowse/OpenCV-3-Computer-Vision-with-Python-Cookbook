#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


image = np.zeros((480, 640), np.uint8)
cv2.ellipse(image, (320, 240), (200, 100), 0, 0, 360, 255, -1)


# In[11]:


m = cv2.moments(image)
for name, val in m.items():
    print(name, '\t', val)


# In[31]:


print('Center X estimated:', m['m10'] / m['m00'])
print('Center Y estimated:', m['m01'] / m['m00'])


# In[ ]:




