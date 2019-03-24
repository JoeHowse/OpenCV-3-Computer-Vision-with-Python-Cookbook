#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'auto')


# In[2]:


img = cv2.imread('../data/Lena.png', cv2.IMREAD_GRAYSCALE)


# In[3]:


corners = cv2.goodFeaturesToTrack(img, 100, 0.05, 10)


# In[10]:


for c in corners:
    x, y = c[0]
    cv2.circle(img, (x, y), 5, 255, -1)
plt.figure(figsize=(10, 10))
plt.imshow(img, cmap='gray')
plt.tight_layout()
plt.show()


# In[ ]:




