#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import matplotlib.pyplot as plt


# In[3]:


#get_ipython().run_line_magic('matplotlib', 'auto')


# In[4]:


image = cv2.imread('../data/Lena.png')


# In[5]:


edges = cv2.Canny(image, 200, 100)


# In[6]:


plt.figure(figsize=(8,5))
plt.subplot(121)
plt.axis('off')
plt.title('original')
plt.imshow(image[:,:,[2,1,0]])
plt.subplot(122)
plt.axis('off')
plt.title('edges')
plt.imshow(edges, cmap='gray')
plt.tight_layout()
plt.show()


# In[ ]:




