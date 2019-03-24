#!/usr/bin/env python
# coding: utf-8

# In[20]:


import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'auto')


# In[25]:


image = cv2.imread('../data/Lena.png')


# In[44]:


KSIZE = 11
ALPHA = 2
kernel = cv2.getGaussianKernel(KSIZE, 0)
kernel = -ALPHA * kernel @ kernel.T
kernel[KSIZE//2, KSIZE//2] += 1 + ALPHA
print(kernel.shape, kernel.dtype, kernel.sum())


# In[45]:


filtered = cv2.filter2D(image, -1, kernel)


# In[51]:


plt.figure(figsize=(8,4))
plt.subplot(121)
plt.axis('off')
plt.title('image')
plt.imshow(image[:, :, [2, 1, 0]])
plt.subplot(122)
plt.axis('off')
plt.title('filtered')
plt.imshow(filtered[:, :, [2, 1, 0]])
plt.tight_layout(True)
plt.show()


# In[9]:


cv2.imshow('before', image)
cv2.imshow('after', filtered)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:




