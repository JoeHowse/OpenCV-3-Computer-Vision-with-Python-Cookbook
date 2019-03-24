#!/usr/bin/env python
# coding: utf-8

# In[67]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'auto')


# In[28]:


img = cv2.imread('../data/Lena.png')


# In[29]:


noise = 30 * np.random.randn(*img.shape)
img = np.uint8(np.clip(img + noise, 0, 255))


# In[95]:


denoised_nlm = cv2.fastNlMeansDenoisingColored(img, None, 10)


# In[94]:


plt.figure(0, figsize=(10,6))
plt.subplot(121)
plt.axis('off')
plt.title('original')
plt.imshow(img[:,:,[2,1,0]])
plt.subplot(122)
plt.axis('off')
plt.title('denoised')
plt.imshow(denoised_nlm[:,:,[2,1,0]])
plt.show()


# In[ ]:




