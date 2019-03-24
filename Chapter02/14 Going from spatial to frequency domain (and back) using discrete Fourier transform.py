#!/usr/bin/env python
# coding: utf-8

# In[14]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'auto')


# In[9]:


image = cv2.imread('../data/Lena.png', 0).astype(np.float32) / 255


# In[24]:


fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
print('FFT shape:', fft.shape)
print('FFT data type:', fft.dtype)


# In[30]:


shifted = np.fft.fftshift(fft, axes=[0, 1])


# In[32]:


magnitude = cv2.magnitude(shifted[:,:,0], shifted[:,:,1])
print(magnitude.shape)
magnitude = np.log(magnitude)
# magnitude -= magnitude.min()
# magnitude /= magnitude.max()
print(magnitude.dtype)


# In[34]:


plt.axis('off')
plt.imshow(magnitude, cmap='gray')
plt.tight_layout(True)
plt.show()


# In[6]:


cv2.imshow('magnitude', magnitude)
cv2.waitKey()
cv2.destroyAllWindows()


# In[7]:


restored = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)


# In[18]:


cv2.imshow('restored', restored)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:




