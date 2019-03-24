#!/usr/bin/env python
# coding: utf-8

# In[14]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'auto')


# In[8]:


image = cv2.imread('../data/Lena.png', 0).astype(np.float32) / 255


# In[30]:


fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
print(fft.shape)
print(fft.dtype)


# In[31]:


fft_shift = np.fft.fftshift(fft, axes=[0, 1])
sz = 25
mask = np.zeros(fft.shape, np.uint8)
mask[image.shape[0]//2-sz:image.shape[0]//2+sz,
     image.shape[1]//2-sz:image.shape[1]//2+sz, :] = 1
fft_shift *= 1-mask
fft = np.fft.ifftshift(fft_shift, axes=[0, 1])


# In[32]:


filtered = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
print(filtered.shape)
print(filtered.dtype)


# In[33]:


plt.figure()
plt.subplot(121)
plt.axis('off')
plt.title('original')
plt.imshow(image, cmap='gray')
plt.subplot(122)
plt.axis('off')
plt.title('no high frequencies')
plt.imshow(filtered, cmap='gray')
plt.tight_layout(True)
plt.show()


# In[17]:


cv2.imshow('filtered', filtered)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:




