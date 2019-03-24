#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


imgs_names = ['33', '100', '179', '892', '1560', '2933']

exp_times = []
images = []

for name in imgs_names:
    exp_times.append(1/float(name))
    images.append(cv2.imread('../data/hdr/%s.jpg' % name, cv2.IMREAD_COLOR))

exp_times = np.array(exp_times).astype(np.float32)


# In[18]:


calibrate = cv2.createCalibrateDebevec()
response = calibrate.process(images, exp_times)


# In[4]:


merge_debevec = cv2.createMergeDebevec()
hdr = merge_debevec.process(images, exp_times, response)


# In[14]:


tonemap = cv2.createTonemapDurand(2.4)
ldr = tonemap.process(hdr)

ldr = cv2.normalize(ldr, None, 0, 1, cv2.NORM_MINMAX)

cv2.imshow('ldr', ldr)
cv2.waitKey()
cv2.destroyAllWindows()


# In[16]:


merge_mertens = cv2.createMergeMertens()
fusion = merge_mertens.process(images)

fusion = cv2.normalize(fusion, None, 0, 1, cv2.NORM_MINMAX)

cv2.imshow('fusion', fusion)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:




