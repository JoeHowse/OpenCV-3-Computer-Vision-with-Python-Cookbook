#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'auto')


# In[7]:


model = cv2.dnn.readNetFromCaffe('../data/fcn8s-heavy-pascal.prototxt',
                                 '../data/fcn8s-heavy-pascal.caffemodel')


# In[3]:


frame = cv2.imread('../data/scenetext01.jpg')
blob = cv2.dnn.blobFromImage(frame, 1, (frame.shape[1],frame.shape[0]))
model.setInput(blob)
output = model.forward()
print(output.shape)


# In[4]:


labels = output[0].argmax(0)
print(labels.shape)


# In[9]:


plt.figure(figsize=(14,6))
plt.subplot(121)
plt.axis('off')
plt.title('original')
plt.imshow(frame[:,:,[2,1,0]])
plt.subplot(122)
plt.axis('off')
plt.title('segmentation')
plt.imshow(labels)
plt.tight_layout()
plt.show()


# In[ ]:




