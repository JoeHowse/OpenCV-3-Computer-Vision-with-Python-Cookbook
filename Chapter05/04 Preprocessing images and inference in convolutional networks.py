#!/usr/bin/env python
# coding: utf-8

# In[12]:


import cv2
import numpy as np


# In[18]:


image = cv2.imread('../data/Lena.png', cv2.IMREAD_COLOR)
tensor = cv2.dnn.blobFromImage(image, 1.0, (224, 224),
                               (104, 117, 123), False, False);


# In[19]:


tensor = cv2.dnn.blobFromImages([image, image], 1.0, (224, 224),
                                (104, 117, 123), False, True);


# In[15]:


net = cv2.dnn.readNetFromCaffe('../data/bvlc_googlenet.prototxt', 
                               '../data/bvlc_googlenet.caffemodel')


# In[16]:


net.setInput(tensor);
prob = net.forward();


# In[17]:


net.setInput(tensor, 'data');
prob = net.forward('prob');


# In[ ]:




