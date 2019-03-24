#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2
import numpy as np


# In[2]:


net_caffe = cv2.dnn.readNetFromCaffe('../data/bvlc_googlenet.prototxt', 
                                     '../data/bvlc_googlenet.caffemodel')


# In[3]:


net_torch = cv2.dnn.readNetFromTorch('../data/torch_enet_model.net')


# In[4]:


net_tensorflow = cv2.dnn.readNetFromTensorflow('../data/tensorflow_inception_graph.pb')

