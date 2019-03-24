#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


net = cv2.dnn.readNetFromCaffe('../data/bvlc_googlenet.prototxt', 
                               '../data/bvlc_googlenet.caffemodel')

if not net.empty():
    print('Net loaded successfully\n')
    
print('Net contains:')
for t in net.getLayerTypes():
    print('\t%d layers of type %s' % (net.getLayersCount(t), t))


# In[4]:


layers_ids, in_shapes, out_shapes = net.getLayersShapes([1, 3, 224, 224])

layers_names = net.getLayerNames()

print('Net layers shapes:')
for l in range(len(layers_names)):
    in_num, out_num = len(in_shapes[l]), len(out_shapes[l])
    print('Layer "%s" has %d input(s) and %d output(s)' 
          % (layers_names[l], in_num, out_num))
    for i in range(in_num):
        print('\tinput #%d has shape' % i, in_shapes[l][i].flatten())
    
    for i in range(out_num):
        print('\toutput #%d has shape' % i, out_shapes[l][i].flatten())


# In[ ]:




