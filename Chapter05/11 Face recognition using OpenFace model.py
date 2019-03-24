#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np


# In[3]:


model = cv2.dnn.readNetFromCaffe('../data/face_detector/deploy.prototxt', 
                                 '../data/face_detector/res10_300x300_ssd_iter_140000.caffemodel')
CONF_THR = 0.5


# In[8]:


video = cv2.VideoCapture('../data/faces.mp4')
c = 0
while True:
    ret, frame = video.read()
    if not ret: break
        
    h, w = frame.shape[0:2]
    blob = cv2.dnn.blobFromImage(frame, 1, (300*w//h,300), (104,177,123), False)
    model.setInput(blob)
    output = model.forward()
    
    for i in range(output.shape[2]):
        conf = output[0,0,i,2]
        if conf > CONF_THR:
            label = output[0,0,i,1]
            x0,y0,x1,y1 = (output[0,0,i,3:7] * [w,h,w,h]).astype(int)
            cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0), 2)
            cv2.putText(frame, 'conf: {:.2f}'.format(conf), (x0,y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    c += 1
    if c == 60:
        cv2.imwrite('/home/alexeysp/projects/recipes/figures/ch5_face_detections.png', frame)
    
    cv2.imshow('frame', frame)
    key = cv2.waitKey(3)
    if key == 27: break
        
cv2.destroyAllWindows()


# In[ ]:




