#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


# In[7]:


#get_ipython().run_line_magic('matplotlib', 'auto')


# In[13]:


cv2.namedWindow('frame')

for name, tracker in (('KCF', cv2.TrackerKCF_create), 
                      ('MIL', cv2.TrackerMIL_create), 
                      ('TLD', cv2.TrackerTLD_create)):
    tracker = tracker()
    initialized = False

    video = cv2.VideoCapture('../data/traffic.mp4')
    bbox = (878, 266, 1153-878, 475-266)

    i = 0
    while True:
        i += 1
        t0 = time.time()
        ok, frame = video.read()
        if not ok: 
            break

        if initialized:
            tracked, bbox = tracker.update(frame)
        else:
            cv2.imwrite('/tmp/frame.png', frame)
            tracked = tracker.init(frame, bbox)
            initialized = True

        fps = 1 / (time.time() - t0)
        cv2.putText(frame, 'tracker: {}, fps: {:.1f}'.format(name, fps),
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX,  3, (255, 0, 0), 4)            
        if tracked:
            bbox = tuple(map(int, bbox))
            cv2.rectangle(frame, (bbox[0], bbox[1]), 
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]), 
                          (0, 255, 0), 3)
        cv2.imshow('frame', frame)
        if i == 50:
            cv2.imwrite('frame{}.png'.format(name), frame)
        if cv2.waitKey(3) == 27:
            break

cv2.destroyAllWindows()


# In[ ]:




