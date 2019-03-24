#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2, numpy as np

cv2.namedWindow('window')

fill_val = np.array([255, 255, 255], np.uint8)

def trackbar_callback(idx, value):
    fill_val[idx] = value
    
cv2.createTrackbar('R', 'window', 255, 255, lambda v: trackbar_callback(2, v))
cv2.createTrackbar('G', 'window', 255, 255, lambda v: trackbar_callback(1, v))
cv2.createTrackbar('B', 'window', 255, 255, lambda v: trackbar_callback(0, v))

while True:
    image = np.full((500, 500, 3), fill_val)
    cv2.imshow('window', image)
    key = cv2.waitKey(3)
    if key == 27: 
        break

cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




