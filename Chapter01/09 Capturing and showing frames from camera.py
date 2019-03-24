#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2

capture = cv2.VideoCapture(0)

while True:
    has_frame, frame = capture.read()
    if not has_frame:
        print('Can\'t get frame')
        break
        
    cv2.imshow('frame', frame)
    key = cv2.waitKey(3)
    if key == 27:
        print('Pressed Esc')
        break

capture.release()
cv2.destroyAllWindows()


# In[ ]:




