#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


def contours_pca(contours):
    # join all contours points into the single matrix and remove unit dimensions
    cnt_pts = np.vstack(contours).squeeze().astype(np.float32)

    mean, eigvec = cv2.PCACompute(cnt_pts, None)

    center = mean.squeeze().astype(np.int32)
    delta = (150*eigvec).astype(np.int32)
    return center, delta


# In[3]:


def draw_pca_results(image, contours, center, delta):
    cv2.drawContours(image, contours, -1, (255, 255, 0))

    cv2.line(image, tuple((center + delta[0])), 
                    tuple((center - delta[0])), 
                    (0, 255, 0), 2)

    cv2.line(image, tuple((center + delta[1])), 
                    tuple((center - delta[1])), 
                    (0, 0, 255), 2)

    cv2.circle(image, tuple(center), 20, (0, 255, 255), 2)


# In[4]:


cap = cv2.VideoCapture("../data/opencv_logo.mp4")

while True:
    status_cap, frame = cap.read()
    if not status_cap:
        break
    
    frame = cv2.resize(frame, (0, 0), frame, 0.5, 0.5)
    edges = cv2.Canny(frame, 250, 150)
    
    _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours):
        center, delta = contours_pca(contours)
        draw_pca_results(frame, contours, center, delta)
        
    cv2.imshow('PCA', frame)
    if cv2.waitKey(100) == 27:
        break

cv2.destroyAllWindows()


# In[ ]:




