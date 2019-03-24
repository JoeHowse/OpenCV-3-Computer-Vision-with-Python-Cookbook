#!/usr/bin/env python
# coding: utf-8

# In[76]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[86]:


#get_ipython().run_line_magic('matplotlib', 'auto')


# In[87]:


img = np.zeros((500, 500), np.uint8)


# In[88]:


cv2.circle(img, (200, 200), 50, 255, 3)
cv2.line(img, (100, 400), (400, 350), 255, 3);


# In[89]:


circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 15, param1=200, param2=30)[0]


# In[91]:


lines = cv2.HoughLinesP(img, 1, np.pi/180, 100, 100, 10)[0]


# In[102]:


dbg_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)    
for x1, y1, x2, y2 in lines:
    print('Detected line: ({} {}) ({} {})'.format(x1, y1, x2, y2))
    cv2.line(dbg_img, (x1, y1), (x2, y2), (0, 255, 0), 2)    

for c in circles:
    print('Detected circle: center=({} {}), radius={}'.format(c[0], c[1], c[2]))
    cv2.circle(dbg_img, (c[0], c[1]), c[2], (0, 255, 0), 2)
    
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.title('original')
plt.axis('off')
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.title('detected primitives')
plt.axis('off')
plt.imshow(dbg_img)
plt.show()


# In[ ]:




